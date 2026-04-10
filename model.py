from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32   # Two types of heads: one for the Query (number of heads for Q)
    n_kv_heads: Optional[int] = None    # Second type of heads for the KV Cache (grouped wuery attentions)
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None  # These last two parameters represent the hidden dimension of the ffn layer
    # they try to keep the number of total parameters the same, even if we are working with a grouped query attention
    norm_eps: float = 1e-5

    # Needed for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # used as denominator to avoid dividion by zero
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) #RMS

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device:str, theta: float = 10000.0):
    # as written in the paper, the dimension  of the embedding must be even.
    assert head_dim % 2 == 0, " Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula: theta_i = 10000 ^ (-2(i-1)/dim) for i in arange(1, dim/2)
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    #shape: (Head_dim/2)
    theta = 1.0 /(theta ** (theta_numerator / head_dim)).to(device)
    # now we considers the "m's', possible position of a token, and can be many, as input we give the max_seq_length*2 (*2 because we also have th prompt that can be long)
    # construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device = device)
    # Now we multiply each m for each combination of theta using the outer product
    # Shape: (Seq_len) outer_product*(Head_Dim/2) -> (Seq_Len, Head_dim /2)
    freqs = torch.outer(m, theta).float()   # outer product with the desired property
    # Now we compute the complex numbers in the polar form: c = R* exp(i * m* theta), where R = 1 as follows:
    # Shape: (Seq_Len, Head/Dim / 2) -> (Seq_Len, Head_Dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

# here the technical mathematical part related to the manipulation of the vectors to improve the product efficiency
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # x is already a vector divided in its many heads dimensions
    # Unite the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number, the first dimension will be the real part, the second the imaginary one
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2) but complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number, divide between complex and real converting each part to real
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # stack the two dimensions to a single one, flattening it
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape) # we use the shape of the original tensor

    return x_out.type_as(x).to(device)  # so this is how we calculated the embeddings


# function used to repeat the keys and values a number of times so to match ne number of heads of the query in the grouped query attention mechanism
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)   # python automatically fills it if we use the expand function
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Compared to the true code here we have a simplification that removes parallelization

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        # as in the vanilla transformer we have the w matrices and no bias
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # we also store the values - create the cache - for the keys and values
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    # now the salient part where we introduce the forward method
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int, # pos of the token in the sentence
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim) we know that the sequence length is 1

        # we multiply the q,k,v for the corresponding w matrices
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        # note: this H_Q * Head_Dim is equal to Dim by definition, we are not changing the shape here, only the values
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        # Here we might change the dim because H_KV, the number of heads for KV may be smaller than Q (Grouped MuliQuery attention)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        # Same here
        xv = self.wv(x)
        
        # We divide each vector into its head for each query, key or value
        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # the next step is to apply the rotary position encoding, which won't change the size of the vector
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # now let's go over the kv-cache, here we caxhe k and v and append to the k and v
        # this allows then to compute the attention using all the k, all the v but only the latest q
        # Replace the entry in the cache for that particular position (start_pos) for every cache
        # we then estract all the keys and values to later compute the attention
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # the sequence length of the input is always one, but the sequence length of the cached kv, is equal to start position +1
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # the number of heads for k and v is not the same as the one for q in the grouped query attention
        # so what we do we siply repeat the k and v and apply those repeated one to a batch of queries
        # this part, computed in this way, can be improved
        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # the function repeat_kv repeats the keys a number of time = to n_rep which is the ratio between the q heads and kv heads
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # Here we apply the formula by multiplying everything
        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Now we apply the softmax
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # The output of the softmax is multiplied by the values
        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # So after concatenating (removing all the heads) so to have a single matrix, we multiply it by the wo matrix
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        # for the swiglu activation we have that the hidden size is calculated this way
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # we also have a multiplier
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    # Then we define the swish (silu) function
    def forward(self, x: torch.Tensor): # the implementation starts from the definition of the forward method
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


# The encoder block contains all the small blocks
class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        # the dimension of the head is the dimension of the vector divided by the number of heads
        self.head_dim = args.dim // args.n_heads    

        self.attention = SelfAttention(args)    # self attention block
        self.feed_forward = FeedForward(args)   # ff block

        # Normalization nr.1 BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # here we call the RMSNorm block we defined previously
        # Normalization nr.2 BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    # we define the forward method
    # start_pos is the position of the token we are dealing with
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        # we calculate the attention of the normalized version of the input x using also the frequencies for the rotary positional encodings
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        # here we have the application of the feedforward layer, that we apply after the normalization
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
# Begin with the skeleton of the whole model

class Transformer(nn.Module): # This is the model repeated Nx times (except of the softmax)
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers # number of layers of the model Nx = 32
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)   # convert inputs to embeddings

        self.layers = nn.ModuleList() #list of the  layers that we pass the embeddings to
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args)) # These are the block of which the model is composed of (norm, self attention, FF SwiGlu, etc)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)    # the last layer output is sent to the normalization
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        # we need to precompute the frequencies of the rotary positional encodings
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim// self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)
    

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # the sequence length that we want is always 1 due to kv cache
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        # This model is good for inference not for training due to the kv cache, since for training you need to be able to process multiple tokens
        # That is why we will be using the pretrained llama weights
        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens) #from tokens to embedding, note that Dim for this model is 4096

        # retrieve the pairs (m,theta) corresponding to the positions [ start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]   # we precompute something related to the positional encoding that we then pass ot the successive layer

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()

        return output