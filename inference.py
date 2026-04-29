# once the building blocks of the modelare complete we go over the inference part

from typing import Optional
import torch
import time
from pathlib import Path
import json  # necessary to load the parameters
from sentencepiece import SentencePieceProcessor  # necessary to lead the tokenizer
from tqdm import tqdm
from datetime import datetime
import os

from model import ModelArgs, Transformer


class LLaMA:  # this will be the model

    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            # we load the checkpoints and save it into the cpu
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            # show the time it takes to load
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            # now we also load the parameters which is the json file
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )

        # we load the tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        # using the tokenizer we enhance the vocab size
        model_args.vocab_size = tokenizer.vocab_size()

        # whenever pytorch want to create a new tensor which one should it use?
        if device == "cuda":
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.float32)

        model = Transformer(model_args).to(device=device, dtype=torch.float16)

        # when we load a checkpoint, we load a set of keys and values, each key is a metric in the model, a weight of the linear layer or bias etc
        # we load model using strict = True so to match the name of variables in the file with the names defined in the classes
        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(
                f"Loaded state dict in {time.time() - prev_time:.2f}s"
            )  # how much time it took to load

        return LLaMA(model, tokenizer, model_args)

    # here we implement the method after which we choose the token, which is the top_P method
    # top_p: all token so that the cumulative prob is at least 0.9 (90%)
    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        # This is done by using the tokenizer
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]
        # Make sure the batch size for the prompt is not too large
        batch_size = len(prompt_tokens)
        assert (
            batch_size <= self.args.max_batch_size
        ), f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum sequence length
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device
        )
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            # replace padding token with prompt token (only for initial ones)
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = (
            tokens != pad_id
        )  # True if the token is a prompt token, False otherwise
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        # for loop for generating the tokens
        for cur_pos in cur_iterator:
            with torch.no_grad():
                # we pass one token at the time (the onw we want to output)
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied before the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token
            # replace padding tokens by infering the tokens only for the ones that we did not replace before
            next_token = torch.where(
                prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for a padding position that we want to inference
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            # we stop if we reach eos for all tokens
            if all(eos_reached):
                break

        # now prepare the output
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

    # now we have the output of the models, the logits. We transpose them into probabilities by using the softmax
    # but using these probabilities to select the tokens we use the top_P strategy that we now define
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        # keep track of which token we want to keep or not
        mask = probs_sum - probs_sort > p
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    # use cuda only if your model allows it
    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    """prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        "Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>",
        # Zero shot prompt
        "Tell me if the following person is actually Doraemon disguised as human:
        Name: Corte Riccardo
        Decision: 
        "",
    ]"""

    prompts = [
    "Tell me if the following person is actually Doraemon disguised as human: Name: Corte Riccardo Decision: ",
    "In simple terms, a trasformer model is",
    "What is the capital of France?"
    
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    # here we are actually inferencing the model
    # we want to generate at max 64 tokens
    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    """assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f"{out_texts[i]}")
        print("-" * 50)"""


    os.makedirs("outputs", exist_ok=True)

    save_path = f"outputs/inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = {
        "model": "LLaMA-2-7B",
        "device": str(model.args.device),
        "max_gen_len": 64,
        "num_prompts": len(prompts),
        "outputs": []
    }

    for prompt, output in zip(prompts, out_texts):
        results["outputs"].append({
            "prompt": prompt,
            "output": output,
        })

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {save_path}")