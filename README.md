# LLaMA Inference from Scratch

This project implements a **LLaMA-style transformer from scratch in PyTorch** and runs
autoregressive inference with the official pretrained LLaMA 2 weights.

The focus is on a clean, readable reconstruction of:

- the LLaMA 2 architecture (RMSNorm, rotary positional embeddings, grouped-query attention,
  SwiGLU feed-forward)
- autoregressive decoding with KV caching
- top-p (nucleus) sampling

It is an implementation and inference study, there is no training loop.

---

## Status

- Core LLaMA architecture implemented from scratch
- Pretrained **LLaMA 2 7B** weights loaded successfully
- Autoregressive inference run on GPU (Tesla T4)
- Token-by-token generation with KV caching
- Reproducible outputs saved under `outputs/`

---

## Example Output

Prompt:
> What is the capital of France?

Output:
> Paris is the capital of France and the most populous city in France. It is situated on the River Seine, in northern France, at the heart of the Ile-de-France region. The city is a major European centre of finance, commerce, fashion, science, and ...

Full outputs are available in [`examples/sample_inference_output.json`](examples/sample_inference_output.json).

---

## How Inference Works

1. Load model parameters (`params.json`)
2. Load pretrained weights (`consolidated.00.pth`)
3. Build the transformer architecture from scratch
4. Tokenize the input prompt
5. Generate tokens autoregressively:
   - forward pass
   - top-p sampling
   - KV cache reuse
6. Decode tokens to text

---

## Run the Project

First, download the LLaMA 2 weights from Meta's official site:
<https://www.llama.com/llama-downloads/>

```bash
pip install -r requirements.txt
python inference.py
```

`inference.py` runs a set of example prompts (zero-shot, few-shot and a short translation
task) defined in `__main__`. Set `allow_cuda = True` there to run on GPU.

Model weights (`*.pth`, `*.bin`) are gitignored and are not part of this repository. Use of
the LLaMA 2 weights is subject to the Llama 2 Community License (`LICENSE`, `USE_POLICY.md`).

---

## Acknowledgements

Developed following the educational materials by **Umar Jamil**, whose tutorials on building
transformer architectures from scratch were instrumental in implementing the LLaMA model.
The pretrained weights are provided by **Meta** under the LLaMA 2 license.
