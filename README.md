# LLaMA Bayesian Analysis

Work-in-progress implementation of LLaMA-style inference from scratch in PyTorch.

The current objective is to reproduce the core inference pipeline of a LLaMA 2-style transformer, including:

- RMSNorm
- Rotary positional embeddings
- Grouped-query attention
- KV cache for autoregressive decoding
- SwiGLU feed-forward layers
- Top-p sampling

The project is currently focused on running and validating inference. The next step is to compare generated outputs under different sampling settings and interpret the decoding process through a Bayesian/probabilistic lens.

## Status

Currently working on:

- cleaning the implementation
- validating inference on sample prompts
- documenting model components
- adding output comparisons for different decoding parameters
- Implementing a Bayesian Analysis to study if the behaviour of the inference model can be associated to a prior-likelihood-posterior metric

## Run inference

```bash
pip install -r requirements.txt
python inference.py
