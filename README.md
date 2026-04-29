# LLaMA Inference from Scratch & Bayesian Analysis (WIP)

This project implements a **LLaMA-style transformer from scratch in PyTorch**, with a focus on understanding:

- autoregressive inference
- attention mechanisms (grouped-query attention)
- rotary positional embeddings (RoPE)
- KV caching for efficient decoding

The long-term goal is to study whether transformer models can perform **implicit Bayesian inference**.

---

## Current Status

- Implemented core LLaMA architecture from scratch
- Successfully loaded **LLaMA 2 7B weights**
- Ran autoregressive inference on GPU (Tesla T4)
- Implemented token-by-token generation with KV caching
- Saving reproducible outputs for analysis

---

## Example Output

Prompt:
> What is the capital of France?

Output:
> Paris is the capital of France and the most populous city in France. It is situated on the River Seine, in northern France, at the heart of the Ile-de-France region. The city is a major European centre of finance, commerce, fashion, science, and ...

Full outputs are available in [`examples/sample_inference_output.json`](examples/sample_inference_output.json).

---

## How Inference Works

The pipeline:

1. Load model parameters (`params.json`)
2. Load pretrained weights (`consolidated.00.pth`)
3. Build transformer architecture from scratch
4. Tokenize input prompt
5. Generate tokens autoregressively:
   - forward pass
   - sampling (top-p)
   - KV cache reuse
6. Decode tokens to text

---

## Project Goal (Ongoing Work)

The objective is to explore whether transformer models can approximate **Bayesian inference**.

### Core Idea

Train or analyze a transformer on synthetic data generated from known distributions, and compare its predictions to:

- exact Bayesian posterior
- classical inference methods

---

## Planned Experiments

### 1. Function Inference

- Generate noisy data from known functions (e.g. sinusoidal, polynomial)
- Train or prompt the model to predict the function
- Compare predictions to:
  - ground truth
  - Bayesian posterior

---

### 2. Implicit Prior Analysis

Investigate whether:

- pretraining or prompt structure induces a **prior**
- model predictions change as more data is provided
- behavior matches Bayesian updating

---

### 3. Hidden State Analysis

- Measure how much information about true parameters is encoded
- Possible metric:
  - mutual information between hidden states and ground truth parameters

---

### 4. Transformer vs Bayesian Inference

Compare:

- transformer predictions
- analytical Bayesian solutions

Goal:
understand whether transformers approximate posterior inference or use a different mechanism.

---

## Future Directions

- Compare LLaMA-style architecture vs vanilla transformer
- Study geometry of hidden representations
- Analyze information flow across layers

---

## Run the Project

First: Install the necessary LLama weights from the meta official website: https://www.llama.com/llama-downloads/

```bash
pip install -r requirements.txt
python inference.py


---

## Acknowledgements

This implementation was developed following the excellent educational materials by **:contentReference[oaicite:0]{index=0}**, whose tutorials on building transformer architectures from scratch were instrumental in understanding and implementing the LLaMA model.

The pretrained model weights used in this project are provided by **:contentReference[oaicite:1]{index=1}** under the LLaMA 2 license.

This project builds upon these resources for educational and research purposes.
