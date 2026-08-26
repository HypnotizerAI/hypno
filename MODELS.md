# Supported Models

Hypno runs any HuggingFace safetensors model with a LLaMA-style architecture (RMSNorm, SiLU, RoPE, GQA). Here are 10 tested and recommended models:

## Tier 1 — Verified (end-to-end tested)

| # | Model | Params | Q4_0 Size | RAM (FP16 KV) | tok/s (8T AVX-512) | Use case |
|---|---|---|---|---|---|---|
| 1 | **TinyLlama 1.1B Chat** | 1.1B | 0.6 GB | ~0.5 GB | 80-120 | Fast local chat, testing |
| 2 | **Qwen 2.5 3B** | 3B | 1.7 GB | ~0.7 GB | 40-60 | Code, multilingual |
| 3 | **Phi-3 Mini 4K** | 3.8B | 2.1 GB | ~0.8 GB | 30-50 | Reasoning, small footprint |
| 4 | **Llama 3.2 3B** | 3B | 1.7 GB | ~0.7 GB | 40-55 | General purpose |
| 5 | **Gemma 2 2B** | 2B | 1.1 GB | ~0.5 GB | 50-70 | Fast, high quality |

## Tier 2 — Compatible (architecture match confirmed)

| # | Model | Params | Q4_0 Size | RAM (FP16 KV) | tok/s (8T AVX-512) | Use case |
|---|---|---|---|---|---|---|
| 6 | **Mistral 7B v0.3** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Strong general model |
| 7 | **Llama 3.1 8B** | 8B | 4.5 GB | ~1.5 GB | 15-25 | Latest Llama |
| 8 | **Qwen 2.5 7B** | 7B | 4.0 GB | ~1.3 GB | 18-28 | Multilingual powerhouse |
| 9 | **DeepSeek Coder 6.7B** | 6.7B | 3.8 GB | ~1.2 GB | 20-30 | Code generation |
| 10 | **Phi-3 Medium 14B** | 14B | 7.9 GB | ~2.5 GB | 8-12 | Best quality/footprint ratio |

## Quick start

```bash
# Download any model from HuggingFace
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./tinyllama

# Convert to .hypno (Q4_0 for small RAM)
hypno-convert --model-dir ./tinyllama --out tinyllama.hypno --quantize Q4_0

# Chat
hypno-cli --model tinyllama.hypno

# Or serve via API
hypno-server --model tinyllama.hypno --port 8080
```

## GGUF models

Download GGUF files from HuggingFace (TheBloke, Bartowski, etc.) and convert:

```bash
hypno-convert --model-dir ./llama-3.2-3b-q4_k_m.gguf --out llama3.hypno --gguf --quantize FP32
```

## Notes

- All models listed use a LLaMA-style architecture with RMSNorm, SiLU activation, RoPE, and grouped-query attention (GQA)
- Q4_0 sizes are approximate — actual size depends on vocabulary size and number of tensors
- RAM estimates assume 2K context with FP16 KV cache and Q4_0 weights. Double the numbers for FP32 KV cache
- tok/s estimates are for token generation (not prompt processing) on an 8-thread AVX-512 CPU. Your numbers will vary
- GGUF conversion supports Q4_0, Q8_0, F32, F16 formats directly. Other GGUF quant types are dequantized to FP32 during conversion
