# Supported Models

Hypno runs any model with a LLaMA-style architecture (RMSNorm, SiLU activation,
RoPE positional encoding, grouped-query attention). This covers virtually every
modern open-weight LLM.

All sizes assume Q4_0 quantization. RAM estimates assume 2K context with FP16 KV
cache and 2 active layers. tok/s estimated on 8-thread AVX-512 CPU.

## How to use

```bash
# Download from HuggingFace
hypno pull Qwen/Qwen2.5-3B

# Or convert a local GGUF
hypno convert model.gguf -o model.hypno --gguf

# Chat
hypno run --model model.hypno

# Serve API
hypno serve --model model.hypno --port 8080
```

---

## Llama Family (Meta)

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **Llama 3.2 1B** | 1.1B | 0.6 GB | ~0.5 GB | 80-120 | Fastest, great for testing |
| **Llama 3.2 3B** | 3.2B | 1.8 GB | ~0.7 GB | 40-55 | Best small Llama |
| **Llama 3.1 8B** | 8.0B | 4.5 GB | ~1.5 GB | 15-25 | Latest 8B |
| **Llama 3 8B** | 8.0B | 4.5 GB | ~1.5 GB | 15-25 | Solid general purpose |
| **Llama 3.1 70B** | 70B | 39 GB | ~8.5 GB | 2-4 | Needs lots of RAM |
| **Llama 2 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Classic, widely available |
| **Llama 2 13B** | 13B | 7.3 GB | ~2.2 GB | 10-15 | Good quality/size ratio |
| **CodeLlama 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Code generation |
| **CodeLlama 13B** | 13B | 7.3 GB | ~2.2 GB | 10-15 | Better code quality |
| **CodeLlama 34B** | 34B | 19 GB | ~4.5 GB | 4-6 | Best code Llama |

---

## Mistral Family

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **Mistral 7B v0.3** | 7.3B | 4.1 GB | ~1.3 GB | 20-30 | Strong general model |
| **Mistral Nemo 12B** | 12B | 6.7 GB | ~2.1 GB | 12-18 | Tekken tokenizer |
| **Mixtral 8×7B** | 46.7B | 26 GB | ~6 GB | 3-5 | MoE, ~12B active params |
| **Mixtral 8×22B** | 141B | 79 GB | ~15 GB | 1-2 | MoE, ~39B active params |
| **Mistral Small 22B** | 22B | 12 GB | ~3.5 GB | 6-10 | Dense, good quality |
| **Codestral 22B** | 22B | 12 GB | ~3.5 GB | 6-10 | Code-focused |

---

## Qwen Family (Alibaba)

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **Qwen 2.5 0.5B** | 0.5B | 0.3 GB | ~0.3 GB | 150-200 | Tiny, fast |
| **Qwen 2.5 1.5B** | 1.5B | 0.9 GB | ~0.5 GB | 70-100 | Great tiny model |
| **Qwen 2.5 3B** | 3.1B | 1.7 GB | ~0.7 GB | 40-60 | Code + multilingual |
| **Qwen 2.5 7B** | 7.6B | 4.3 GB | ~1.4 GB | 18-28 | Multilingual powerhouse |
| **Qwen 2.5 14B** | 14B | 7.9 GB | ~2.4 GB | 10-16 | Great quality |
| **Qwen 2.5 32B** | 32B | 18 GB | ~4.2 GB | 4-7 | Strong reasoning |
| **Qwen 2.5 72B** | 72B | 40 GB | ~8.8 GB | 2-4 | Best Qwen |
| **Qwen 2.5 Coder 7B** | 7.6B | 4.3 GB | ~1.4 GB | 18-28 | Code specialist |
| **Qwen 3 0.6B** | 0.6B | 0.4 GB | ~0.3 GB | 140-190 | Newest tiny |
| **Qwen 3 1.7B** | 1.7B | 1.0 GB | ~0.5 GB | 65-95 | Newest small |
| **Qwen 3 4B** | 4B | 2.2 GB | ~0.8 GB | 35-50 | Latest mid-size |
| **Qwen 3 8B** | 8B | 4.5 GB | ~1.5 GB | 15-25 | Latest 8B |

---

## Phi Family (Microsoft)

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **Phi-3 Mini 4K** | 3.8B | 2.1 GB | ~0.8 GB | 30-50 | Excellent reasoning |
| **Phi-3 Mini 128K** | 3.8B | 2.1 GB | ~2.5 GB | 25-40 | Long context variant |
| **Phi-3 Small 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Strong small model |
| **Phi-3 Medium 14B** | 14B | 7.9 GB | ~2.4 GB | 10-16 | Best quality/footprint |
| **Phi-3.5 Mini** | 3.8B | 2.1 GB | ~0.8 GB | 30-50 | Updated Phi-3 |
| **Phi-4 14B** | 14B | 7.9 GB | ~2.4 GB | 10-16 | Latest Phi |

---

## Gemma Family (Google)

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **Gemma 2 2B** | 2.6B | 1.5 GB | ~0.6 GB | 50-70 | Fast, high quality |
| **Gemma 2 9B** | 9.2B | 5.2 GB | ~1.6 GB | 14-22 | Strong 9B |
| **Gemma 2 27B** | 27B | 15 GB | ~3.8 GB | 5-8 | Best Gemma |
| **Gemma 3 1B** | 1B | 0.6 GB | ~0.4 GB | 85-130 | Latest tiny |
| **Gemma 3 4B** | 4B | 2.2 GB | ~0.8 GB | 35-50 | Latest small |
| **Gemma 3 12B** | 12B | 6.7 GB | ~2.1 GB | 12-18 | Latest mid |
| **Gemma 3 27B** | 27B | 15 GB | ~3.8 GB | 5-8 | Latest large |
| **CodeGemma 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Code specialist |

---

## DeepSeek Family

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **DeepSeek Coder 1.3B** | 1.3B | 0.7 GB | ~0.5 GB | 70-100 | Tiny coder |
| **DeepSeek Coder 6.7B** | 6.7B | 3.8 GB | ~1.2 GB | 20-30 | Excellent code |
| **DeepSeek Coder 33B** | 33B | 18 GB | ~4.5 GB | 4-7 | Best code quality |
| **DeepSeek V2 Lite** | 15.7B | 8.8 GB | ~2.6 GB | 8-14 | MoE, ~2.4B active |
| **DeepSeek V3** | 671B | — | — | — | MoE, ~37B active, needs 40+ GB |

---

## Other Notable Models

| Model | Params | Q4_0 Size | Est. RAM | tok/s | Notes |
|-------|--------|-----------|----------|-------|-------|
| **TinyLlama 1.1B** | 1.1B | 0.6 GB | ~0.5 GB | 80-120 | Classic tiny model |
| **SmolLM 135M** | 135M | 0.1 GB | ~0.1 GB | 500+ | Smallest usable LLM |
| **SmolLM 360M** | 360M | 0.2 GB | ~0.2 GB | 300-400 | Tiny but capable |
| **SmolLM 1.7B** | 1.7B | 1.0 GB | ~0.5 GB | 65-95 | Best sub-2B |
| **SmolLM2 1.7B** | 1.7B | 1.0 GB | ~0.5 GB | 65-95 | Updated SmolLM |
| **OLMo 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Fully open (data+code) |
| **OLMo 7B 0424** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Improved OLMo |
| **OLMoE 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | MoE OLMo, ~1B active |
| **OpenELM 270M** | 270M | 0.15 GB | ~0.2 GB | 350-450 | Tiny, efficient |
| **OpenELM 1.1B** | 1.1B | 0.6 GB | ~0.5 GB | 80-120 | Efficient 1B |
| **OpenELM 3B** | 3B | 1.7 GB | ~0.7 GB | 40-60 | Efficient 3B |
| **StableLM 3B** | 3B | 1.7 GB | ~0.7 GB | 40-55 | Stability AI |
| **StableLM Zephyr 3B** | 3B | 1.7 GB | ~0.7 GB | 40-55 | Chat-tuned |
| **StarCoder2 3B** | 3B | 1.7 GB | ~0.7 GB | 40-55 | Code generation |
| **StarCoder2 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | Better code |
| **StarCoder2 15B** | 15B | 8.4 GB | ~2.5 GB | 8-14 | Best StarCoder |
| **Command R 35B** | 35B | 20 GB | ~5 GB | 3-5 | Cohere, strong RAG |
| **Command R+ 104B** | 104B | 58 GB | ~12 GB | 1-2 | Cohere's biggest |
| **Falcon 7B** | 7B | 4.0 GB | ~1.3 GB | 20-30 | TII, older but solid |
| **Falcon 40B** | 40B | 22 GB | ~5.5 GB | 3-5 | TII large model |

---

## What fits in 2 GB RAM?

| Model | Size | Quality |
|-------|------|---------|
| ✅ SmolLM 135M-1.7B | 0.1-1.0 GB | Basic to decent |
| ✅ TinyLlama 1.1B | 0.6 GB | Decent for size |
| ✅ Qwen 2.5 0.5B-1.5B | 0.3-0.9 GB | Good tiny models |
| ✅ Llama 3.2 1B | 0.6 GB | Great 1B |
| ✅ Gemma 3 1B | 0.6 GB | Latest tiny |
| ✅ OpenELM 270M-1.1B | 0.15-0.6 GB | Efficient |
| ± Llama 3.2 3B / Qwen 2.5 3B | 1.7-1.8 GB | Tight fit, 30-50 tok/s |

## LLMs in 2 GB VRAM (CUDA/Metal)

The same Q4_0 models that fit CPU RAM also fit GPU VRAM — the math is identical.
For GPUs with 2 GB:
- **3B models** (1.7 GB) + 0.3 GB KV cache = perfect fit
- **Any 1B model** leaves plenty of headroom
- GPU inference can 2-5× faster than CPU for same model

---

## Notes

- All models use LLaMA-style architecture. Any model with `config.json` containing
  `"hidden_size"`, `"num_attention_heads"`, `"num_hidden_layers"` etc. should work.
- GGUF conversion supports Q4_0, Q8_0, F32, F16 directly. Other quant types
  (Q4_K, Q5_K, Q6_K, etc.) are dequantized to FP32 during conversion.
- LoRA adapters in PEFT format (`adapter_config.json` + `adapter_model.safetensors`)
  can be merged into base weights or converted standalone.
- RAM estimates include: model weights (Q4_0), FP16 KV cache (2K context), 2
  active layers, and scratch buffers. For exact numbers, benchmark with your model.
- tok/s estimates are for token generation only, not prompt prefill.
