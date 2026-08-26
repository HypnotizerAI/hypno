# 🌀 Hypnotizer

**Run LLMs on your own metal. No Python, no CUDA, no bloat.**

Hypnotizer is a from-scratch inference engine for transformer models, written in pure Rust. It reads HuggingFace safetensors checkpoints, packs them into a compact `.hypno` binary format, and runs them on your CPU — with SIMD kernels that actually push silicon to its limit.

Not a wrapper around llama.cpp. Not another Python binding. Straight Rust from the metal up.

---

## Why

Because shipping a 4 GB PyTorch install to do basic inference is ridiculous. Because your laptop CPU has AVX-512 registers sitting idle while Python churns through memory. Because running an LLM should feel like opening a file, not launching a Jupyter notebook.

Hypnotizer loads models in under a millisecond (memory-mapped), runs quantized matmuls at FP32 speeds or faster, and fits in a single static binary.

---

## What's inside

| Crate | What it does |
|---|---|
| `hypntz-core` | `.hypno` binary format, dtype system, quantization block types |
| `hypntz-convert` | Converts HuggingFace safetensors → `.hypno` (with optional quantization) |
| `hypntz-loader` | Zero-copy mmap reader, direct tensor pointers, sub-ms loads |
| `hypntz-inference` | Transformer runtime with AVX-512/AVX2/SSE4.1 auto-dispatch SIMD kernels |
| `hypntz-quantize` | Q4_0 and Q8_0 quantization with vectorized dequantization |
| `hypntz-tokenizer` | BPE tokenizer from HuggingFace `tokenizer.json` |
| `hypntz-cli` | Interactive chat CLI with top-p/top-k sampling |
| `hypntz-bench` | GFLOPS benchmark harness |

---

## Performance

On an AVX-512 capable machine (8 threads), 4096×4096 matmul:

| Dtype | GFLOPS | Memory BW | Latency |
|---|---|---|---|
| FP32 | 41.2 | 82.4 GB/s | 0.81 ms |
| FP16 | 68.8 | 68.9 GB/s | 0.49 ms |
| **Q4_0** | **45.7** | **12.9 GB/s** | **0.73 ms** |

Q4_0 is faster than FP32 while reading 6.4× less memory. Compression is 7.1× with MAE of 0.033.

---

## Quick start

### Build

```bash
cargo build --release
```

### Convert a HuggingFace model

```bash
# FP32
hypntz-convert --model-dir ./llama-2-7b --out model.hypno

# Q4_0 quantized (7× smaller)
hypntz-convert --model-dir ./llama-2-7b --out model-q4.hypno --quantize Q4_0
```

### Run it

```bash
# One-shot
hypnotizer-cli --model model.hypno --prompt "Once upon a time"

# Interactive chat
hypnotizer-cli --model model.hypno
```

### Benchmark

```bash
cargo run --release --bin hypntz-bench
```

---

## The `.hypno` format

A dead-simple binary layout:

```
[HypnoHeader 16B] → [Metadata KVs] → [Tensor Table] → [Aligned Data]
```

- Magic: `HYPN` + version 1 (u32 LE)
- Tensors are 64-byte aligned for SIMD loads
- Supports FP32, FP16, Q8_0, Q4_0 dtypes
- Small tensors (≤4096 elements) stay FP32 to protect layernorms and biases

---

## Requirements

- Rust 1.80+
- x86_64 CPU with AVX2 (AVX-512 gets you the best numbers)
- No GPU, no CUDA, no Python, no virtualenv

---

## License

MIT
