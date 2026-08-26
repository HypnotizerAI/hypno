# 🌀 HYPNO

<p align="center">
  <strong>Run large language models on your own hardware.<br>No Python. No CUDA. No cloud. Just Rust and SIMD.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-linux%20|%20macos%20|%20windows-blue" alt="Platforms">
  <img src="https://img.shields.io/badge/rust-1.80%2B-orange" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

---

HYPNO is a from-scratch LLM inference engine. It takes HuggingFace model checkpoints, packs them into a compact `.hypno` binary, and runs them directly on your CPU — with hand-tuned SIMD kernels that wring every last cycle out of your silicon.

Not llama.cpp. Not ONNX Runtime. Not a Python wrapper. Pure Rust from the metal up.

---

## Why HYPNO

| Problem | HYPNO's answer |
|---|---|
| 4 GB PyTorch install for inference | Single binary under 10 MB |
| Python GIL bottlenecking throughput | Rust + rayon, all cores saturated |
| AVX-512 registers sitting idle | Runtime CPU feature detection, auto-dispatch |
| Model loading takes seconds | Memory-mapped — loads in **under a millisecond** |
| 16 GB VRAM required for 7B models | Q4_0 quantization: run 7B in ~4 GB RAM |
| Docker images with 200 layers | `COPY --from=builder` → ~50 MB final image |

---

## Performance

Every kernel is auto-dispatched at runtime based on your CPU's actual capabilities (AVX-512 → AVX2+FMA → SSE4.1 → scalar fallback). No compile-time feature flags needed.

### MatMul (4096 × 4096, 8 threads)

```
                          GFLOPS ▲
                           70 ┤                              ╔═════
                           60 ┤              ╔═════          ║ FP16
                           50 ┤  ╔═════      ║ Q4_0         ║ 68.8
                              ┤  ║ FP32      ║ 45.7         ╚═════
                           40 ┤  ║ 41.2      ╚═════
                              ┤  ╚═════              ╔═════
                           30 ┤                       ║ scalar FP32
                              ┤                       ║ ~3.3 GFLOPS
                           20 ┤                       ╚═════
                              ┤
                           10 ┤
                              └─────┬─────┬─────┬─────┬─────
```

| Kernel | Shape | GFLOPS | Memory BW | Latency | vs Scalar |
|---|---|---|---|---|---|
| MatMul FP32 | 4096×4096 | 41.2 | 82.4 GB/s | 0.81 ms | 12.5× |
| MatMul FP16 | 4096×4096 | 68.8 | 68.9 GB/s | 0.49 ms | 1.9× vs FP32 |
| **MatMul Q4_0** | **4096×4096** | **45.7** | **12.9 GB/s** | **0.73 ms** | **faster than FP32** |
| MatMul FP32 | 4096×14336 | 39.2 | 78.5 GB/s | 2.99 ms | 25.1× |
| MatMul FP16 | 4096×14336 | 65.1 | 65.1 GB/s | 1.80 ms | 2.0× vs FP32 |
| **MatMul Q4_0** | **4096×14336** | **46.4** | **13.1 GB/s** | **2.53 ms** | **faster than FP32** |

### Quantization accuracy

| Method | Compression | MAE |
|---|---|---|
| Q4_0 | 7.1× | 0.033 |
| Q8_0 | 2.0× | 0.001 |

Q4_0 compresses a 7B model from ~14 GB FP32 → ~2 GB on disk, with negligible quality loss.

### Q4_0 kernel evolution

```
Before:  4.3 GFLOPS ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (element-by-element loop)
After:  45.7 GFLOPS ██████████████████████████████████ (block dequantize + AVX2 FMA)
                    └────────── 10.7× faster ──────────┘
```

---

## Getting started

### Quick install

```bash
# Clone and build (takes ~30 seconds)
git clone https://github.com/HypnotizerAI/hypno.git
cd hypno
cargo build --release
```

### Convert a model

Point `hypno-convert` at any HuggingFace safetensors directory:

```bash
# Download a model first (example: TinyLlama)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./tinyllama

# FP32 — exact quality, full size
hypno-convert --model-dir ./tinyllama --out model.hypno

# Q4_0 — 7× smaller, still fast
hypno-convert --model-dir ./tinyllama --out model-q4.hypno --quantize Q4_0
```

### Chat

```bash
# One-shot completion
hypno-cli --model model.hypno --prompt "Explain quantum computing in one sentence" --max-tokens 128

# Interactive chat session
hypno-cli --model model.hypno --temperature 0.7 --top-p 0.9

# Pull a model and chat in one command
hypno-cli --pull TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quantize Q4_0
```

### Docker

```bash
docker compose up -d
docker compose exec hypno hypno-cli --model /models/model.hypno
```

Or build and run manually:

```bash
docker build -t hypno .
docker run -it -v $(pwd)/models:/models hypno --model /models/model.hypno
```

### Benchmark your machine

```bash
cargo run --release --bin hypno-bench
```

Outputs a JSON report to `/tmp/hypno_bench_results.json` with GFLOPS, bandwidth, and accuracy for every kernel.

---

## How it works

### The `.hypno` format

```
┌──────────────┬─────────────────┬──────────────────┬────────────────────┐
│ Header (16B) │ Metadata KVs    │ Tensor Table     │ Aligned Data       │
│ HYPN + v1    │ model config,   │ per-tensor:      │ 64-byte aligned    │
│              │ tokenizer JSON  │ name, shape,     │ payloads for       │
│              │                 │ dtype, offset    │ SIMD loads         │
└──────────────┴─────────────────┴──────────────────┴────────────────────┘
```

- Magic bytes `HYPN` + version (u32 LE)
- All tensor data 64-byte aligned — direct SIMD loads, no copies
- Small tensors (≤ 4096 elements) auto-stay FP32 to protect layernorms and biases
- Tokenizer vocabulary embedded directly in the file — no separate `tokenizer.json`

### Inference pipeline

```
.hypno file ──mmap──▶ zero-copy tensor access
                         │
                         ▼
              ┌─────────────────────┐
              │  Tokenizer (BPE)    │  prompt → token IDs
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Embedding lookup   │  token ID → vector
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Transformer layers │  RMSNorm → Attention (RoPE, GQA, KV cache)
              │  (× N layers)       │  → FFN (SiLU gate) → residual
              └─────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  LM head + sampling │  top-p / top-k / temperature
              └─────────────────────┘
                         │
                         ▼
                    decoded text
```

---

## Project structure

```
hypno/
├── crates/
│   ├── hypno-core/          # .hypno format, dtypes, quantization blocks
│   ├── hypno-convert/       # safetensors → .hypno converter
│   ├── hypno-loader/        # zero-copy mmap reader
│   ├── hypno-inference/     # transformer runtime + SIMD kernels
│   ├── hypno-quantize/      # Q4_0 / Q8_0 encode-decode
│   ├── hypno-tokenizer/     # BPE tokenizer
│   ├── hypno-cli/           # interactive chat CLI
│   └── hypno-bench/         # GFLOPS benchmark harness
├── tests/
│   └── e2e_test.py          # end-to-end: synthetic model → convert → infer
├── Dockerfile               # multi-stage, ~50 MB final image
├── docker-compose.yml       # one-command setup
└── README.md
```

---

## Supported platforms

| Platform | Architecture | SIMD | Status |
|---|---|---|---|
| **Linux** | x86_64 | AVX-512, AVX2, SSE4.1 | ✅ primary |
| **macOS** | x86_64 | AVX2, SSE4.1 | ✅ |
| **macOS** | ARM64 (Apple Silicon) | NEON | ✅ |
| **Windows** | x86_64 | AVX-512, AVX2, SSE4.1 | ✅ |

All platform detection and SIMD dispatch happens at runtime. One binary, every CPU.

---

## Requirements

- Rust 1.80+ (just `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Any x86_64 or ARM64 CPU
- That's it. No CUDA. No Python. No virtualenv.

---

## License

Apache 2.0 — see [LICENSE-APACHE](LICENSE-APACHE).
