# 🌀 Hypno

<p align="center">
  <strong>Run LLMs on your machine. No Python. No CUDA. Just Rust.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-linux%20|%20macos%20|%20windows-blue" alt="Platforms">
  <img src="https://img.shields.io/badge/rust-1.80%2B-orange" alt="Rust">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
</p>

---

Hypno is a pure-Rust inference engine for large language models. One binary does
everything: download models from HuggingFace, convert GGUF/safetensors checkpoints,
chat interactively, and serve an OpenAI-compatible API.

No PyTorch. No ONNX. No llama.cpp bindings. Just one static binary that loads
models in under a millisecond and saturates every core.

```bash
hypno pull Qwen/Qwen2.5-3B          # download from HuggingFace
hypno run --model model.hypno        # chat
hypno serve --model model.hypno      # API server on :8080
hypno convert model.gguf --gguf      # convert GGUF → .hypno
hypno bench                          # kernel benchmarks
```

---

## Why

Because `pip install torch` shouldn't be a prerequisite for running a
transformer. Because AVX-512 registers have been sitting unused since 2017.
Because reading a file from disk should not take 3 seconds when `mmap` exists.

Hypno is built around a few sharp edges:

| Edge | Payoff |
|---|---|
| Memory-mapped model files | Loads in **&lt;1 ms** — the OS pages in what you need |
| Runtime CPU feature detection | One binary, auto-dispatching AVX-512 → AVX2 → SSE4.1 → scalar |
| Q4_0 quantization with SIMD dequantize | 7.1× smaller on disk, **faster matmul than FP32** |
| Single static binary | ~15 MB, one `cargo build --release` |
| LoRA merging built into the converter | `--lora-dir` merges adapters into base weights before quantizing |

---

## Performance

Run `hypno bench` to see numbers for your machine. Here's what we get on an
AVX-512 CPU with 8 threads:

<p align="center">
  <img src="benchmarks/matmul_gflops.svg" alt="MatMul GFLOPS by dtype" width="600">
  <img src="benchmarks/bandwidth.svg" alt="Memory bandwidth comparison" width="600">
</p>

### Quantization

| Method | Bits/element | Compression | MAE (accuracy loss) |
|---|---|---|---|
| FP32 | 32 | 1.0× | 0 (reference) |
| FP16 | 16 | 2.0× | &lt; 0.001 |
| Q8_0 | 8.5 | ~3.8× | ~0.001 |
| **Q4_0** | **4.5** | **7.1×** | **0.033** |

A 7B model goes from ~14 GB FP32 → ~2 GB Q4_0 on disk. You can run it on a laptop.

### Q4_0 kernel evolution

<p align="center">
  <img src="benchmarks/q4_0_speedup.svg" alt="Q4_0 before vs after" width="600">
</p>

The original element-by-element loop walked every nibble and checked `if row ==
current_row` — 128× redundant work per row. The rewrite dequantizes each block
once into a stack buffer and computes the dot product with 4 × 8-wide AVX2 FMA.
Zero per-element branching.

---

## Quick start

```bash
git clone https://github.com/HypnotizerAI/hypno.git
cd hypno
cargo build --release
```

You now have a single `hypno` binary in `target/release/`.

### Pull a model from HuggingFace

```bash
# TinyLlama — great for testing
hypno pull TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Qwen 2.5 3B — multilingual, strong
hypno pull Qwen/Qwen2.5-3B

# With custom quantization and output path
hypno pull mistralai/Mistral-7B-Instruct-v0.3 --quantize Q4_0 --out mistral.hypno
```

### Convert a model from disk

```bash
# Standard safetensors checkpoint
hypno convert --model-dir ./model --out model.hypno

# With quantization
hypno convert --model-dir ./model --out model-q4.hypno --quantize Q4_0

# Convert from GGUF (llama.cpp format)
hypno convert model.gguf -o model.hypno --gguf

# Validate the output
hypno convert --model-dir ./model --out model.hypno --validate
```

### Merge a LoRA adapter

```bash
# Convert just the adapter (standalone)
hypno convert --lora-only --lora-dir ./my-fine-tune --out adapter.hypno

# Merge LoRA into base model, then quantize
hypno convert \
    --model-dir ./llama-2-7b \
    --lora-dir ./my-fine-tune \
    --quantize Q4_0 \
    --out llama-merged-q4.hypno
```

### Chat

```bash
# One-shot generation
hypno run --model model.hypno --prompt "Explain monads" --max-tokens 128

# Interactive session with temperature and top-p
hypno run --model model.hypno --temperature 0.7 --top-p 0.95

# Non-interactive mode (good for piping)
echo "Write a haiku about Rust" | hypno run --model model.hypno --prompt -
```

### Serve an API

```bash
hypno serve --model model.hypno --port 8080
```

Then use any OpenAI-compatible client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello!"}],"stream":false}'

curl http://localhost:8080/v1/models
curl http://localhost:8080/health
```

### Benchmark

```bash
hypno bench
# Outputs to /tmp/hypno_bench_results.json
```

---

## Commands reference

| Command | What it does |
|---|---|
| `hypno pull <HUGGINGFACE_ID>` | Download model from HuggingFace Hub, auto-convert to .hypno |
| `hypno convert [OPTIONS]` | Convert safetensors, GGUF, or LoRA adapters to .hypno |
| `hypno run --model <PATH>` | Interactive chat or single-generation inference |
| `hypno serve --model <PATH>` | OpenAI-compatible HTTP API server (`/v1/chat/completions`) |
| `hypno bench` | Kernel benchmarks: MatMul (FP32/FP16/Q4_0), RMSNorm, Softmax |

---

## Supported models

Hypno runs any model with a LLaMA-style architecture (RMSNorm, SiLU, RoPE, GQA).
That's virtually every modern open-weight LLM — see [MODELS.md](MODELS.md) for a
list of 65+ verified models with Q4_0 sizes, RAM estimates, and tok/s projections.

A few highlights:

| Model | Q4_0 Size | Est. RAM | tok/s |
|---|---|---|---|
| SmolLM 135M | 0.1 GB | ~0.1 GB | 500+ |
| TinyLlama 1.1B | 0.6 GB | ~0.5 GB | 80-120 |
| Qwen 2.5 3B | 1.7 GB | ~0.7 GB | 40-60 |
| Llama 3.2 3B | 1.8 GB | ~0.7 GB | 40-55 |
| Mistral 7B | 4.0 GB | ~1.3 GB | 20-30 |
| Gemma 2 9B | 5.2 GB | ~1.6 GB | 14-22 |

---

## How it works

### The `.hypno` format

```
┌──────────────┬─────────────────┬──────────────────┬──────────────────┐
│ Header (16B) │ Metadata KVs    │ Tensor Table     │ Aligned Data     │
│ HYPN + v1    │ config, vocab,  │ per-tensor:      │ all payloads at  │
│              │ tokenizer JSON  │ name, shape,     │ 64-byte offsets  │
│              │                 │ dtype, offset    │ for direct loads │
└──────────────┴─────────────────┴──────────────────┴──────────────────┘
```

- Magic bytes `HYPN` + version u32 LE
- Tensor data is 64-byte aligned — load directly into `__m256`/`__m512` registers, no copies
- Small tensors (&le;4096 elements, mostly layernorms and biases) auto-stay FP32 even when quantizing
- Tokenizer vocabulary is embedded — no separate `tokenizer.json` at runtime

### Inference pipeline

```
.hypno file ──mmap──▶ zero-copy tensor access      (hypno-loader)
                           │
                    ┌──────▼──────┐
                    │  Tokenizer  │                 (hypno-tokenizer)
                    │  BPE encode │  prompt → [token IDs]
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Embedding  │                 (hypno-inference)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │  Transformer × N layers │
              │  RMSNorm → Attention    │
              │    (RoPE, GQA, KV cache)│
              │  → FFN (SiLU gate)      │
              │  → residual add         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  LM head    │
                    │  + sampling │  top-p / top-k / temperature
                    └──────┬──────┘
                           │
                    decoded text
```

### SIMD dispatch

Every kernel checks CPUID at startup and picks the best path:

```rust
pub fn matmul_f32(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    if cpu_features().avx512f {
        unsafe { matmul_f32_avx512(y, w, x, bias, n, m) }
    } else if cpu_features().avx2 && cpu_features().fma {
        unsafe { matmul_f32_avx2(y, w, x, bias, n, m) }
    } else if cpu_features().sse4_1 {
        unsafe { matmul_f32_sse41(y, w, x, bias, n, m) }
    } else {
        matmul_f32_scalar(y, w, x, bias, n, m)
    }
}
```

One binary, every x86_64 CPU from 2013 onward. ARM NEON on Apple Silicon works the same way.

---

## Project structure

```
hypno/
├── crates/
│   ├── hypno/               # unified binary: run, serve, convert, pull, bench
│   ├── hypno-core/          # .hypno binary format + dtypes
│   ├── hypno-convert/       # safetensors → .hypno + GGUF + LoRA merge
│   ├── hypno-loader/        # zero-copy mmap reader
│   ├── hypno-inference/     # transformer + SIMD kernels
│   ├── hypno-quantize/      # Q4_0/Q8_0 encode/decode
│   ├── hypno-tokenizer/     # BPE from tokenizer.json
│   └── hypno-pull/          # HuggingFace download + auto-convert
├── benchmarks/              # SVG performance charts
├── MODELS.md                # 65+ supported model registry
├── Dockerfile               # multi-stage, ~50 MB image
├── docker-compose.yml
└── README.md
```

---

## Platforms

| OS | Architecture | SIMD path | Status |
|---|---|---|---|
| **Linux** | x86_64 | AVX-512, AVX2, SSE4.1 | ✅ primary |
| **macOS** | x86_64 | AVX2, SSE4.1 | ✅ |
| **macOS** | ARM64 (Apple Silicon) | NEON | ✅ |
| **Windows** | x86_64 | AVX-512, AVX2, SSE4.1 | ✅ |

---

## Requirements

- Rust 1.80+
- Any x86_64 or ARM64 CPU

That's the whole list.

---

## License

[Apache 2.0](LICENSE-APACHE)
