# Contributing to Hypno

Heck yes — you want to contribute. Here's how.

## Getting started

```bash
git clone https://github.com/HypnotizerAI/hypno.git
cd hypno
cargo build --release
cargo test
```

## What to work on

- **SIMD kernels** — squeeze more GFLOPS out of matmul, RMSNorm, attention
- **Quantization** — new dtypes (Q5_1, Q2_K), better accuracy-speed tradeoffs
- **Model support** — new architectures beyond LLaMA-style transformers
- **Sampling** — beam search, speculative decoding, Mirostat
- **Cross-platform** — ARM NEON, Apple Silicon AMX, WebAssembly SIMD

## Pull request process

1. Fork the repo
2. Create a branch (`git checkout -b cool-feature`)
3. Make your changes — include tests
4. `cargo test && cargo fmt && cargo clippy`
5. Push and open a PR
6. Keep PRs focused — one thing per PR

## Style

- Run `cargo fmt` before committing
- Run `cargo clippy` and fix warnings
- Use `unsafe` only in kernels — document why it's sound
- Prefer stack buffers over heap in hot paths
- Benchmarks go in `crates/hypno-bench/`

## License

By contributing, you agree your work will be licensed under Apache 2.0.
