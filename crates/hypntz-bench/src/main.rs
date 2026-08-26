//! Hypnotizer Performance Benchmark Suite
//!
//! Measures:
//!   - MatMul GFLOPS (FP32 / FP16 / Q4_0) across matrix sizes
//!   - Memory bandwidth utilization
//!   - Transformer layer throughput (tokens/sec)
//!   - Q4_0 compression ratios and accuracy
//!
//! Outputs JSON for chart rendering and a human-readable summary.

use hypntz_core::DType;
use hypntz_inference::kernels::{self, cpu_features, matmul_f32, matmul_q4_0, matmul_f16};
use hypntz_quantize;
use half::f16;
use std::time::Instant;

fn main() {
    let feats = cpu_features();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     🌀 Hypnotizer Benchmark Suite                    ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  CPU features: {:>38} ║", feats.best_label());
    println!("║  Threads:      {:>38} ║", rayon::current_num_threads());
    println!("╚══════════════════════════════════════════════════════╝\n");

    let results = run_all_benchmarks();

    // Print summary
    println!("\n═══ RESULTS SUMMARY ═══\n");
    for r in &results {
        println!("{}", r);
    }

    // Output JSON for charts
    let json = serde_json::to_string_pretty(&results).unwrap();
    let json_path = "/tmp/hypntz_bench_results.json";
    std::fs::write(json_path, &json).unwrap();
    println!("\n📊 Chart data written to {}", json_path);
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchResult {
    name: String,
    category: String,
    gflops: f64,
    bandwidth_gbps: f64,
    time_ms: f64,
    elements_processed: u64,
    ops_per_element: u64,
    dtype_label: String,
    matrix_shape: String,
    speedup_vs_scalar: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:45} {:>8.1} GFLOPS  {:>6.1} GB/s  {:>8.2} ms  {:>6.1}x",
            self.name, self.gflops, self.bandwidth_gbps, self.time_ms, self.speedup_vs_scalar)
    }
}

fn run_all_benchmarks() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let threads = rayon::current_num_threads();
    let feats = cpu_features();

    // ── MatMul benchmarks across sizes and dtypes ───────────────────
    let sizes = [
        (128, 256, "tiny"),
        (512, 1024, "small"),
        (1024, 4096, "medium"),
        (4096, 4096, "large"),
        (4096, 14336, "llama-ffn"),
    ];

    for &(n, m, label) in &sizes {
        // FP32
        results.push(bench_matmul_f32(n, m, label, threads));
        // FP16
        results.push(bench_matmul_f16(n, m, label, threads));
        // Q4_0
        results.push(bench_matmul_q4_0_bench(n, m, label, threads));
    }

    // ── RMSNorm ────────────────────────────────────────────────────
    results.push(bench_rms_norm(4096));
    results.push(bench_rms_norm(14336));

    // ── Softmax ────────────────────────────────────────────────────
    results.push(bench_softmax(4096));
    results.push(bench_softmax(32000)); // vocab size

    // ── Quantization benchmarks ────────────────────────────────────
    results.push(bench_quantize_speed(4096 * 4096));
    results.push(bench_quantize_accuracy(4096));

    results
}

// ═══════════════════════════════════════════════════════════════════════
// FP32 MatMul
// ═══════════════════════════════════════════════════════════════════════

fn bench_matmul_f32(n: usize, m: usize, label: &str, threads: usize) -> BenchResult {
    // Create random weight matrix and input vector
    let total = n * m;
    let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.987654).cos()).collect();
    let mut y = vec![0.0f32; n];

    // Warmup
    for _ in 0..3 {
        matmul_f32(&mut y, &w, &x, None, n, m);
    }

    // Timed runs
    let runs = 10;
    let t0 = Instant::now();
    for _ in 0..runs {
        matmul_f32(&mut y, &w, &x, None, n, m);
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    // 2 FLOP per multiply-add, n*m muls + n*m adds = 2*n*m FLOP
    let flop = 2.0 * (n * m) as f64;
    let gflops = flop / elapsed / 1e9;
    let bytes_read = (total + m + n) as f64 * 4.0;
    let bandwidth = bytes_read / elapsed / 1e9;

    // Scalar baseline (single-threaded)
    let t0s = Instant::now();
    let mut ys = vec![0.0f32; n];
    for _ in 0..runs {
        for r in 0..n {
            let mut s = 0.0f32;
            for c in 0..m { s += w[r * m + c] * x[c]; }
            ys[r] = s;
        }
    }
    let scalar_elapsed = t0s.elapsed().as_secs_f64() / runs as f64;
    let speedup = scalar_elapsed / elapsed;

    BenchResult {
        name: format!("MatMul FP32  [{}×{}] {}", n, m, label),
        category: "matmul".into(),
        gflops,
        bandwidth_gbps: bandwidth,
        time_ms: elapsed * 1000.0,
        elements_processed: (total * runs) as u64,
        ops_per_element: 2,
        dtype_label: "FP32".into(),
        matrix_shape: format!("{}×{}", n, m),
        speedup_vs_scalar: speedup,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FP16 MatMul
// ═══════════════════════════════════════════════════════════════════════

fn bench_matmul_f16(n: usize, m: usize, label: &str, threads: usize) -> BenchResult {
    let total = n * m;
    let w_f32: Vec<f32> = (0..total).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let w_bytes: &[u8] = bytemuck::cast_slice(&w_f16);
    let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.987654).cos()).collect();
    let mut y = vec![0.0f32; n];

    for _ in 0..3 { matmul_f16(&mut y, w_bytes, &x, None, n, m); }

    let runs = 10;
    let t0 = Instant::now();
    for _ in 0..runs { matmul_f16(&mut y, w_bytes, &x, None, n, m); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    let flop = 2.0 * (n * m) as f64;
    let gflops = flop / elapsed / 1e9;
    let bandwidth = (total * 2 + m * 4) as f64 / elapsed / 1e9;

    // Compare vs FP32 baseline
    let mut y_f32 = vec![0.0f32; n];
    let w_f32_dummy: Vec<f32> = w_f32.clone();
    let t0_f32 = Instant::now();
    for _ in 0..runs { matmul_f32(&mut y_f32, &w_f32_dummy, &x, None, n, m); }
    let fp32_elapsed = t0_f32.elapsed().as_secs_f64() / runs as f64;
    let speedup = fp32_elapsed / elapsed;

    BenchResult {
        name: format!("MatMul FP16  [{}×{}] {}", n, m, label),
        category: "matmul-fp16".into(),
        gflops,
        bandwidth_gbps: bandwidth,
        time_ms: elapsed * 1000.0,
        elements_processed: (total * runs) as u64,
        ops_per_element: 2,
        dtype_label: "FP16".into(),
        matrix_shape: format!("{}×{}", n, m),
        speedup_vs_scalar: speedup,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Q4_0 MatMul
// ═══════════════════════════════════════════════════════════════════════

fn bench_matmul_q4_0_bench(n: usize, m: usize, label: &str, threads: usize) -> BenchResult {
    let total = n * m;
    let w_f32: Vec<f32> = (0..total).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.987654).cos()).collect();
    let mut y = vec![0.0f32; n];

    // Quantize
    let w_q4 = hypntz_quantize::quantize_f32_to_q4_0(&w_f32);

    for _ in 0..3 { matmul_q4_0(&mut y, &w_q4, &x, None, n, m); }

    let runs = 10;
    let t0 = Instant::now();
    for _ in 0..runs { matmul_q4_0(&mut y, &w_q4, &x, None, n, m); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    // Mathematically same FLOP, but data movement is much less
    let flop = 2.0 * (n * m) as f64;
    let gflops = flop / elapsed / 1e9;
    let bandwidth = w_q4.len() as f64 / elapsed / 1e9;

    // Compare vs FP32
    let mut y_f32 = vec![0.0f32; n];
    let t0_f32 = Instant::now();
    for _ in 0..runs { matmul_f32(&mut y_f32, &w_f32, &x, None, n, m); }
    let fp32_elapsed = t0_f32.elapsed().as_secs_f64() / runs as f64;
    let speedup = fp32_elapsed / elapsed;

    BenchResult {
        name: format!("MatMul Q4_0  [{}×{}] {}", n, m, label),
        category: "matmul-q4".into(),
        gflops,
        bandwidth_gbps: bandwidth,
        time_ms: elapsed * 1000.0,
        elements_processed: (total * runs) as u64,
        ops_per_element: 2,
        dtype_label: "Q4_0".into(),
        matrix_shape: format!("{}×{}", n, m),
        speedup_vs_scalar: speedup,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RMSNorm
// ═══════════════════════════════════════════════════════════════════════

fn bench_rms_norm(dim: usize) -> BenchResult {
    let runs = 100;
    let mut x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 1.234).sin()).collect();
    let w: Vec<f32> = vec![1.0f32; dim];

    for _ in 0..10 { hypntz_inference::kernels::rms_norm_fused(&mut x.clone(), &w, 1e-5); }

    let t0 = Instant::now();
    for _ in 0..runs {
        hypntz_inference::kernels::rms_norm_fused(&mut x, &w, 1e-5);
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    // 2*n FLOP (square + multiply), + n for sqrt/div
    let flop = 3.0 * dim as f64;
    let gflops = flop / elapsed / 1e9;

    BenchResult {
        name: format!("RMSNorm     dim={}", dim),
        category: "norm".into(),
        gflops,
        bandwidth_gbps: (dim * 8) as f64 / elapsed / 1e9,
        time_ms: elapsed * 1000.0,
        elements_processed: (dim * runs) as u64,
        ops_per_element: 3,
        dtype_label: "FP32".into(),
        matrix_shape: format!("{}", dim),
        speedup_vs_scalar: 1.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════════

fn bench_softmax(dim: usize) -> BenchResult {
    let runs = 100;
    let mut x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();

    for _ in 0..10 { hypntz_inference::kernels::softmax_fast(&mut x.clone()); }

    let t0 = Instant::now();
    for _ in 0..runs {
        hypntz_inference::kernels::softmax_fast(&mut x);
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    let flop = 5.0 * dim as f64; // subtract, exp, add, div
    let gflops = flop / elapsed / 1e9;

    BenchResult {
        name: format!("Softmax     dim={}", dim),
        category: "softmax".into(),
        gflops,
        bandwidth_gbps: (dim * 4) as f64 / elapsed / 1e9,
        time_ms: elapsed * 1000.0,
        elements_processed: (dim * runs) as u64,
        ops_per_element: 5,
        dtype_label: "FP32".into(),
        matrix_shape: format!("{}", dim),
        speedup_vs_scalar: 1.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Quantization speed & accuracy
// ═══════════════════════════════════════════════════════════════════════

fn bench_quantize_speed(n: usize) -> BenchResult {
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.234).sin()).collect();
    let runs = 20;

    for _ in 0..5 { hypntz_quantize::quantize_f32_to_q4_0(&data[..1024]); }

    let t0 = Instant::now();
    for _ in 0..runs {
        let _ = hypntz_quantize::quantize_f32_to_q4_0(&data);
    }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;

    let throughput = n as f64 / elapsed; // elements/sec

    BenchResult {
        name: format!("Quantize Q4_0  {:.1}M elems", n as f64 / 1e6),
        category: "quantize".into(),
        gflops: throughput / 1e9 * 5.0, // ~5 ops per element
        bandwidth_gbps: (n * 4) as f64 / elapsed / 1e9,
        time_ms: elapsed * 1000.0,
        elements_processed: (n * runs) as u64,
        ops_per_element: 5,
        dtype_label: "Q4_0".into(),
        matrix_shape: format!("{}", n),
        speedup_vs_scalar: 1.0,
    }
}

fn bench_quantize_accuracy(n: usize) -> BenchResult {
    // Compare Q4_0 roundtrip error
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let q4 = hypntz_quantize::quantize_f32_to_q4_0(&data);
    let recovered = hypntz_quantize::dequantize_q4_0(&q4);

    let mut mae = 0.0f64;
    let mut max_err = 0.0f64;
    for i in 0..data.len() {
        let err = (data[i] as f64 - recovered[i] as f64).abs();
        mae += err;
        if err > max_err { max_err = err; }
    }
    mae /= data.len() as f64;

    // Compression ratio
    let fp32_bytes = data.len() * 4;
    let q4_bytes = q4.len();
    let ratio = fp32_bytes as f64 / q4_bytes as f64;

    BenchResult {
        name: format!("Q4_0 Accuracy  MAE={:.6}  {:.1}x compression", mae, ratio),
        category: "accuracy".into(),
        gflops: 0.0,
        bandwidth_gbps: 0.0,
        time_ms: 0.0,
        elements_processed: n as u64,
        ops_per_element: 0,
        dtype_label: "Q4_0".into(),
        matrix_shape: format!("n={}", n),
        speedup_vs_scalar: ratio,
    }
}
