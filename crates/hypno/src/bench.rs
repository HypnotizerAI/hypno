//! `hypno bench` — Kernel benchmark suite.

use clap::Parser;
use half::f16;
use hypno_inference::kernels::{cpu_features, matmul_f32, matmul_q4_0, matmul_f16};
use std::time::Instant;

#[derive(Parser)]
pub struct Args {
    /// Number of threads
    #[arg(long, default_value = "4")]
    pub threads: usize,

    /// Output JSON to this path (default: /tmp/hypno_bench_results.json)
    #[arg(long, default_value = "/tmp/hypno_bench_results.json")]
    pub out: String,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().unwrap_or(());
    let feats = cpu_features();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║         🌀 Hypno Benchmark Suite               ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  CPU features: {:>38} ║", feats.best_label());
    println!("║  Threads:      {:>38} ║", args.threads);
    println!("╚══════════════════════════════════════════════════════╝\n");

    let results = run_all_benchmarks();

    println!("\n═══ RESULTS SUMMARY ═══\n");
    for r in &results { println!("{}", r); }

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&args.out, &json)?;
    println!("\n📊 Chart data written to {}", args.out);
    Ok(())
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchResult {
    name: String, category: String, gflops: f64, bandwidth_gbps: f64,
    time_ms: f64, elements_processed: u64, ops_per_element: u64,
    dtype_label: String, matrix_shape: String, speedup_vs_scalar: f64,
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
    let sizes = [(128, 256, "tiny"), (512, 1024, "small"), (1024, 4096, "medium"), (4096, 4096, "large"), (4096, 14336, "llama-ffn")];
    for &(n, m, label) in &sizes {
        results.push(bench_matmul_f32(n, m, label, threads));
        results.push(bench_matmul_f16(n, m, label, threads));
        results.push(bench_matmul_q4(n, m, label, threads));
    }
    results.push(bench_rms_norm(4096));
    results.push(bench_rms_norm(14336));
    results.push(bench_softmax(4096));
    results.push(bench_softmax(32000));
    results.push(bench_quantize(4096 * 4096));
    results.push(bench_accuracy(4096));
    results
}

fn bench_matmul_f32(n: usize, m: usize, label: &str, _threads: usize) -> BenchResult {
    let total = n * m;
    let w: Vec<f32> = (0..total).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.987654).cos()).collect();
    let mut y = vec![0.0f32; n];
    for _ in 0..3 { matmul_f32(&mut y, &w, &x, None, n, m); }
    let runs = 10;
    let t0 = Instant::now();
    for _ in 0..runs { matmul_f32(&mut y, &w, &x, None, n, m); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    let flop = 2.0 * (n * m) as f64;
    let gflops = flop / elapsed / 1e9;
    let bandwidth = (total + m + n) as f64 * 4.0 / elapsed / 1e9;
    let mut ys = vec![0.0f32; n];
    let t0s = Instant::now();
    for _ in 0..runs { for r in 0..n { let mut s = 0.0f32; for c in 0..m { s += w[r*m+c]*x[c]; } ys[r] = s; } }
    let speedup = (t0s.elapsed().as_secs_f64() / runs as f64) / elapsed;
    BenchResult { name: format!("MatMul FP32  [{}×{}] {}", n, m, label), category: "matmul".into(), gflops, bandwidth_gbps: bandwidth, time_ms: elapsed * 1000.0, elements_processed: (total * runs) as u64, ops_per_element: 2, dtype_label: "FP32".into(), matrix_shape: format!("{}×{}", n, m), speedup_vs_scalar: speedup }
}

fn bench_matmul_f16(n: usize, m: usize, label: &str, _threads: usize) -> BenchResult {
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
    let gflops = 2.0 * (n * m) as f64 / elapsed / 1e9;
    let bandwidth = (total * 2 + m * 4) as f64 / elapsed / 1e9;
    let mut y_f32 = vec![0.0f32; n];
    let t0_f32 = Instant::now();
    for _ in 0..runs { matmul_f32(&mut y_f32, &w_f32, &x, None, n, m); }
    let speedup = (t0_f32.elapsed().as_secs_f64() / runs as f64) / elapsed;
    BenchResult { name: format!("MatMul FP16  [{}×{}] {}", n, m, label), category: "matmul-fp16".into(), gflops, bandwidth_gbps: bandwidth, time_ms: elapsed * 1000.0, elements_processed: (total * runs) as u64, ops_per_element: 2, dtype_label: "FP16".into(), matrix_shape: format!("{}×{}", n, m), speedup_vs_scalar: speedup }
}

fn bench_matmul_q4(n: usize, m: usize, label: &str, _threads: usize) -> BenchResult {
    let total = n * m;
    let w_f32: Vec<f32> = (0..total).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.987654).cos()).collect();
    let mut y = vec![0.0f32; n];
    let w_q4 = hypno_quantize::quantize_f32_to_q4_0(&w_f32);
    for _ in 0..3 { matmul_q4_0(&mut y, &w_q4, &x, None, n, m); }
    let runs = 10;
    let t0 = Instant::now();
    for _ in 0..runs { matmul_q4_0(&mut y, &w_q4, &x, None, n, m); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    let gflops = 2.0 * (n * m) as f64 / elapsed / 1e9;
    let bandwidth = w_q4.len() as f64 / elapsed / 1e9;
    let mut y_f32 = vec![0.0f32; n];
    let t0_f32 = Instant::now();
    for _ in 0..runs { matmul_f32(&mut y_f32, &w_f32, &x, None, n, m); }
    let speedup = (t0_f32.elapsed().as_secs_f64() / runs as f64) / elapsed;
    BenchResult { name: format!("MatMul Q4_0  [{}×{}] {}", n, m, label), category: "matmul-q4".into(), gflops, bandwidth_gbps: bandwidth, time_ms: elapsed * 1000.0, elements_processed: (total * runs) as u64, ops_per_element: 2, dtype_label: "Q4_0".into(), matrix_shape: format!("{}×{}", n, m), speedup_vs_scalar: speedup }
}

fn bench_rms_norm(dim: usize) -> BenchResult {
    let runs = 100;
    let mut x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 1.234).sin()).collect();
    let w: Vec<f32> = vec![1.0f32; dim];
    for _ in 0..10 { hypno_inference::kernels::rms_norm_fused(&mut x.clone(), &w, 1e-5); }
    let t0 = Instant::now();
    for _ in 0..runs { hypno_inference::kernels::rms_norm_fused(&mut x, &w, 1e-5); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    BenchResult { name: format!("RMSNorm     dim={}", dim), category: "norm".into(), gflops: 3.0 * dim as f64 / elapsed / 1e9, bandwidth_gbps: (dim * 8) as f64 / elapsed / 1e9, time_ms: elapsed * 1000.0, elements_processed: (dim * runs) as u64, ops_per_element: 3, dtype_label: "FP32".into(), matrix_shape: dim.to_string(), speedup_vs_scalar: 1.0 }
}

fn bench_softmax(dim: usize) -> BenchResult {
    let runs = 100;
    let mut x: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.01).sin()).collect();
    for _ in 0..10 { hypno_inference::kernels::softmax_fast(&mut x.clone()); }
    let t0 = Instant::now();
    for _ in 0..runs { hypno_inference::kernels::softmax_fast(&mut x); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    BenchResult { name: format!("Softmax     dim={}", dim), category: "softmax".into(), gflops: 5.0 * dim as f64 / elapsed / 1e9, bandwidth_gbps: (dim * 4) as f64 / elapsed / 1e9, time_ms: elapsed * 1000.0, elements_processed: (dim * runs) as u64, ops_per_element: 5, dtype_label: "FP32".into(), matrix_shape: dim.to_string(), speedup_vs_scalar: 1.0 }
}

fn bench_quantize(n: usize) -> BenchResult {
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.234).sin()).collect();
    let runs = 20;
    for _ in 0..5 { hypno_quantize::quantize_f32_to_q4_0(&data[..1024]); }
    let t0 = Instant::now();
    for _ in 0..runs { let _ = hypno_quantize::quantize_f32_to_q4_0(&data); }
    let elapsed = t0.elapsed().as_secs_f64() / runs as f64;
    let throughput = n as f64 / elapsed;
    BenchResult { name: format!("Quantize Q4_0  {:.1}M elems", n as f64 / 1e6), category: "quantize".into(), gflops: throughput / 1e9 * 5.0, bandwidth_gbps: (n * 4) as f64 / elapsed / 1e9, time_ms: elapsed * 1000.0, elements_processed: (n * runs) as u64, ops_per_element: 5, dtype_label: "Q4_0".into(), matrix_shape: n.to_string(), speedup_vs_scalar: 1.0 }
}

fn bench_accuracy(n: usize) -> BenchResult {
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 1.234567).sin()).collect();
    let q4 = hypno_quantize::quantize_f32_to_q4_0(&data);
    let recovered = hypno_quantize::dequantize_q4_0(&q4);
    let mut mae = 0.0f64;
    for i in 0..data.len() { mae += (data[i] as f64 - recovered[i] as f64).abs(); }
    mae /= data.len() as f64;
    let ratio = (data.len() * 4) as f64 / q4.len() as f64;
    BenchResult { name: format!("Q4_0 Accuracy  MAE={:.6}  {:.1}x compression", mae, ratio), category: "accuracy".into(), gflops: 0.0, bandwidth_gbps: 0.0, time_ms: 0.0, elements_processed: n as u64, ops_per_element: 0, dtype_label: "Q4_0".into(), matrix_shape: format!("n={}", n), speedup_vs_scalar: ratio }
}
