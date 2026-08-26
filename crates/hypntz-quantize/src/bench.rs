//! Benchmarking utilities for quantization accuracy and performance.

use std::time::Instant;

/// Result of a quantization benchmark comparing Q4_0 against FP32 reference.
#[derive(Debug, Clone)]
pub struct QuantBenchResult {
    /// Total number of elements quantized.
    pub n_elements: usize,
    /// Bytes used by FP32 representation.
    pub fp32_bytes: usize,
    /// Bytes used by FP16 representation.
    pub fp16_bytes: usize,
    /// Bytes used by Q4_0 representation.
    pub q4_0_bytes: usize,
    /// Bytes used by Q8_0 representation.
    pub q8_0_bytes: usize,
    /// Mean absolute error (Q4_0 vs FP32).
    pub q4_0_mae: f64,
    /// Max absolute error (Q4_0 vs FP32).
    pub q4_0_max_err: f64,
    /// Mean absolute error (Q8_0 vs FP32).
    pub q8_0_mae: f64,
    /// Max absolute error (Q8_0 vs FP32).
    pub q8_0_max_err: f64,
    /// Dot product relative error (Q4_0 vs FP32).
    pub q4_0_dot_err: f64,
    /// Dot product relative error (Q8_0 vs FP32).
    pub q8_0_dot_err: f64,
    /// Compression ratio FP32 / Q4_0.
    pub q4_0_compression: f64,
    /// Compression ratio FP32 / Q8_0.
    pub q8_0_compression: f64,
    /// Quantization time (ms).
    pub quant_time_ms: f64,
}

/// Run a comprehensive benchmark on a set of weight matrices.
pub fn benchmark_quantization(weights: &[&[f32]]) -> QuantBenchResult {
    use crate::{quantize_f32_to_q4_0, quantize_f32_to_q8_0, dequantize_q4_0, dequantize_q8_0};
    use half::f16;

    let t0 = Instant::now();

    let mut total_n = 0usize;
    let mut total_fp32_bytes = 0usize;
    let mut total_fp16_bytes = 0usize;
    let mut total_q4_bytes = 0usize;
    let mut total_q8_bytes = 0usize;

    let mut q4_mae_sum = 0.0f64;
    let mut q4_max_err = 0.0f64;
    let mut q8_mae_sum = 0.0f64;
    let mut q8_max_err = 0.0f64;

    for w in weights {
        let n = w.len();
        total_n += n;
        total_fp32_bytes += n * 4;
        total_fp16_bytes += n * 2;

        // Q4_0
        let q4 = quantize_f32_to_q4_0(w);
        total_q4_bytes += q4.len();
        let dq4 = dequantize_q4_0(&q4);

        for i in 0..n {
            let err = (w[i] as f64 - dq4[i] as f64).abs();
            q4_mae_sum += err;
            if err > q4_max_err { q4_max_err = err; }
        }

        // Q8_0
        let q8 = quantize_f32_to_q8_0(w);
        total_q8_bytes += q8.len();
        let dq8 = dequantize_q8_0(&q8);

        for i in 0..n {
            let err = (w[i] as f64 - dq8[i] as f64).abs();
            q8_mae_sum += err;
            if err > q8_max_err { q8_max_err = err; }
        }
    }

    let q4_mae = q4_mae_sum / total_n as f64;
    let q8_mae = q8_mae_sum / total_n as f64;

    // Dot product error: use a random probe vector
    let q4_dot_err = compute_dot_error(weights, &|w| quantize_f32_to_q4_0(w));
    let q8_dot_err = compute_dot_error(weights, &|w| quantize_f32_to_q8_0(w));

    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

    QuantBenchResult {
        n_elements: total_n,
        fp32_bytes: total_fp32_bytes,
        fp16_bytes: total_fp16_bytes,
        q4_0_bytes: total_q4_bytes,
        q8_0_bytes: total_q8_bytes,
        q4_0_mae: q4_mae,
        q4_0_max_err: q4_max_err,
        q8_0_mae: q8_mae,
        q8_0_max_err: q8_max_err,
        q4_0_dot_err: q4_dot_err,
        q8_0_dot_err: q8_dot_err,
        q4_0_compression: total_fp32_bytes as f64 / total_q4_bytes as f64,
        q8_0_compression: total_fp32_bytes as f64 / total_q8_bytes as f64,
        quant_time_ms: elapsed,
    }
}

fn compute_dot_error(weights: &[&[f32]], quantize: &dyn Fn(&[f32]) -> Vec<u8>) -> f64 {
    use crate::q4_0;
    // Create a random probe vector
    let probe_len = weights.iter().map(|w| w.len()).max().unwrap_or(1);
    let probe: Vec<f32> = (0..probe_len)
        .map(|i| ((i as f32 * 1.234 + 0.567).sin()))
        .collect();

    let mut fp32_sum = 0.0f64;
    let mut q_sum = 0.0f64;

    for w in weights {
        let n = w.len();
        let fp32_dot: f32 = probe[..n].iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        fp32_sum += fp32_dot as f64;

        let q_w = quantize(w);
        let q_dot = q4_0::dot_product_q4_0_scalar(&probe[..n], &q_w, n) as f64;
        q_sum += q_dot;
    }

    (fp32_sum - q_sum).abs() / fp32_sum.abs().max(1e-10)
}
