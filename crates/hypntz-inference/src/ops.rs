//! Core tensor operators — all heavy ops auto-dispatch to optimal SIMD.
//!
//! MatMul: AVX-512 > AVX2+FMA > SSE4.1 > scalar
//! RMSNorm: AVX2 fused-reduce
//! Softmax: AVX2 polynomial-approximated exp

use hypntz_core::DType;

/// Auto-dispatched FP32 matmul.
pub fn matmul_vec(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    crate::kernels::matmul_f32(y, w, x, bias, n, m);
}

/// Auto-dispatched Q4_0 matmul.
pub fn matmul_vec_q4_0(y: &mut [f32], w_q: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    crate::kernels::matmul_q4_0(y, w_q, x, bias, n, m);
}

/// Auto-dispatched FP16 matmul (F16C hardware conversion).
pub fn matmul_vec_f16(y: &mut [f32], w: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    crate::kernels::matmul_f16(y, w, x, bias, n, m);
}

/// Generic matmul dispatch based on weight dtype.
pub fn matmul_vec_auto(
    y: &mut [f32], w: &[u8], dtype: DType, x: &[f32],
    bias: Option<&[f32]>, n: usize, m: usize,
) {
    match dtype {
        DType::FP32 => matmul_vec(y, bytemuck::cast_slice(w), x, bias, n, m),
        DType::FP16 => matmul_vec_f16(y, w, x, bias, n, m),
        DType::Q4_0 => matmul_vec_q4_0(y, w, x, bias, n, m),
        DType::Q8_0 => {
            let w_f32: Vec<f32> = hypntz_quantize::dequantize_q8_0(w);
            matmul_vec(y, &w_f32, x, bias, n, m);
        }
    }
}

/// RMSNorm in-place, SIMD-fused.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    crate::kernels::rms_norm_fused(&mut y, weight, eps);
    y
}

/// RMSNorm in-place, SIMD-fused.
pub fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    crate::kernels::rms_norm_fused(x, weight, eps);
}

/// Softmax in-place, AVX2 polynomial exp.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    softmax_in_place(&mut y);
    y
}

/// Softmax in-place, AVX2 polynomial exp.
pub fn softmax_in_place(x: &mut [f32]) {
    crate::kernels::softmax_fast(x);
}

/// SiLU activation.
pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// SiLU in-place.
pub fn silu_in_place(x: &mut [f32]) {
    for xi in x.iter_mut() { *xi = *xi / (1.0 + (-*xi).exp()); }
}

/// Rotary Position Embeddings (RoPE).
pub fn rope(q: &mut [f32], k: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    let pf = pos as f32;
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let cos = (pf * freq).cos();
        let sin = (pf * freq).sin();
        let q0 = q[i]; let q1 = q[i + 1];
        q[i] = q0 * cos - q1 * sin;
        q[i + 1] = q0 * sin + q1 * cos;
        let k0 = k[i]; let k1 = k[i + 1];
        k[i] = k0 * cos - k1 * sin;
        k[i + 1] = k0 * sin + k1 * cos;
    }
}

/// Batched RoPE across a sequence.
pub fn rope_batched(q: &mut [f32], k: &mut [f32], seq_len: usize, head_dim: usize, theta: f32) {
    for pos in 0..seq_len {
        let s = pos * head_dim;
        rope(&mut q[s..s + head_dim], &mut k[s..s + head_dim], head_dim, pos, theta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_vec() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![5.0f32, 6.0];
        let mut y = vec![0.0f32; 2];
        matmul_vec(&mut y, &w, &x, None, 2, 2);
        assert!((y[0] - 17.0).abs() < 0.001);
        assert!((y[1] - 39.0).abs() < 0.001);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = softmax(&x);
        assert!(((y.iter().sum::<f32>()) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rms_norm() {
        let x = vec![2.0f32; 4];
        let w = vec![1.0f32; 4];
        let y = rms_norm(&x, &w, 1e-5);
        for yi in &y { assert!((yi - 1.0).abs() < 0.001); }
    }
}
