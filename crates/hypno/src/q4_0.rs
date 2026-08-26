//! Vectorized Q4_0 dequantization and dot-product kernels.
//!
//! Provides architecture-specific SIMD implementations with automatic
//! dispatch at runtime. Falls back to scalar code when no SIMD is available.
//!
//! Q4_0 block layout (18 bytes):
//!   bytes 0..2:  f16 scale
//!   bytes 2..18: 16 packed bytes → 32 nibbles (4-bit quantized values)
//! Dequantize: val = (nibble - 8) * scale

use half::f16;

/// Architecture-dispatched dot product of f32 vector x against Q4_0 quantized vector y_q.
#[allow(dead_code)]
pub fn dot_product_q4_0(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_product_q4_0_avx2(x, y_q, n) };
            }
        }
        #[cfg(target_feature = "sse4.1")]
        {
            if is_x86_feature_detected!("sse4.1") {
                return unsafe { dot_product_q4_0_sse41(x, y_q, n) };
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { dot_product_q4_0_neon(x, y_q, n) };
            }
        }
    }

    dot_product_q4_0_scalar(x, y_q, n)
}

/// Scalar fallback for Q4_0 dot product.
pub fn dot_product_q4_0_scalar(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    let n_blocks = n.div_ceil(32);
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let offset = b * 18;
        if offset + 18 > y_q.len() { break; }
        let scale = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let qs = &y_q[offset + 2..offset + 18];

        for i in 0..16 {
            let byte = qs[i];
            let lo = (byte & 0x0F) as i32;
            let hi = ((byte >> 4) & 0x0F) as i32;
            let idx = b * 32 + i * 2;
            if idx < n {
                sum += x[idx] * ((lo - 8) as f32 * scale);
            }
            if idx + 1 < n {
                sum += x[idx + 1] * ((hi - 8) as f32 * scale);
            }
        }
    }

    sum
}

/// SSE4.1-accelerated Q4_0 dot product using 4-wide SIMD accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(dead_code)]
unsafe fn dot_product_q4_0_sse41(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = n.div_ceil(32);
    let mut acc = _mm_setzero_ps();

    for b in 0..n_blocks {
        let offset = b * 18;
        if offset + 18 > y_q.len() { break; }
        let scale_val = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let scale = _mm_set1_ps(scale_val);
        let qs = &y_q[offset + 2..offset + 18];
        let base_idx = b * 32;

        for i in 0..16 {
            let byte = qs[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            let idx = base_idx + i * 2;

            // Load 2 adjacent elements (lo, hi) into a 4-wide register
            if idx + 2 <= n {
                let x_vals = [x[idx], x[idx + 1], 0.0, 0.0];
                let xv = _mm_loadu_ps(x_vals.as_ptr());
                let qv = _mm_set_ps(0.0, 0.0, hi, lo);
                let scaled = _mm_mul_ps(qv, scale);
                acc = _mm_add_ps(acc, _mm_mul_ps(xv, scaled));
            } else {
                // Tail: accumulate scalar
                let mut local = 0.0f32;
                if idx < n { local += x[idx] * lo * scale_val; }
                if idx + 1 < n { local += x[idx + 1] * hi * scale_val; }
                acc = _mm_add_ps(acc, _mm_set_ps(0.0, 0.0, 0.0, local));
            }
        }
    }

    // Horizontal sum of 4 floats
    let mut result_arr = [0.0f32; 4];
    _mm_storeu_ps(result_arr.as_mut_ptr(), acc);
    result_arr.iter().sum()
}

/// AVX2-accelerated Q4_0 dot product using 8-wide SIMD accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn dot_product_q4_0_avx2(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = n.div_ceil(32);
    let mut acc = _mm256_setzero_ps();

    for b in 0..n_blocks {
        let offset = b * 18;
        if offset + 18 > y_q.len() { break; }
        let scale_val = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let scale = _mm256_set1_ps(scale_val);
        let qs = &y_q[offset + 2..offset + 18];
        let base_idx = b * 32;

        // Process 4 pairs (8 elements) at a time
        for i in (0..16).step_by(4) {
            let idx = base_idx + i * 2;
            if idx + 8 > n { break; } // Skip incomplete 8-wide chunks

            // Load 8 f32 values from x
            let xv = _mm256_loadu_ps(x.as_ptr().add(idx));

            // Build 8 quantized values from 4 bytes
            let mut q_vals = [0.0f32; 8];
            for j in 0..4 {
                let byte = qs[i + j];
                q_vals[j * 2] = ((byte & 0x0F) as i32 - 8) as f32;
                q_vals[j * 2 + 1] = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            }
            let qv = _mm256_loadu_ps(q_vals.as_ptr());

            // acc += xv * (qv * scale)
            acc = _mm256_fmadd_ps(xv, _mm256_mul_ps(qv, scale), acc);
        }

        // Tail: any remaining elements in this block
        for i in ((n - base_idx).div_ceil(2)).min(16) & !3..16 {
            let idx = base_idx + i * 2;
            let byte = qs[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            let mut local = 0.0f32;
            if idx < n { local += x[idx] * lo * scale_val; }
            if idx + 1 < n { local += x[idx + 1] * hi * scale_val; }
            acc = _mm256_add_ps(acc, _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, local));
        }
    }

    // Horizontal sum of 8 floats
    let mut result_arr = [0.0f32; 8];
    _mm256_storeu_ps(result_arr.as_mut_ptr(), acc);
    result_arr.iter().sum()
}

/// NEON-accelerated Q4_0 dot product (4-wide SIMD).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_q4_0_neon(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    use std::arch::aarch64::*;
    let n_blocks = n.div_ceil(32);
    let mut acc = vdupq_n_f32(0.0);

    for b in 0..n_blocks {
        let offset = b * 18;
        if offset + 18 > y_q.len() { break; }
        let scale_val = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let scale = vdupq_n_f32(scale_val);
        let qs = &y_q[offset + 2..offset + 18];
        let base_idx = b * 32;

        for i in 0..16 {
            let byte = qs[i];
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            let idx = base_idx + i * 2;

            if idx + 2 <= n {
                let xv = vld1q_f32([x[idx], x[idx + 1], 0.0, 0.0].as_ptr());
                let qv = vld1q_f32([lo, hi, 0.0, 0.0].as_ptr());
                acc = vfmaq_f32(acc, xv, vmulq_f32(qv, scale));
            } else {
                let mut local = 0.0f32;
                if idx < n { local += x[idx] * lo * scale_val; }
                if idx + 1 < n { local += x[idx + 1] * hi * scale_val; }
                acc = vaddq_f32(acc, vld1q_f32([local, 0.0, 0.0, 0.0].as_ptr()));
            }
        }
    }

    // Horizontal sum
    let mut result_arr = [0.0f32; 4];
    vst1q_f32(result_arr.as_mut_ptr(), acc);
    result_arr.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_dot_product_q4_0_vs_fp32() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

        // Quantize y manually
        let n_blocks = n / 32;
        let mut y_q = vec![0u8; n_blocks * 18];
        for b in 0..n_blocks {
            let start = b * 32;
            let max_abs = y[start..start + 32].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            y_q[b * 18] = f16::from_f32(scale).to_le_bytes()[0];
            y_q[b * 18 + 1] = f16::from_f32(scale).to_le_bytes()[1];

            for i in 0..16 {
                let v0 = y[start + i * 2];
                let v1 = y[start + i * 2 + 1];
                let q0 = ((v0 / scale).round() as i32).clamp(-8, 7);
                let q1 = ((v1 / scale).round() as i32).clamp(-8, 7);
                y_q[b * 18 + 2 + i] = ((q0 + 8) as u8 & 0x0F) | (((q1 + 8) as u8 & 0x0F) << 4);
            }
        }

        let scalar_dot = dot_product_q4_0_scalar(&x, &y_q, n);
        let simd_dot = dot_product_q4_0(&x, &y_q, n);

        assert!((scalar_dot - simd_dot).abs() < 0.01,
            "scalar={}, simd={}", scalar_dot, simd_dot);

        let fp32_dot: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rel_err = (fp32_dot - simd_dot).abs() / fp32_dot.abs().max(1e-6);
        assert!(rel_err < 0.15, "relative error {} too large (fp32={}, q4={})", rel_err, fp32_dot, simd_dot);
    }

    #[test]
    fn test_dot_product_exact_small() {
        // Test with exactly 32 elements
        let x: Vec<f32> = (0..32).map(|_i| 1.0f32).collect();
        let _y: Vec<f32> = (0..32).map(|_i| 1.0f32).collect();

        let mut y_q = vec![0u8; 18];
        let max_abs = 1.0;
        let scale = max_abs / 7.0;
        y_q[0] = f16::from_f32(scale).to_le_bytes()[0];
        y_q[1] = f16::from_f32(scale).to_le_bytes()[1];

        for i in 0..16 {
            let q = ((1.0 / scale).round() as i32).clamp(-8, 7);
            let nib = (q + 8) as u8 & 0x0F;
            y_q[2 + i] = nib | (nib << 4);
        }

        let result = dot_product_q4_0(&x, &y_q, 32);
        // Expected: sum(1 * deq) for 32 values where deq ≈ 1.0
        assert!(result > 0.0, "Expected positive dot product, got {}", result);
    }
}
