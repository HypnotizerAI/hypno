//! Vectorized Q8_0 dot-product kernel.

use half::f16;

/// Scalar dot product with Q8_0 quantized data.
pub fn dot_product_q8_0_scalar(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    let n_blocks = n.div_ceil(32);
    let mut sum = 0.0f32;

    for b in 0..n_blocks {
        let offset = b * 34;
        let scale = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let qs = &y_q[offset + 2..offset + 34];

        for j in 0..32 {
            let idx = b * 32 + j;
            if idx < n {
                let deq = (qs[j] as i32 - 128) as f32 * scale;
                sum += x[idx] * deq;
            }
        }
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_q8_0_avx2(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let n_blocks = n.div_ceil(32);
    let mut acc = _mm256_setzero_ps();
    let offset_128 = _mm256_set1_ps(128.0);

    for b in 0..n_blocks {
        let offset = b * 34;
        let scale_val = f16::from_le_bytes([y_q[offset], y_q[offset + 1]]).to_f32();
        let scale = _mm256_set1_ps(scale_val);
        let base_idx = b * 32;

        for j in (0..32).step_by(8) {
            let idx = base_idx + j;
            if idx + 8 > n { break; }

            let x_vec = _mm256_loadu_ps(x.as_ptr().add(idx));

            // Load 8 u8 values and expand to f32
            let q_u8 = _mm_loadl_epi64(y_q.as_ptr().add(offset + 2 + j) as *const __m128i);
            let q_i32 = _mm256_cvtepu8_epi32(q_u8);
            let q_f32 = _mm256_cvtepi32_ps(q_i32);
            let q_centered = _mm256_sub_ps(q_f32, offset_128);
            let deq = _mm256_mul_ps(q_centered, scale);

            acc = _mm256_fmadd_ps(x_vec, deq, acc);
        }
    }

    // Horizontal sum
    let acc_low = _mm256_castps256_ps128(acc);
    let acc_high = _mm256_extractf128_ps(acc, 1);
    let acc_sum = _mm_add_ps(acc_low, acc_high);
    let acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
    let acc_sum = _mm_hadd_ps(acc_sum, acc_sum);
    let mut result = 0.0f32;
    _mm_store_ss(&mut result, acc_sum);
    result
}

/// Architecture-dispatched Q8_0 dot product.
pub fn dot_product_q8_0(x: &[f32], y_q: &[u8], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_q8_0_avx2(x, y_q, n) };
        }
    }
    dot_product_q8_0_scalar(x, y_q, n)
}
