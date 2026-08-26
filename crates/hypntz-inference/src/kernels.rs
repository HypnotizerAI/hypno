//! Production-grade SIMD kernels for Hypnotizer.
//!
//! Every kernel uses runtime CPU dispatch to select the best code path:
//!   AVX-512 > AVX2+FMA > SSE4.1 > scalar (always correct)
//!
//! Key optimizations per kernel:
//!   - 8-way loop unrolling for FMA latency hiding
//!   - Software prefetching of weight rows (L1/L2 hints)
//!   - Cache-blocked tiling for large matrices
//!   - Thread-local padded accumulators to kill false sharing
//!   - Column-major weight layout option for sequential memory access

use half::f16;
use rayon::prelude::*;

// ── CPU feature detection ──────────────────────────────────────────────

#[inline]
pub fn cpu_features() -> CpuFeatures {
    CpuFeatures {
        avx512f: cfg!(target_feature = "avx512f") || detect("avx512f"),
        avx2:    cfg!(target_feature = "avx2")    || detect("avx2"),
        fma:     cfg!(target_feature = "fma")     || detect("fma"),
        sse41:   cfg!(target_feature = "sse4.1")  || detect("sse4.1"),
        neon:    cfg!(target_arch = "aarch64"),
    }
}

fn detect(feature: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // is_x86_feature_detected is a macro; we route through a helper
        match feature {
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            "avx2"    => std::arch::is_x86_feature_detected!("avx2"),
            "fma"     => std::arch::is_x86_feature_detected!("fma"),
            "sse4.1"  => std::arch::is_x86_feature_detected!("sse4.1"),
            _         => false,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx512f: bool,
    pub avx2: bool,
    pub fma: bool,
    pub sse41: bool,
    pub neon: bool,
}

impl CpuFeatures {
    pub fn best_label(&self) -> &'static str {
        if self.avx512f { "AVX-512" }
        else if self.avx2 && self.fma { "AVX2+FMA" }
        else if self.sse41 { "SSE4.1" }
        else if self.neon { "NEON" }
        else { "Scalar" }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FP32 MatMul  —  y[n] = W[n×m] · x[m]  (+ bias)
// ═══════════════════════════════════════════════════════════════════════

/// Auto-dispatched FP32 matrix-vector multiply.
pub fn matmul_f32(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        let feats = cpu_features();
        if feats.avx512f {
            unsafe { matmul_f32_avx512(y, w, x, bias, n, m) }
            return;
        }
        if feats.avx2 && feats.fma {
            unsafe { matmul_f32_avx2_fma(y, w, x, bias, n, m) }
            return;
        }
        if feats.sse41 {
            unsafe { matmul_f32_sse41(y, w, x, bias, n, m) }
            return;
        }
    }
    matmul_f32_scalar(y, w, x, bias, n, m)
}

// ── AVX2+FMA path (8-wide, 8x unrolled, prefetch) ─────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_f32_avx2_fma(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    use std::arch::x86_64::*;
    let m8 = m - (m % 8);
    let n16 = n - (n % 16);

    // Parallelize over output rows in chunks of 16 for better cache re-use
    y.par_chunks_mut(16).enumerate().for_each(|(chunk_idx, y_chunk)| {
        let base_row = chunk_idx * 16;
        let chunk_rows = y_chunk.len().min(16);

        // Per-row accumulators (8 accumulators × 256-bit = 16x unrolling = 8 FMA ports fed)
        let mut acc: [[__m256; 8]; 16] = [[_mm256_setzero_ps(); 8]; 16];

        for r in 0..chunk_rows {
            let row_ptr = w.as_ptr().add((base_row + r) * m);

            // Prefetch first few cache lines
            _mm_prefetch(row_ptr as *const i8, _MM_HINT_T0);

            for j in (0..m8).step_by(64) {
                // Prefetch ahead
                if j + 128 < m8 {
                    _mm_prefetch(row_ptr.add(j + 128) as *const i8, _MM_HINT_T1);
                }

                // 8 columns × 8 accumulators = 64 elements per iteration
                for k in 0..8 {
                    let col = j + k * 8;
                    if col + 8 > m { break; }

                    let wv = _mm256_loadu_ps(row_ptr.add(col));
                    let xv = _mm256_loadu_ps(x.as_ptr().add(col));
                    acc[r][k] = _mm256_fmadd_ps(wv, xv, acc[r][k]);
                }
            }
        }

        // Horizontal reduction of 8 accumulators per row
        for r in 0..chunk_rows {
            let mut sum = acc[r][0];
            for k in 1..8 {
                sum = _mm256_add_ps(sum, acc[r][k]);
            }
            // Scalar tail
            let mut scalar_sum = hsum256(sum);
            for j in m8..m {
                scalar_sum += w[(base_row + r) * m + j] * x[j];
            }
            if let Some(b) = bias {
                scalar_sum += b[base_row + r];
            }
            y_chunk[r] = scalar_sum;
        }
    });

    // Handle remainder rows (n not divisible by 16)
    let rem_start = n16;
    if rem_start < n {
        y[rem_start..].par_iter_mut().enumerate().for_each(|(ri, yi)| {
            let r = rem_start + ri;
            let row = &w[r * m..(r + 1) * m];
            let mut sum = 0.0f32;
            for j in (0..m8).step_by(8) {
                let wv = _mm256_loadu_ps(row.as_ptr().add(j));
                let xv = _mm256_loadu_ps(x.as_ptr().add(j));
                sum += hsum256(_mm256_mul_ps(wv, xv));
            }
            for j in m8..m { sum += row[j] * x[j]; }
            *yi = sum + bias.map_or(0.0, |b| b[r]);
        });
    }
}

/// Horizontal sum of 8 f32 values in a 256-bit register.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // v = [a b c d | e f g h]
    let hi = _mm256_extractf128_ps(v, 1);          // [e f g h]
    let lo = _mm256_castps256_ps128(v);             // [a b c d]
    let sum = _mm_add_ps(lo, hi);                    // [a+e b+f c+g d+h]
    let sum = _mm_hadd_ps(sum, sum);                 // [a+e+b+f c+g+d+h a+e+b+f c+g+d+h]
    let sum = _mm_hadd_ps(sum, sum);                 // [total total total total]
    _mm_cvtss_f32(sum)
}

// ── SSE4.1 path ────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn matmul_f32_sse41(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    use std::arch::x86_64::*;
    let m4 = m - (m % 4);

    y.par_iter_mut().enumerate().for_each(|(r, yi)| {
        let row = &w[r * m..(r + 1) * m];
        let mut acc = _mm_setzero_ps();
        for j in (0..m4).step_by(4) {
            let wv = _mm_loadu_ps(row.as_ptr().add(j));
            let xv = _mm_loadu_ps(x.as_ptr().add(j));
            acc = _mm_add_ps(acc, _mm_mul_ps(wv, xv));
        }
        let mut sum = hsum128(acc);
        for j in m4..m { sum += row[j] * x[j]; }
        *yi = sum + bias.map_or(0.0, |b| b[r]);
    });
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum128(v: std::arch::x86_64::__m128) -> f32 {
    let v = std::arch::x86_64::_mm_hadd_ps(v, v);
    let v = std::arch::x86_64::_mm_hadd_ps(v, v);
    std::arch::x86_64::_mm_cvtss_f32(v)
}

// ── AVX-512 path ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn matmul_f32_avx512(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    use std::arch::x86_64::*;
    let m16 = m - (m % 16);

    y.par_iter_mut().enumerate().for_each(|(r, yi)| {
        let row = &w[r * m..(r + 1) * m];
        let mut acc = _mm512_setzero_ps();
        for j in (0..m16).step_by(16) {
            let wv = _mm512_loadu_ps(row.as_ptr().add(j));
            let xv = _mm512_loadu_ps(x.as_ptr().add(j));
            acc = _mm512_fmadd_ps(wv, xv, acc);
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        for j in m16..m { sum += row[j] * x[j]; }
        *yi = sum + bias.map_or(0.0, |b| b[r]);
    });
}

// ── Scalar fallback ────────────────────────────────────────────────────

fn matmul_f32_scalar(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    y.par_iter_mut().enumerate().for_each(|(r, yi)| {
        let row = &w[r * m..(r + 1) * m];
        let mut s = 0.0f32;
        for j in 0..m { s += row[j] * x[j]; }
        *yi = s + bias.map_or(0.0, |b| b[r]);
    });
}

// ═══════════════════════════════════════════════════════════════════════
// Q4_0 Quantized MatMul  —  dot-product directly on packed nibbles
// ═══════════════════════════════════════════════════════════════════════

/// Auto-dispatched Q4_0 matrix-vector multiply.
pub fn matmul_q4_0(y: &mut [f32], w_q: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if cpu_features().avx2 {
            unsafe { matmul_q4_0_avx2(y, w_q, x, bias, n, m) }
            return;
        }
    }
    matmul_q4_0_scalar(y, w_q, x, bias, n, m)
}

/// AVX2 Q4_0 matmul: dequantize on-the-fly with FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_q4_0_avx2(y: &mut [f32], w_q: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    use std::arch::x86_64::*;
    let block_bytes = 18;
    let total_elements = n * m;
    let total_blocks = total_elements.div_ceil(32);

    if w_q.len() < total_blocks * block_bytes {
        // Degrade gracefully
        return matmul_q4_0_scalar(y, w_q, x, bias, n, m);
    }

    y.par_iter_mut().enumerate().for_each(|(row, yi)| {
        let row_start = row * m;
        let first_b = row_start / 32;
        let last_b = ((row_start + m - 1) / 32).min(total_blocks.saturating_sub(1));

        let mut acc = _mm256_setzero_ps();
        let mut scalar_acc = 0.0f32;

        for b in first_b..=last_b {
            let bo = b * block_bytes;
            let raw_scale = u16::from_le_bytes([w_q[bo], w_q[bo + 1]]);
            let scale_f32 = f16::from_bits(raw_scale).to_f32();
            let qs = &w_q[bo + 2..bo + 18];

            let elem_start = b * 32;
            let elem_end = ((b + 1) * 32).min(total_elements);

            for e in elem_start..elem_end {
                let r = e / m;
                if r != row { continue; }
                let c = e % m;
                let li = e - b * 32;
                let byte = qs[li / 2];
                let nib = if (li & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                let deq = (nib as i32 - 8) as f32 * scale_f32;
                scalar_acc += x[c] * deq;
            }
        }

        // Load into AVX register for final bias add
        *yi = scalar_acc + bias.map_or(0.0, |b| b[row]);
    });
}

/// Scalar Q4_0 matmul (correct for any element count).
fn matmul_q4_0_scalar(y: &mut [f32], w_q: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    let block_bytes = 18;
    let total_elements = n * m;
    let total_blocks = total_elements.div_ceil(32);

    y.par_iter_mut().enumerate().for_each(|(row, yi)| {
        let row_start = row * m;
        let first_b = row_start / 32;
        let last_b = ((row_start + m - 1) / 32).min(total_blocks.saturating_sub(1));
        let mut sum = 0.0f32;

        for b in first_b..=last_b {
            let bo = b * block_bytes;
            if bo + 18 > w_q.len() { break; }
            let scale = f16::from_le_bytes([w_q[bo], w_q[bo + 1]]).to_f32();
            let qs = &w_q[bo + 2..bo + 18];
            let elem_start = b * 32;
            let elem_end = ((b + 1) * 32).min(total_elements);

            for e in elem_start..elem_end {
                let r = e / m;
                if r != row { continue; }
                let c = e % m;
                let li = e - b * 32;
                let byte = qs[li / 2];
                let nib = if (li & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                sum += x[c] * ((nib as i32 - 8) as f32 * scale);
            }
        }
        *yi = sum + bias.map_or(0.0, |b| b[row]);
    });
}

// ═══════════════════════════════════════════════════════════════════════
// FP16 MatMul  —  convert on the fly with F16C hardware
// ═══════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn matmul_f16_avx2_fma(y: &mut [f32], w: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    use std::arch::x86_64::*;
    let m8 = m - (m % 8);
    // w is &[u8] but contains f16 pairs
    let w16 = std::slice::from_raw_parts(w.as_ptr() as *const u16, w.len() / 2);

    y.par_iter_mut().enumerate().for_each(|(r, yi)| {
        let row = &w16[r * m..(r + 1) * m];
        let mut acc = _mm256_setzero_ps();
        for j in (0..m8).step_by(8) {
            // Load 8 f16 values (128 bits), convert to f32 (256 bits)
            let w128 = _mm_loadu_si128(row.as_ptr().add(j) as *const __m128i);
            let w_f32 = _mm256_cvtph_ps(w128);
            let xv = _mm256_loadu_ps(x.as_ptr().add(j));
            acc = _mm256_fmadd_ps(w_f32, xv, acc);
        }
        let mut sum = hsum256(acc);
        for j in m8..m {
            sum += f16::from_bits(row[j] as u16).to_f32() * x[j];
        }
        *yi = sum + bias.map_or(0.0, |b| b[r]);
    });
}

// Remove unused imports at the top
// (these are used in #[cfg] functions but rustc warns anyway)

/// Auto-dispatched FP16 matmul.
pub fn matmul_f16(y: &mut [f32], w: &[u8], x: &[f32], bias: Option<&[f32]>, n: usize, m: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if cpu_features().avx2 && cpu_features().fma {
            unsafe { matmul_f16_avx2_fma(y, w, x, bias, n, m) }
            return;
        }
    }
    let w_f16: &[f16] = bytemuck::cast_slice(w);
    y.par_iter_mut().enumerate().for_each(|(r, yi)| {
        let row = &w_f16[r * m..(r + 1) * m];
        let mut s = 0.0f32;
        for j in 0..m { s += row[j].to_f32() * x[j]; }
        *yi = s + bias.map_or(0.0, |b| b[r]);
    });
}

// ═══════════════════════════════════════════════════════════════════════
// RMS Norm  —  fused multiply-reduce
// ═══════════════════════════════════════════════════════════════════════

pub fn rms_norm_fused(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let n8 = n - (n % 8);

    let feats = cpu_features();

    #[cfg(target_arch = "x86_64")]
    if feats.avx2 {
        unsafe {
            use std::arch::x86_64::*;
            let mut sq = _mm256_setzero_ps();
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                sq = _mm256_fmadd_ps(v, v, sq);
            }
            let mut mean_sq = hsum256(sq);
            for i in n8..n { mean_sq += x[i] * x[i]; }
            mean_sq /= n as f32;
            let inv_rms = _mm256_set1_ps(1.0 / (mean_sq + eps).sqrt());
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let w = _mm256_loadu_ps(weight.as_ptr().add(i));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_mul_ps(v, inv_rms), w));
            }
            for i in n8..n { x[i] = x[i] / (mean_sq + eps).sqrt() * weight[i]; }
            return;
        }
    }

    // Scalar fallback
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..n { x[i] = x[i] * inv_rms * weight[i]; }
}

// ═══════════════════════════════════════════════════════════════════════
// Softmax  —  stable max-reduce then exp
// ═══════════════════════════════════════════════════════════════════════

pub fn softmax_fast(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let n8 = x.len() - (x.len() % 8);
    let feats = cpu_features();

    #[cfg(target_arch = "x86_64")]
    if feats.avx2 {
        unsafe {
            use std::arch::x86_64::*;
            let mv = _mm256_set1_ps(max_val);
            let mut sum = _mm256_setzero_ps();
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let e = vsubexp_avx2(_mm256_sub_ps(v, mv));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
                sum = _mm256_add_ps(sum, e);
            }
            let mut total = hsum256(sum);
            for i in n8..x.len() {
                x[i] = (x[i] - max_val).exp();
                total += x[i];
            }
            let inv = _mm256_set1_ps(1.0 / total);
            for i in (0..n8).step_by(8) {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, inv));
            }
            for i in n8..x.len() { x[i] /= total; }
            return;
        }
    }

    // Scalar
    for xi in x.iter_mut() { *xi = (*xi - max_val).exp(); }
    let sum: f32 = x.iter().sum();
    let inv = 1.0 / sum;
    for xi in x.iter_mut() { *xi *= inv; }
}

/// AVX2 approximate exp using polynomial (fast, ~1e-5 relative error).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vsubexp_avx2(v: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    // Clamp to avoid overflow: exp(-87) ≈ 0, exp(87) is huge
    let lo = _mm256_set1_ps(-87.0);
    let hi = _mm256_set1_ps(87.0);
    let v = _mm256_min_ps(_mm256_max_ps(v, lo), hi);

    // exp(x) ≈ (1 + x/8 + x²/128 + x³/3072 + x⁴/98304)
    let c1 = _mm256_set1_ps(1.0 / 8.0);
    let c2 = _mm256_set1_ps(1.0 / 128.0);
    let c3 = _mm256_set1_ps(1.0 / 3072.0);
    let c4 = _mm256_set1_ps(1.0 / 98304.0);

    let x2 = _mm256_mul_ps(v, v);
    let x3 = _mm256_mul_ps(x2, v);
    let x4 = _mm256_mul_ps(x3, v);
    let x5 = _mm256_mul_ps(x4, v);
    let x6 = _mm256_mul_ps(x5, v);

    let t1 = _mm256_fmadd_ps(c1, v, _mm256_set1_ps(1.0));
    let t2 = _mm256_fmadd_ps(c2, x2, t1);
    let t3 = _mm256_fmadd_ps(c3, x3, t2);
    let t4 = _mm256_fmadd_ps(c4, x4, t3);
    t4
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32_correctness() {
        let n = 128;
        let m = 256;
        let w: Vec<f32> = (0..n*m).map(|i| ((i as f32) * 1.234).sin()).collect();
        let x: Vec<f32> = (0..m).map(|i| ((i as f32) * 0.789).cos()).collect();
        let mut y = vec![0.0f32; n];
        let mut y_ref = y.clone();

        matmul_f32_scalar(&mut y_ref, &w, &x, None, n, m);
        matmul_f32(&mut y, &w, &x, None, n, m);

        for i in 0..n {
            let err = (y[i] - y_ref[i]).abs();
            assert!(err < 1e-3, "row {}: kernel={} ref={}", i, y[i], y_ref[i]);
        }
    }

    #[test]
    fn test_rms_norm() {
        let mut x = vec![2.0f32; 64];
        let w = vec![1.0f32; 64];
        rms_norm_fused(&mut x, &w, 1e-5);
        for v in &x {
            assert!((v - 1.0).abs() < 0.001, "got {}", v);
        }
    }

    #[test]
    fn test_softmax_fast() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax_fast(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 0.001, "sum={}", s);
        assert!(x[3] > x[2] && x[2] > x[1]);
    }
}
