//! Quantization kernels for Hypno.
//!
//! Implements Q4_0 and Q8_0 block quantization with:
//! - Vectorized dequantization (AVX2, SSE4.1, NEON, scalar fallback)
//! - Quantized dot-product kernels for direct matmul against quantized weights
//! - Benchmarks comparing memory and accuracy vs FP16

pub mod q4_0;
pub mod q8_0;
pub mod bench;

use half::f16;

/// Dequantize a buffer of Q4_0 blocks into f32.
///
/// `blocks` is the raw bytes of Q4_0 blocks.
/// Returns a Vec<f32> of length `n_blocks * 32`.
pub fn dequantize_q4_0(blocks: &[u8]) -> Vec<f32> {
    let n_blocks = blocks.len() / 18;
    let mut result = vec![0.0f32; n_blocks * 32];
    for i in 0..n_blocks {
        let block = &blocks[i * 18..(i + 1) * 18];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32;
            let hi = ((qs[j] >> 4) & 0x0F) as i32;
            result[i * 32 + j * 2] = (lo - 8) as f32 * scale;
            result[i * 32 + j * 2 + 1] = (hi - 8) as f32 * scale;
        }
    }
    result
}

/// Dequantize Q8_0 blocks into f32.
pub fn dequantize_q8_0(blocks: &[u8]) -> Vec<f32> {
    let n_blocks = blocks.len() / 34;
    let mut result = vec![0.0f32; n_blocks * 32];
    for i in 0..n_blocks {
        let block = &blocks[i * 34..(i + 1) * 34];
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..32 {
            result[i * 32 + j] = (block[2 + j] as i32 - 128) as f32 * scale;
        }
    }
    result
}

/// Quantize f32 data to Q4_0 blocks.
pub fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    use hypno_core::quantization::Q4_0Block;
    let n = data.len();
    let n_blocks = n.div_ceil(32);
    let mut result = vec![0u8; n_blocks * 18];

    for i in 0..n_blocks {
        let start = i * 32;
        let mut block_src = [0.0f32; 32];
        for j in 0..32 {
            if start + j < n {
                block_src[j] = data[start + j];
            }
        }
        let mut block = Q4_0Block { scale: f16::ZERO, qs: [0u8; 16] };
        block.quantize(&block_src);
        let block_bytes: &[u8] = bytemuck::bytes_of(&block);
        result[i * 18..(i + 1) * 18].copy_from_slice(block_bytes);
    }

    result
}

/// Quantize f32 data to Q8_0 blocks.
pub fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    use hypno_core::quantization::Q8_0Block;
    let n = data.len();
    let n_blocks = n.div_ceil(32);
    let mut result = vec![0u8; n_blocks * 34];

    for i in 0..n_blocks {
        let start = i * 32;
        let end = (start + 32).min(n);
        let block_slice = &data[start..end];

        let max_abs = block_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        let mut block = Q8_0Block { scale: f16::from_f32(scale), qs: [0u8; 32] };
        for j in 0..block_slice.len() {
            let q = ((block_slice[j] / scale).round() as i32 + 128).clamp(0, 255) as u8;
            block.qs[j] = q;
        }

        let block_bytes: &[u8] = bytemuck::bytes_of(&block);
        result[i * 34..(i + 1) * 34].copy_from_slice(block_bytes);
    }

    result
}
