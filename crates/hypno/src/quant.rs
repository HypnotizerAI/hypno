use half::f16;

/// Q4_0 block: 32 elements quantized to 4-bit integers with a shared f16 scale.
///
/// Layout (18 bytes total):
/// - bytes 0..2:    f16 scale factor
/// - bytes 2..18:   16 bytes of packed 4-bit values (32 nibbles)
///
/// Each nibble encodes a value 0..15. Dequantize: `val = (nibble - 8) * scale`
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
pub struct Q4_0Block {
    /// Scale factor as f16
    pub scale: f16,
    /// 32 4-bit quantized values packed into 16 bytes.
    /// Byte i stores: `qs[i*2]` in low nibble, `qs[i*2+1]` in high nibble.
    pub qs: [u8; 16],
}

impl Q4_0Block {
    pub const BLOCK_SIZE: usize = 32;
    pub const BLOCK_BYTES: usize = 18;

    /// Dequantize this block into `dst` (must have space for 32 f32 values).
    pub fn dequantize(&self, dst: &mut [f32; 32]) {
        let scale: f32 = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let lo = (byte & 0x0F) as i32;
            let hi = ((byte >> 4) & 0x0F) as i32;
            dst[i * 2] = (lo - 8) as f32 * scale;
            dst[i * 2 + 1] = (hi - 8) as f32 * scale;
        }
    }

    /// Dequantize this block into f32 slice starting at `dst_offset`.
    pub fn dequantize_into(&self, dst: &mut [f32], dst_offset: usize) {
        let scale: f32 = self.scale.to_f32();
        for i in 0..16 {
            let byte = self.qs[i];
            let lo = (byte & 0x0F) as i32;
            let hi = ((byte >> 4) & 0x0F) as i32;
            dst[dst_offset + i * 2] = (lo - 8) as f32 * scale;
            dst[dst_offset + i * 2 + 1] = (hi - 8) as f32 * scale;
        }
    }

    /// Quantize 32 f32 values into this block.
    pub fn quantize(&mut self, src: &[f32]) {
        // Find max absolute value for scaling
        let max_abs = src.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 {
            max_abs / 7.0 // range [-7, 7] maps to nibbles 1..15 (0 reserved for -8)
        } else {
            1.0
        };
        self.scale = f16::from_f32(scale);

        for i in 0..16 {
            let v0 = src[i * 2];
            let v1 = src[i * 2 + 1];

            let q0 = ((v0 / scale).round() as i32).clamp(-8, 7);
            let q1 = ((v1 / scale).round() as i32).clamp(-8, 7);

            let nib0 = ((q0 + 8) as u8) & 0x0F;
            let nib1 = ((q1 + 8) as u8) & 0x0F;

            self.qs[i] = nib0 | (nib1 << 4);
        }
    }
}

/// Q8_0 block: 32 elements quantized to 8-bit integers with a shared f16 scale.
///
/// Layout (34 bytes total):
/// - bytes 0..2:    f16 scale factor
/// - bytes 2..34:   32 u8 quantized values
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
pub struct Q8_0Block {
    pub scale: f16,
    pub qs: [u8; 32],
}

impl Q8_0Block {
    pub const BLOCK_SIZE: usize = 32;
    pub const BLOCK_BYTES: usize = 34;

    pub fn dequantize(&self, dst: &mut [f32; 32]) {
        let scale = self.scale.to_f32();
        for i in 0..32 {
            dst[i] = (self.qs[i] as i32 - 128) as f32 * scale;
        }
    }

    pub fn dequantize_into(&self, dst: &mut [f32], dst_offset: usize) {
        let scale = self.scale.to_f32();
        for i in 0..32 {
            dst[dst_offset + i] = (self.qs[i] as i32 - 128) as f32 * scale;
        }
    }
}

// ── Free functions for batch quantization/dequantization ──

/// Quantize a slice of f32 values into Q4_0 blocks. len must be divisible by 32.
pub fn quantize_f32_to_q4_0(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * Q4_0Block::BLOCK_BYTES);
    let mut block = Q4_0Block { scale: half::f16::ZERO, qs: [0u8; 16] };
    for i in 0..n_blocks {
        block.quantize(&data[i * 32..(i + 1) * 32]);
        out.extend_from_slice(bytemuck::bytes_of(&block));
    }
    out
}

/// Dequantize Q4_0 blocks back to f32. q4_data must be divisible by 18 bytes.
pub fn dequantize_q4_0(q4_data: &[u8]) -> Vec<f32> {
    let n_blocks = q4_data.len() / Q4_0Block::BLOCK_BYTES;
    let mut out = vec![0.0f32; n_blocks * 32];
    let mut tmp = [0.0f32; 32];
    for i in 0..n_blocks {
        let block: &Q4_0Block = bytemuck::from_bytes(
            &q4_data[i * Q4_0Block::BLOCK_BYTES..(i + 1) * Q4_0Block::BLOCK_BYTES]
        );
        block.dequantize(&mut tmp);
        out[i * 32..(i + 1) * 32].copy_from_slice(&tmp);
    }
    out
}

/// Extract a single row from a Q4_0-quantized 2D matrix.
///
/// The matrix has shape `[n_rows, row_width]` stored in row-major order.
/// Each Q4_0 block covers 32 consecutive elements. `row_width` must be
/// a multiple of 32.
pub fn q4_0_extract_row(q4_data: &[u8], row_idx: usize, row_width: usize) -> Vec<f32> {
    let blocks_per_row = row_width / 32;
    let row_start_block = row_idx * blocks_per_row;
    let mut out = vec![0.0f32; row_width];
    let mut tmp = [0.0f32; 32];
    for i in 0..blocks_per_row {
        let block_offset = (row_start_block + i) * Q4_0Block::BLOCK_BYTES;
        let block: &Q4_0Block = bytemuck::from_bytes(
            &q4_data[block_offset..block_offset + Q4_0Block::BLOCK_BYTES]
        );
        block.dequantize(&mut tmp);
        out[i * 32..(i + 1) * 32].copy_from_slice(&tmp);
    }
    out
}

/// Extract a single row from a Q8_0-quantized 2D matrix.
///
/// Each Q8_0 block covers 32 consecutive elements. `row_width` must be
/// a multiple of 32. Q8_0 blocks are 34 bytes each (2-byte f16 scale + 32 bytes u8 data).
pub fn q8_0_extract_row(q8_data: &[u8], row_idx: usize, row_width: usize) -> Vec<f32> {
    let blocks_per_row = row_width / 32;
    let row_start_block = row_idx * blocks_per_row;
    let block_bytes = Q8_0Block::BLOCK_BYTES;
    let mut out = vec![0.0f32; row_width];
    for i in 0..blocks_per_row {
        let bo = (row_start_block + i) * block_bytes;
        let scale = half::f16::from_le_bytes([q8_data[bo], q8_data[bo + 1]]).to_f32();
        let qs = &q8_data[bo + 2..bo + 34];
        for j in 0..32 {
            // Q8_0: value = (qs[j] - 128) * scale
            out[i * 32 + j] = (qs[j] as i32 - 128) as f32 * scale;
        }
    }
    out
}

/// Quantize a slice of f32 values into Q8_0 blocks. len must be divisible by 32.
pub fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * Q8_0Block::BLOCK_BYTES);
    for i in 0..n_blocks {
        let chunk = &data[i * 32..(i + 1) * 32];
        let max_abs = chunk.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = half::f16::from_f32(if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 });
        let block = Q8_0Block {
            scale,
            qs: std::array::from_fn(|j| ((chunk[j] / scale.to_f32()).round() as i32).clamp(-127, 127).wrapping_add(128) as u8),
        };
        out.extend_from_slice(bytemuck::bytes_of(&block));
    }
    out
}

/// Dequantize Q8_0 blocks back to f32. q8_data must be divisible by 34 bytes.
pub fn dequantize_q8_0(q8_data: &[u8]) -> Vec<f32> {
    let n_blocks = q8_data.len() / Q8_0Block::BLOCK_BYTES;
    let mut out = vec![0.0f32; n_blocks * 32];
    let mut tmp = [0.0f32; 32];
    for i in 0..n_blocks {
        let block: &Q8_0Block = bytemuck::from_bytes(
            &q8_data[i * Q8_0Block::BLOCK_BYTES..(i + 1) * Q8_0Block::BLOCK_BYTES]
        );
        block.dequantize(&mut tmp);
        out[i * 32..(i + 1) * 32].copy_from_slice(&tmp);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let src: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.1).collect();
        let mut block = Q4_0Block { scale: f16::ZERO, qs: [0u8; 16] };
        block.quantize(&src);
        let mut dst = [0f32; 32];
        block.dequantize(&mut dst);

        // Check that values are approximately preserved
        for i in 0..32 {
            let diff = (src[i] - dst[i]).abs();
            assert!(diff < 0.5, "element {} diff {} too large", i, diff);
        }
    }

    #[test]
    fn test_q4_0_zero() {
        let src = [0f32; 32];
        let mut block = Q4_0Block { scale: f16::ZERO, qs: [0u8; 16] };
        block.quantize(&src);
        let mut dst = [0f32; 32];
        block.dequantize(&mut dst);
        for i in 0..32 {
            assert!((dst[i]).abs() < 1e-6, "expected 0, got {}", dst[i]);
        }
    }
}
