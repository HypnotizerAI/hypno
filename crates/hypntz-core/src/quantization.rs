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
