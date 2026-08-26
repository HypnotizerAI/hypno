/// Data type enum for tensor elements.
///
/// Stored as u32 in the binary format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    /// 32-bit IEEE 754 floating point (4 bytes per element)
    FP32 = 0,
    /// 16-bit IEEE 754 floating point (2 bytes per element)
    FP16 = 1,
    /// 8-bit block quantization: 32 floats per block, u8 quantized + f16 scale
    Q8_0 = 2,
    /// 4-bit block quantization: 32 floats per block, 4-bit quantized + f16 scale
    Q4_0 = 3,
}

impl DType {
    /// Convert from the u32 value stored in the file.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::FP32),
            1 => Some(Self::FP16),
            2 => Some(Self::Q8_0),
            3 => Some(Self::Q4_0),
            _ => None,
        }
    }

    /// Size in bytes of one element in uncompressed form (for FP types).
    /// For quantized types, returns 0 — use `block_size` and `block_bytes` instead.
    pub fn element_size(self) -> usize {
        match self {
            Self::FP32 => 4,
            Self::FP16 => 2,
            Self::Q8_0 | Self::Q4_0 => 0,
        }
    }

    /// Number of elements per quantization block.
    pub fn block_size(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 32,
            _ => 1,
        }
    }

    /// Bytes per quantization block (scale + packed data).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 2 + 32,     // f16 scale + 32 u8
            Self::Q4_0 => 2 + 16,     // f16 scale + 32 * 4 bits
            _ => 0,
        }
    }

    /// Total bytes for `n` elements of this type.
    pub fn data_bytes(self, n_elems: usize) -> usize {
        match self {
            Self::FP32 => n_elems * 4,
            Self::FP16 => n_elems * 2,
            Self::Q8_0 => {
                let blocks = n_elems.div_ceil(32);
                blocks * 34 // 2 (scale) + 32 (u8)
            }
            Self::Q4_0 => {
                let blocks = n_elems.div_ceil(32);
                blocks * 18 // 2 (scale) + 16 (packed 4-bit)
            }
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q4_0 => write!(f, "Q4_0"),
        }
    }
}
