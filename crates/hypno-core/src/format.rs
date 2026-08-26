use crate::DType;

/// Magic bytes for `.hypno` files: "HYPN" in ASCII.
pub const MAGIC_BYTES: [u8; 4] = [0x48, 0x59, 0x50, 0x4E];

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;

/// All data payloads are aligned to this boundary.
pub const ALIGNMENT: u64 = 64;

/// Fixed-size header at the start of every `.hypno` file (16 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct HypnoHeader {
    /// Magic bytes: must equal [`MAGIC_BYTES`] = b"HYPN".
    pub magic: [u8; 4],
    /// Format version number.
    pub version: u32,
    /// Number of key-value metadata entries.
    pub metadata_kv_count: u32,
    /// Number of tensors in the file.
    pub tensor_count: u32,
}

impl HypnoHeader {
    pub fn new(metadata_kv_count: u32, tensor_count: u32) -> Self {
        Self {
            magic: MAGIC_BYTES,
            version: FORMAT_VERSION,
            metadata_kv_count,
            tensor_count,
        }
    }

    pub fn validate(&self) -> Result<(), FormatError> {
        if self.magic != MAGIC_BYTES {
            return Err(FormatError::InvalidMagic(self.magic));
        }
        if self.version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Metadata key-value pair (serialized form).
#[derive(Debug, Clone)]
pub struct MetaKV {
    pub key: String,
    pub value: String,
}

/// Serialized size of a metadata KV in bytes.
impl MetaKV {
    pub fn serialized_size(&self) -> usize {
        4 + self.key.len() + 4 + self.value.len()
    }
}

/// Metadata describing a single tensor in the file.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor name (e.g. "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Number of dimensions.
    pub ndim: u32,
    /// Shape array: [dim0, dim1, ...] in row-major order.
    pub shape: Vec<u64>,
    /// Data type of the stored tensor.
    pub dtype: DType,
    /// Absolute byte offset of the tensor data in the file.
    pub offset: u64,
    /// Size of the tensor data in bytes (padded to alignment).
    pub data_len: u64,
}

impl TensorMeta {
    /// Total number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    /// Uncompressed byte size of the raw elements (as f32).
    pub fn uncompressed_bytes(&self) -> usize {
        self.num_elements() * 4
    }

    /// Serialized size of this tensor metadata entry in the table.
    pub fn serialized_size(&self) -> usize {
        4 + self.name.len()  // name_len + name
        + 4                  // ndim
        + (self.ndim as usize) * 8  // shape
        + 4                  // dtype
        + 8                  // offset
        + 8                  // data_len
    }
}

/// The complete parsed contents of a `.hypno` file header/metadata.
#[derive(Debug, Clone)]
pub struct HypnoManifest {
    pub header: HypnoHeader,
    pub metadata: Vec<MetaKV>,
    pub tensors: Vec<TensorMeta>,
}

/// Errors encountered when parsing or validating a `.hypno` file.
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("invalid magic bytes: expected '{:?}', got '{:?}'", MAGIC_BYTES, .0)]
    InvalidMagic([u8; 4]),

    #[error("unsupported format version {0} (expected 1)")]
    UnsupportedVersion(u32),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("buffer too short: needed {needed} bytes, got {got}")]
    BufferTooShort { needed: usize, got: usize },

    #[error("unknown dtype code: {0}")]
    UnknownDType(u32),

    #[error("invalid tensor offset {offset}: out of file bounds ({file_size})")]
    InvalidOffset { offset: u64, file_size: u64 },
}
