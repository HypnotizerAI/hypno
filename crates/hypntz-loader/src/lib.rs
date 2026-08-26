//! Zero-copy memory-mapped `.hypno` file reader.
//!
//! Maps the file into virtual address space and parses the header + tensor
//! registry without copying tensor payloads. Tensor data is accessed via
//! direct pointers into the mapped region.
//!
//! ## Platform support
//! - Linux / macOS: `mmap`
//! - Windows: `CreateFileMappingW` + `MapViewOfFile`

mod mmap;

use hypntz_core::{
    DType, FormatError, HypnoHeader, HypnoManifest, MetaKV, TensorMeta,
    ALIGNMENT,
};
use std::path::Path;

/// A zero-copy view of a `.hypno` model file.
///
/// On construction, the file is memory-mapped and the header/metadata is parsed.
/// Tensor data remains in the mapped region and is accessed via pointer.
pub struct HypnoModel {
    /// The underlying memory map (keeps the mapping alive).
    _mmap: mmap::Mmap,
    /// Pointer to the start of the mapped region.
    data: *const u8,
    /// Total mapped size in bytes.
    size: usize,
    /// Parsed file manifest.
    pub manifest: HypnoManifest,
}

// The pointer is to memory-mapped read-only data — safe to share across threads.
unsafe impl Send for HypnoModel {}
unsafe impl Sync for HypnoModel {}

impl HypnoModel {
    /// Open and memory-map a `.hypno` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, FormatError> {
        let mmap = mmap::Mmap::open(path.as_ref())?;
        let data = mmap.as_ptr();
        let size = mmap.len();

        // Parse header
        if size < 16 {
            return Err(FormatError::BufferTooShort { needed: 16, got: size });
        }
        let header: &HypnoHeader = unsafe { &*(data as *const HypnoHeader) };
        header.validate()?;

        let mut cursor = 16usize;

        // Parse metadata KVs
        let mut metadata = Vec::with_capacity(header.metadata_kv_count as usize);
        for _ in 0..header.metadata_kv_count {
            if cursor + 4 > size {
                return Err(FormatError::BufferTooShort { needed: cursor + 4, got: size });
            }
            let key_len = u32::from_le_bytes(read4(data, cursor)) as usize;
            cursor += 4;

            if cursor + key_len > size {
                return Err(FormatError::BufferTooShort { needed: cursor + key_len, got: size });
            }
            let key = std::str::from_utf8(&slice(data, cursor, key_len))?.to_string();
            cursor += key_len;

            if cursor + 4 > size {
                return Err(FormatError::BufferTooShort { needed: cursor + 4, got: size });
            }
            let value_len = u32::from_le_bytes(read4(data, cursor)) as usize;
            cursor += 4;

            if cursor + value_len > size {
                return Err(FormatError::BufferTooShort { needed: cursor + value_len, got: size });
            }
            let value = std::str::from_utf8(&slice(data, cursor, value_len))?.to_string();
            cursor += value_len;

            metadata.push(MetaKV { key, value });
        }

        // Parse tensor metadata table
        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        let file_size = size as u64;
        for _ in 0..header.tensor_count {
            let name_len = read_u32(data, &mut cursor, size)? as usize;
            let name = std::str::from_utf8(&slice(data, cursor, name_len))?.to_string();
            cursor += name_len;

            let ndim = read_u32(data, &mut cursor, size)?;
            let mut shape = Vec::with_capacity(ndim as usize);
            for _ in 0..ndim {
                if cursor + 8 > size {
                    return Err(FormatError::BufferTooShort { needed: cursor + 8, got: size });
                }
                let dim = u64::from_le_bytes(read8(data, cursor));
                cursor += 8;
                shape.push(dim);
            }

            let dtype_raw = read_u32(data, &mut cursor, size)?;
            let dtype = DType::from_u32(dtype_raw)
                .ok_or(FormatError::UnknownDType(dtype_raw))?;

            let offset = read_u64(data, &mut cursor, size)?;
            let data_len = read_u64(data, &mut cursor, size)?;

            if offset > file_size || offset + data_len > file_size {
                return Err(FormatError::InvalidOffset { offset, file_size });
            }

            tensors.push(TensorMeta {
                name,
                ndim,
                shape,
                dtype,
                offset,
                data_len,
            });
        }

        Ok(Self {
            _mmap: mmap,
            data,
            size,
            manifest: HypnoManifest {
                header: *header,
                metadata,
                tensors,
            },
        })
    }

    /// Get a direct pointer to a tensor's data in the mapped memory.
    ///
    /// Returns `None` if the tensor name is not found.
    /// The pointer is valid for the lifetime of the `HypnoModel`.
    pub fn get_tensor_data(&self, name: &str) -> Option<(&[u8], DType)> {
        let tensor = self.manifest.tensors.iter().find(|t| t.name == name)?;
        let offset = tensor.offset as usize;
        let len = tensor.data_len as usize;
        if offset + len > self.size {
            return None;
        }
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(self.data.add(offset), len) };
        Some((bytes, tensor.dtype))
    }

    /// Find a tensor metadata entry by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.manifest.tensors.iter().find(|t| t.name == name)
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.manifest.metadata.iter()
            .find(|kv| kv.key == key)
            .map(|kv| kv.value.as_str())
    }
}

// -- Raw byte-reading helpers that work on mapped pointers --

fn read_u32(data: *const u8, cursor: &mut usize, size: usize) -> Result<u32, FormatError> {
    if *cursor + 4 > size {
        return Err(FormatError::BufferTooShort { needed: *cursor + 4, got: size });
    }
    let val = u32::from_le_bytes(read4(data, *cursor));
    *cursor += 4;
    Ok(val)
}

fn read_u64(data: *const u8, cursor: &mut usize, size: usize) -> Result<u64, FormatError> {
    if *cursor + 8 > size {
        return Err(FormatError::BufferTooShort { needed: *cursor + 8, got: size });
    }
    let val = u64::from_le_bytes(read8(data, *cursor));
    *cursor += 8;
    Ok(val)
}

#[inline(always)]
fn read4(data: *const u8, offset: usize) -> [u8; 4] {
    unsafe {
        let p = data.add(offset) as *const [u8; 4];
        *p
    }
}

#[inline(always)]
fn read8(data: *const u8, offset: usize) -> [u8; 8] {
    unsafe {
        let p = data.add(offset) as *const [u8; 8];
        *p
    }
}

#[inline(always)]
fn slice<'a>(data: *const u8, offset: usize, len: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(data.add(offset), len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_hypno() -> (Vec<u8>, tempfile::NamedTempFile) {
        use hypntz_core::quantization::Q4_0Block;

        let mut buf = Vec::new();

        // Header
        let header = HypnoHeader::new(2, 2);
        buf.extend_from_slice(bytemuck::bytes_of(&header));

        // Metadata KVs
        let kvs = [
            ("architecture", "llama"),
            ("hidden_size", "64"),
        ];
        for (k, v) in &kvs {
            buf.extend_from_slice(&(k.len() as u32).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        }

        // We'll create a tensor with FP32 data and one with Q4_0
        let fp32_data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let fp32_bytes: &[u8] = bytemuck::cast_slice(&fp32_data);

        let mut q4_block = Q4_0Block { scale: half::f16::ZERO, qs: [0u8; 16] };
        let q4_src: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        q4_block.quantize(&q4_src);
        let q4_bytes: &[u8] = bytemuck::bytes_of(&q4_block);

        // Calculate offsets (both 64-byte aligned)
        let metadata_end_approx = buf.len()
            // rough estimate for tensor table
            + 4 + 10 + 4 + 1 * 8 + 4 + 8 + 8  // tensor 0
            + 4 + 10 + 4 + 1 * 8 + 4 + 8 + 8; // tensor 1
        let data_start = ((metadata_end_approx as u64 + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        let fp32_offset = data_start;
        let q4_offset = (data_start + fp32_bytes.len() as u64 + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;

        // Tensor table
        // Tensor 0: FP32
        let name0 = "weight";
        buf.extend_from_slice(&(name0.len() as u32).to_le_bytes());
        buf.extend_from_slice(name0.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&64u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&0u32.to_le_bytes()); // dtype = FP32
        buf.extend_from_slice(&fp32_offset.to_le_bytes());
        buf.extend_from_slice(&(fp32_bytes.len() as u64).to_le_bytes());

        // Tensor 1: Q4_0
        let name1 = "qweight";
        buf.extend_from_slice(&(name1.len() as u32).to_le_bytes());
        buf.extend_from_slice(name1.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&32u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&3u32.to_le_bytes()); // dtype = Q4_0
        buf.extend_from_slice(&q4_offset.to_le_bytes());
        buf.extend_from_slice(&(q4_bytes.len() as u64).to_le_bytes());

        // Pad to data_start
        while buf.len() < data_start as usize {
            buf.push(0);
        }

        // Write FP32 data
        buf.extend_from_slice(fp32_bytes);

        // Pad to q4_offset
        while buf.len() < q4_offset as usize {
            buf.push(0);
        }

        // Write Q4_0 data
        buf.extend_from_slice(q4_bytes);

        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&buf).unwrap();
        f.flush().unwrap();
        (buf, f)
    }

    #[test]
    fn test_open_and_read() {
        let (_buf, file) = create_test_hypno();
        let model = HypnoModel::open(file.path()).unwrap();

        assert_eq!(model.manifest.header.metadata_kv_count, 2);
        assert_eq!(model.manifest.header.tensor_count, 2);
        assert_eq!(model.get_metadata("architecture"), Some("llama"));
        assert_eq!(model.get_metadata("hidden_size"), Some("64"));

        // Read FP32 tensor
        let (data, dtype) = model.get_tensor_data("weight").unwrap();
        assert_eq!(dtype, DType::FP32);
        let floats: &[f32] = bytemuck::cast_slice(data);
        assert_eq!(floats.len(), 64);
        assert!((floats[0] - 0.0).abs() < 0.001);
        assert!((floats[63] - 63.0).abs() < 0.001);

        // Read Q4_0 tensor
        let (data, dtype) = model.get_tensor_data("qweight").unwrap();
        assert_eq!(dtype, DType::Q4_0);
        assert_eq!(data.len(), 18); // Q4_0 block size
    }
}
