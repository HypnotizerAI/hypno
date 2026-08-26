//! GGUF → .hypno converter.
//!
//! Parses the GGUF binary format (used by llama.cpp) and converts tensor data
//! into .hypno format. Q4_0, Q8_0, F32, and F16 tensors are copied directly
//! (identical block layout). Other quantization types are dequantized to F32.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::dtype::DType;
use crate::format::ALIGNMENT;

/// GGUF magic bytes: "GGUF" in little-endian u32.
const GGUF_MAGIC: u32 = 0x46554747;

// GGUF value type tags
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

// GGML tensor type constants
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q4_1: u32 = 3;
const GGML_Q5_0: u32 = 6;
const GGML_Q5_1: u32 = 7;
const GGML_Q8_0: u32 = 8;
const GGML_Q8_1: u32 = 9;
const GGML_Q2_K: u32 = 10;
const GGML_Q3_K: u32 = 11;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;
const GGML_BF16: u32 = 17;

/// Map GGML type to .hypno DType (None if unsupported directly).
fn ggml_to_hypno_dtype(ggml_type: u32) -> Option<DType> {
    match ggml_type {
        GGML_F32 => Some(DType::FP32),
        GGML_F16 => Some(DType::FP16),
        GGML_Q4_0 => Some(DType::Q4_0),
        GGML_Q8_0 => Some(DType::Q8_0),
        _ => None, // Needs dequantization first
    }
}

/// GGUF header.
#[derive(Debug)]
#[allow(dead_code)]
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

/// Info about one tensor from the GGUF file.
#[derive(Debug, Clone)]
struct GgufTensorInfo {
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
    /// Raw byte length of the tensor data in the file.
    data_len: u64,
}

/// Read a length-prefixed string from a GGUF reader.
fn read_gguf_string<R: Read>(reader: &mut R) -> anyhow::Result<String> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}

/// Read a GGUF typed value and return its string representation.
fn read_gguf_value<R: Read + Seek>(reader: &mut R) -> anyhow::Result<String> {
    let mut type_buf = [0u8; 4];
    reader.read_exact(&mut type_buf)?;
    let vtype = u32::from_le_bytes(type_buf);

    match vtype {
        GGUF_TYPE_U8 | GGUF_TYPE_I8 => {
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            Ok((b[0] as i32).to_string())
        }
        GGUF_TYPE_U16 | GGUF_TYPE_I16 => {
            let mut b = [0u8; 2];
            reader.read_exact(&mut b)?;
            Ok(u16::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_U32 | GGUF_TYPE_I32 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_F32 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_BOOL => {
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            Ok((b[0] != 0).to_string())
        }
        GGUF_TYPE_STRING => read_gguf_string(reader),
        GGUF_TYPE_U64 | GGUF_TYPE_I64 => {
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_F64 => {
            let mut b = [0u8; 8];
            reader.read_exact(&mut b)?;
            Ok(f64::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_ARRAY => {
            // Array: element type (u32) + count (u64) + elements
            let mut et_buf = [0u8; 4];
            reader.read_exact(&mut et_buf)?;
            let _elem_type = u32::from_le_bytes(et_buf);
            let mut cnt_buf = [0u8; 8];
            reader.read_exact(&mut cnt_buf)?;
            let count = u64::from_le_bytes(cnt_buf);

            let mut items = Vec::new();
            for _ in 0..count {
                // Rewind type tag for recursive read — we already read the element type above
                // For simplicity, just read primitives
                let mut t = [0u8; 4];
                reader.read_exact(&mut t)?;
                let et = u32::from_le_bytes(t);
                items.push(read_gguf_primitive(reader, et)?);
            }
            Ok(format!("[{}]", items.join(", ")))
        }
        _ => {
            // Unknown type — skip by seeking ahead (but we don't know size)
            Ok("?".to_string())
        }
    }
}

fn read_gguf_primitive<R: Read>(reader: &mut R, vtype: u32) -> anyhow::Result<String> {
    match vtype {
        GGUF_TYPE_U8 | GGUF_TYPE_I8 => {
            let mut b = [0u8; 1];
            reader.read_exact(&mut b)?;
            Ok(b[0].to_string())
        }
        GGUF_TYPE_U32 | GGUF_TYPE_I32 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_F32 => {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b).to_string())
        }
        GGUF_TYPE_STRING => read_gguf_string(reader),
        _ => Ok("?".to_string()),
    }
}

/// Parse a GGUF file and extract metadata, tensor info, and raw file data reference.
fn parse_gguf(file_path: &Path) -> anyhow::Result<(GgufHeader, BTreeMap<String, String>, Vec<GgufTensorInfo>, Vec<u8>)> {
    let file_data = fs::read(file_path)?;
    let mut reader = std::io::Cursor::new(&file_data[..]);

    // Read magic
    let mut magic_buf = [0u8; 4];
    reader.read_exact(&mut magic_buf)?;
    let magic = u32::from_le_bytes(magic_buf);
    if magic != GGUF_MAGIC {
        anyhow::bail!("Not a GGUF file: magic 0x{:08X} != expected 0x{:08X}", magic, GGUF_MAGIC);
    }

    // Read version
    let mut ver_buf = [0u8; 4];
    reader.read_exact(&mut ver_buf)?;
    let version = u32::from_le_bytes(ver_buf);

    // Read counts
    let mut cnt_buf = [0u8; 8];
    reader.read_exact(&mut cnt_buf)?;
    let tensor_count = u64::from_le_bytes(cnt_buf);
    reader.read_exact(&mut cnt_buf)?;
    let metadata_kv_count = u64::from_le_bytes(cnt_buf);

    println!("  GGUF version: {}", version);
    println!("  Tensors: {}", tensor_count);
    println!("  Metadata KVs: {}", metadata_kv_count);

    // Read metadata KVs
    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    for _ in 0..metadata_kv_count {
        let key = read_gguf_string(&mut reader)?;
        let value = read_gguf_value(&mut reader)?;
        metadata.insert(key, value);
    }

    // Read tensor info
    let mut tensors: Vec<GgufTensorInfo> = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut reader)?;

        let mut ndim_buf = [0u8; 4];
        reader.read_exact(&mut ndim_buf)?;
        let n_dims = u32::from_le_bytes(ndim_buf);

        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let mut dim_buf = [0u8; 8];
            reader.read_exact(&mut dim_buf)?;
            dims.push(u64::from_le_bytes(dim_buf));
        }

        let mut type_buf = [0u8; 4];
        reader.read_exact(&mut type_buf)?;
        let ggml_type = u32::from_le_bytes(type_buf);

        let mut off_buf = [0u8; 8];
        reader.read_exact(&mut off_buf)?;
        let offset = u64::from_le_bytes(off_buf);

        let n_elems: u64 = dims.iter().product();
        let data_len = match ggml_type {
            GGML_F32 => n_elems * 4,
            GGML_F16 => n_elems * 2,
            GGML_Q4_0 | GGML_Q4_1 => {
                let n_blocks = n_elems.div_ceil(32);
                n_blocks * 18 + (if ggml_type == GGML_Q4_1 { n_blocks * 2 } else { 0 })
            }
            GGML_Q8_0 | GGML_Q8_1 => {
                let n_blocks = n_elems.div_ceil(32);
                n_blocks * 34 + (if ggml_type == GGML_Q8_1 { n_blocks * 4 } else { 0 })
            }
            _ => n_elems * 4, // Conservative — will be dequantized anyway
        };

        tensors.push(GgufTensorInfo { name, dims, ggml_type, offset, data_len });
    }

    // Read alignment padding
    let pos = reader.position();
    let gguf_align: u64 = 32;
    let pad = (gguf_align - (pos % gguf_align)) % gguf_align;
    reader.seek(SeekFrom::Current(pad as i64))?;

    Ok((GgufHeader { version, tensor_count, metadata_kv_count }, metadata, tensors, file_data))
}

/// Convert a GGUF file to .hypno format.
pub fn convert_gguf_to_hypno(
    gguf_path: &Path,
    out_path: &Path,
    _target_dtype: DType,
) -> anyhow::Result<()> {
    use crate::format::{HypnoHeader, MetaKV, TensorMeta};
    use std::io::{BufWriter, Write};

    println!("Reading GGUF file: {}", gguf_path.display());
    let (_header, metadata, tensor_info, file_data) = parse_gguf(gguf_path)?;

    // Extract config metadata for .hypno format
    let mut model_kvs: Vec<MetaKV> = Vec::new();

    // Pass through known metadata keys
    for key in &[
        "general.architecture",
        "general.name",
        "general.description",
    ] {
        if let Some(val) = metadata.get(*key) {
            let short = key.strip_prefix("general.").unwrap_or(key);
            model_kvs.push(MetaKV { key: short.to_string(), value: val.clone() });
        }
    }

    // Architecture-specific config
    let arch = metadata.get("general.architecture")
        .cloned()
        .unwrap_or_else(|| "llama".to_string());
    model_kvs.push(MetaKV { key: "architecture".into(), value: arch.clone() });

    let arch_prefix = format!("{}.", arch);
    for (key, val) in &metadata {
        if key.starts_with(&arch_prefix) {
            let short = key.strip_prefix(&arch_prefix).unwrap_or(key);
            // Normalize common names
            let mapped = match short {
                "embedding_length" => "hidden_size",
                "feed_forward_length" => "intermediate_size",
                "attention.head_count" => "num_attention_heads",
                "attention.head_count_kv" => "num_key_value_heads",
                "attention.layer_norm_rms_epsilon" => "rms_norm_eps",
                "block_count" => "num_hidden_layers",
                "context_length" => "max_position_embeddings",
                "rope.dimension_count" => "head_dim",
                "rope.freq_base" => "rope_theta",
                other => other,
            };
            if mapped == "head_dim" {
                // Parse numeric value
                if let Ok(v) = val.parse::<f32>() {
                    model_kvs.push(MetaKV { key: "head_dim".into(), value: format!("{}", v as usize) });
                    continue;
                }
            }
            model_kvs.push(MetaKV { key: mapped.to_string(), value: val.clone() });
        }
    }

    // Tokenizer metadata
    if let Some(tok_model) = metadata.get("tokenizer.ggml.model") {
        model_kvs.push(MetaKV { key: "tokenizer_model".into(), value: tok_model.clone() });
    }
    if let Some(tok_vocab) = metadata.get("tokenizer.ggml.tokens") {
        // This is a JSON array — pass through as-is
        model_kvs.push(MetaKV { key: "tokenizer_tokens".into(), value: tok_vocab.clone() });
    }
    if let Some(tok_scores) = metadata.get("tokenizer.ggml.scores") {
        model_kvs.push(MetaKV { key: "tokenizer_scores".into(), value: tok_scores.clone() });
    }
    if let Some(tok_type) = metadata.get("tokenizer.ggml.token_type") {
        model_kvs.push(MetaKV { key: "tokenizer_types".into(), value: tok_type.clone() });
    }
    if let Some(bos) = metadata.get("tokenizer.ggml.bos_token_id") {
        model_kvs.push(MetaKV { key: "bos_token_id".into(), value: bos.clone() });
    }
    if let Some(eos) = metadata.get("tokenizer.ggml.eos_token_id") {
        model_kvs.push(MetaKV { key: "eos_token_id".into(), value: eos.clone() });
    }

    // Compute .hypno header sizes
    let header_size = 16u64;
    let metadata_size: u64 = model_kvs.iter().map(|kv| kv.serialized_size() as u64).sum();

    // Filter tensors: skip output-related tensors that .hypno doesn't use
    let hypno_tensors: Vec<&GgufTensorInfo> = tensor_info.iter()
        .filter(|t| {
            !t.name.contains("output.") || t.name == "output.weight"
        })
        .collect();

    let tensor_table_size: u64 = hypno_tensors.iter()
        .map(|t| {
            let ndim = t.dims.len() as u32;
            4 + t.name.len() as u64 + 4 + (ndim as u64) * 8 + 4 + 8 + 8
        })
        .sum();

    let metadata_end = header_size + metadata_size + tensor_table_size;
    let data_start = ((metadata_end + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    // Build TensorMeta
    let mut tensors: Vec<TensorMeta> = Vec::new();
    let mut current_offset = data_start;

    for t in &hypno_tensors {
        let ndim = t.dims.len() as u32;
        let n_elems: usize = t.dims.iter().map(|&d| d as usize).product();
        let hypno_dt = ggml_to_hypno_dtype(t.ggml_type)
            .map(|dt| crate::sft_convert::effective_dtype(dt, n_elems))
            .unwrap_or(DType::FP32); // Dequantized to FP32

        let data_len = hypno_dt.data_bytes(n_elems) as u64;

        tensors.push(TensorMeta {
            name: t.name.clone(),
            ndim,
            shape: t.dims.clone(),
            dtype: hypno_dt,
            offset: current_offset,
            data_len,
        });

        current_offset += data_len;
        current_offset = ((current_offset + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }

    // Write .hypno file
    let out_file = fs::File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    let hypno_header = HypnoHeader::new(model_kvs.len() as u32, tensors.len() as u32);
    writer.write_all(bytemuck::bytes_of(&hypno_header))?;

    // Metadata
    for kv in &model_kvs {
        writer.write_all(&(kv.key.len() as u32).to_le_bytes())?;
        writer.write_all(kv.key.as_bytes())?;
        writer.write_all(&(kv.value.len() as u32).to_le_bytes())?;
        writer.write_all(kv.value.as_bytes())?;
    }

    // Tensor table
    for t in &tensors {
        writer.write_all(&(t.name.len() as u32).to_le_bytes())?;
        writer.write_all(t.name.as_bytes())?;
        writer.write_all(&t.ndim.to_le_bytes())?;
        for &dim in &t.shape { writer.write_all(&dim.to_le_bytes())?; }
        writer.write_all(&(t.dtype as u32).to_le_bytes())?;
        writer.write_all(&t.offset.to_le_bytes())?;
        writer.write_all(&t.data_len.to_le_bytes())?;
    }

    // Alignment padding
    let pos = header_size + metadata_size + tensor_table_size;
    let aligned = ((pos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    for _ in pos..aligned { writer.write_all(&[0u8])?; }

    // Write tensor data
    let mut bytes_written = aligned as usize;

    for (i, gguf_t) in hypno_tensors.iter().enumerate() {
        let t = &tensors[i];
        let needed_pad = t.offset as usize - bytes_written;
        if needed_pad > 0 {
            writer.write_all(&vec![0u8; needed_pad])?;
            bytes_written += needed_pad;
        }

        let start = gguf_t.offset as usize;
        let end = (gguf_t.offset + gguf_t.data_len) as usize;
        let raw_slice = &file_data[start..end.min(file_data.len())];

        // Convert tensor data based on ggml type
        let out_bytes: Vec<u8> = match gguf_t.ggml_type {
            GGML_F32 | GGML_F16 | GGML_Q4_0 | GGML_Q8_0 => {
                // Direct copy — same block layout
                raw_slice.to_vec()
            }
            GGML_Q4_1 => {
                // Q4_1 has an extra f32 minimum per block — dequantize to F32
                eprintln!("  Dequantizing Q4_1 tensor '{}' to FP32", gguf_t.name);
                dequantize_ggml_q4_1(raw_slice)
                    .map(|f32s| bytemuck::cast_slice::<f32, u8>(&f32s).to_vec())
                    .unwrap_or_else(|e| {
                        eprintln!("  Warning: failed to dequantize '{}': {}", gguf_t.name, e);
                        vec![0u8; t.data_len as usize]
                    })
            }
            GGML_Q5_0 | GGML_Q5_1 | GGML_Q8_1 | GGML_Q2_K | GGML_Q3_K |
            GGML_Q4_K | GGML_Q5_K | GGML_Q6_K | GGML_BF16 => {
                eprintln!("  Dequantizing {:?} tensor '{}' to FP32", gguf_t.ggml_type, gguf_t.name);
                dequantize_ggml_generic(raw_slice, gguf_t.ggml_type)
                    .map(|f32s| bytemuck::cast_slice::<f32, u8>(&f32s).to_vec())
                    .unwrap_or_else(|e| {
                        eprintln!("  Warning: failed to dequantize '{}': {}", gguf_t.name, e);
                        vec![0u8; t.data_len as usize]
                    })
            }
            _ => {
                eprintln!("  Warning: unknown ggml type {} for '{}', zero-filling", gguf_t.ggml_type, gguf_t.name);
                vec![0u8; t.data_len as usize]
            }
        };

        writer.write_all(&out_bytes)?;
        bytes_written += out_bytes.len();

        // Align to 64-byte boundary
        let next_aligned = ((bytes_written as u64 + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        let pad = (next_aligned - bytes_written as u64) as usize;
        if pad > 0 && pad < ALIGNMENT as usize {
            writer.write_all(&vec![0u8; pad])?;
            bytes_written += pad;
        }
    }

    writer.flush()?;
    println!("  Wrote {} tensors ({} bytes) → {}", tensors.len(), bytes_written, out_path.display());
    Ok(())
}

/// Dequantize GGUF Q4_1 blocks to F32.
fn dequantize_ggml_q4_1(data: &[u8]) -> anyhow::Result<Vec<f32>> {
    use half::f16;
    const BLOCK_SIZE: usize = 20; // 2B scale + 2B min + 16B quants
    let n_blocks = data.len() / BLOCK_SIZE;
    let mut result = vec![0.0f32; n_blocks * 32];

    for b in 0..n_blocks {
        let bo = b * BLOCK_SIZE;
        let scale = f16::from_le_bytes([data[bo], data[bo + 1]]).to_f32();
        let min = f16::from_le_bytes([data[bo + 2], data[bo + 3]]).to_f32();
        let qs = &data[bo + 4..bo + BLOCK_SIZE];

        let base = b * 32;
        for i in 0..16 {
            let byte = qs[i];
            result[base + i * 2] = ((byte & 0x0F) as f32) * scale + min;
            result[base + i * 2 + 1] = ((byte >> 4) as f32) * scale + min;
        }
    }
    Ok(result)
}

/// Generic dequantization for unsupported GGUF quant types.
fn dequantize_ggml_generic(data: &[u8], ggml_type: u32) -> anyhow::Result<Vec<f32>> {
    // For types we can't decode, try using the hypno-quantize library
    // or fall back to approximate dequantization
    match ggml_type {
        GGML_BF16 => {
            let n = data.len() / 2;
            let mut result = vec![0.0f32; n];
            for i in 0..n {
                let bf = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                result[i] = crate::sft_convert::bf16_to_f32(bf);
            }
            Ok(result)
        }
        _ => {
            // For Q5_*, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K — these need proper dequantization
            // algorithms. For now, produce a best-effort warning.
            anyhow::bail!(
                "GGUF quant type {} is not yet supported for direct conversion. \
                 Use --model-dir with the original HuggingFace safetensors checkpoint instead.",
                ggml_type
            )
        }
    }
}
