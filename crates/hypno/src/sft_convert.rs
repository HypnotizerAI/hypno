//! Converter: reads HuggingFace safetensors and writes `.hypno` format.

use crate::dtype::DType;
use crate::format::{HypnoHeader, MetaKV, TensorMeta, ALIGNMENT};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Determine actual storage dtype: small 1D tensors stay FP32 even when quantizing.
pub fn effective_dtype(target_dtype: DType, n_elems: usize) -> DType {
    if n_elems <= 4096 && target_dtype != DType::FP32 && target_dtype != DType::FP16 {
        DType::FP32
    } else {
        target_dtype
    }
}

/// Convert a HuggingFace model directory to `.hypno` format.
pub fn convert(model_dir: &Path, out_path: &Path, target_dtype: DType, col_major: bool) -> anyhow::Result<()> {
    // 1. Load config.json
    let config_path = model_dir.join("config.json");
    let config: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?,
    )?;

    // 2. Load tokenizer.json (optional)
    let tokenizer_json = model_dir.join("tokenizer.json")
        .exists()
        .then(|| fs::read_to_string(model_dir.join("tokenizer.json")).ok())
        .flatten();

    // 3. Find all .safetensors files
    let mut safetensor_files: Vec<_> = fs::read_dir(model_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") {
                Some(entry.path())
            } else {
                None
            }
        })
        .collect();
    safetensor_files.sort();

    if safetensor_files.is_empty() {
        anyhow::bail!("No .safetensors files found in {}", model_dir.display());
    }

    println!("Found {} safetensors file(s)", safetensor_files.len());

    // 4. Extract metadata from config.json
    let mut metadata_kvs: Vec<MetaKV> = Vec::new();

    // Architecture
    let arch = config["architectures"][0]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    metadata_kvs.push(MetaKV { key: "architecture".into(), value: arch.clone() });

    // Model dimensions
    let hidden_size = config["hidden_size"].as_u64().unwrap_or(0) as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(0) as usize;
    let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(0) as usize;
    let num_key_value_heads = config["num_key_value_heads"]
        .as_u64()
        .unwrap_or(num_attention_heads as u64) as usize;
    let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
    let max_position_embeddings = config["max_position_embeddings"].as_u64().unwrap_or(2048) as usize;
    let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let rope_theta = config.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);

    metadata_kvs.push(MetaKV { key: "hidden_size".into(), value: hidden_size.to_string() });
    metadata_kvs.push(MetaKV { key: "intermediate_size".into(), value: intermediate_size.to_string() });
    metadata_kvs.push(MetaKV { key: "num_attention_heads".into(), value: num_attention_heads.to_string() });
    metadata_kvs.push(MetaKV { key: "num_key_value_heads".into(), value: num_key_value_heads.to_string() });
    metadata_kvs.push(MetaKV { key: "num_hidden_layers".into(), value: num_hidden_layers.to_string() });
    metadata_kvs.push(MetaKV { key: "vocab_size".into(), value: vocab_size.to_string() });
    metadata_kvs.push(MetaKV { key: "max_position_embeddings".into(), value: max_position_embeddings.to_string() });
    metadata_kvs.push(MetaKV { key: "rms_norm_eps".into(), value: rms_norm_eps.to_string() });
    metadata_kvs.push(MetaKV { key: "rope_theta".into(), value: rope_theta.to_string() });

    // Weight layout
    metadata_kvs.push(MetaKV {
        key: "weight_layout".into(),
        value: if col_major { "col_major" } else { "row_major" }.into(),
    });

    // Embed tokenizer.json if available
    if let Some(ref tj) = tokenizer_json {
        metadata_kvs.push(MetaKV { key: "tokenizer_json".into(), value: tj.clone() });
    }

    // 5. Collect tensor metadata
    // First pass: gather all tensor names and shapes from safetensors
    let mut tensor_info: BTreeMap<String, (Vec<usize>, DType)> = BTreeMap::new();

    for sf_path in &safetensor_files {
        let file_data = fs::read(sf_path)?;
        let sf = safetensors::SafeTensors::deserialize(&file_data)?;

        for name in sf.names() {
            let view = sf.tensor(&name)?;
            let shape: Vec<usize> = view.shape().to_vec();
            // All safetensors weights are FP32 or FP16 in the file
            let stored_dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::FP32,
                safetensors::Dtype::F16 => DType::FP16,
                safetensors::Dtype::BF16 => DType::FP32, // Convert BF16 to FP32
                _ => DType::FP32,
            };
            tensor_info.insert(name.to_string(), (shape, stored_dtype));
        }
    }

    println!("Found {} tensors", tensor_info.len());

    // 6. Calculate offsets and build tensor metadata
    // First, compute the total header/metadata size to know where data starts
    let header_size = 16u64;

    let metadata_size: u64 = metadata_kvs.iter()
        .map(|kv| kv.serialized_size() as u64)
        .sum();

    let tensor_table_size: u64 = tensor_info.iter()
        .map(|(name, (shape, _))| {
            let ndim = shape.len() as u32;
            4 + name.len() as u64 + 4 + (ndim as u64) * 8 + 4 + 8 + 8
        })
        .sum();

    let metadata_end = header_size + metadata_size + tensor_table_size;
    // Align data start to 64-byte boundary
    let data_start = ((metadata_end + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    // Build tensor metadata with offsets
    let mut tensors: Vec<TensorMeta> = Vec::new();
    let mut current_offset = data_start;

    for (name, (shape, _stored_dtype)) in &tensor_info {
        let ndim = shape.len() as u32;
        let n_elems: usize = shape.iter().product();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

        // Calculate data length based on effective dtype (small 1D stay FP32)
        let edt = effective_dtype(target_dtype, n_elems);
        let data_len = edt.data_bytes(n_elems) as u64;

        tensors.push(TensorMeta {
            name: name.clone(),
            ndim,
            shape: shape_u64,
            dtype: edt,
            offset: current_offset,
            data_len,
        });

        current_offset += data_len;
        // Align next tensor to 64-byte boundary
        current_offset = ((current_offset + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }

    let total_file_size = current_offset;
    println!("File size estimate: {} bytes ({:.2} MB)", total_file_size, total_file_size as f64 / 1_048_576.0);
    println!("Data start offset: {} (0x{:X})", data_start, data_start);

    // 7. Write the file
    let out_file = fs::File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);

    // Write header
    let header = HypnoHeader::new(metadata_kvs.len() as u32, tensors.len() as u32);
    writer.write_all(bytemuck::bytes_of(&header))?;

    // Write metadata KVs
    for kv in &metadata_kvs {
        writer.write_all(&(kv.key.len() as u32).to_le_bytes())?;
        writer.write_all(kv.key.as_bytes())?;
        writer.write_all(&(kv.value.len() as u32).to_le_bytes())?;
        writer.write_all(kv.value.as_bytes())?;
    }

    // Write tensor metadata table
    for t in &tensors {
        writer.write_all(&(t.name.len() as u32).to_le_bytes())?;
        writer.write_all(t.name.as_bytes())?;
        writer.write_all(&t.ndim.to_le_bytes())?;
        for &dim in &t.shape {
            writer.write_all(&dim.to_le_bytes())?;
        }
        writer.write_all(&(t.dtype as u32).to_le_bytes())?;
        writer.write_all(&t.offset.to_le_bytes())?;
        writer.write_all(&t.data_len.to_le_bytes())?;
    }

    // Pad to data_start
    let pos = header_size + metadata_size + tensor_table_size;
    let aligned = ((pos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    for _ in pos..aligned {
        writer.write_all(&[0u8])?;
    }

    // 8. Write tensor data
    let mut bytes_written = aligned as usize;
    let mut buffer = Vec::new();

    for (i, sf_path) in safetensor_files.iter().enumerate() {
        println!("Processing file {}/{}: {}", i + 1, safetensor_files.len(), sf_path.display());
        let file_data = fs::read(sf_path)?;
        let sf = safetensors::SafeTensors::deserialize(&file_data)?;

        for t in &tensors {
            if let Ok(view) = sf.tensor(&t.name) {
                let src_data: Vec<f32> = match view.dtype() {
                    safetensors::Dtype::F32 => {
                        bytemuck::cast_slice::<u8, f32>(view.data()).to_vec()
                    }
                    safetensors::Dtype::F16 => {
                        use half::f16;
                        let f16_data: &[f16] = bytemuck::cast_slice(view.data());
                        f16_data.iter().map(|v| v.to_f32()).collect()
                    }
                    safetensors::Dtype::BF16 => {
                        // BF16 → F32 conversion
                        let bf16_data = view.data();
                        bf16_data.chunks_exact(2)
                            .map(|chunk| {
                                let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                                bf16_to_f32(bf)
                            })
                            .collect()
                    }
                    _ => {
                        eprintln!("Warning: unsupported dtype for tensor {}", t.name);
                        continue;
                    }
                };

                buffer.clear();

                let edt = effective_dtype(target_dtype, src_data.len());

                match edt {
                    DType::FP32 => {
                        let bytes: &[u8] = bytemuck::cast_slice(&src_data);
                        buffer.extend_from_slice(bytes);
                    }
                    DType::FP16 => {
                        use half::f16;
                        let f16_data: Vec<f16> = src_data.iter().map(|&v| f16::from_f32(v)).collect();
                        let bytes: &[u8] = bytemuck::cast_slice(&f16_data);
                        buffer.extend_from_slice(bytes);
                    }
                    DType::Q4_0 => {
                        buffer = crate::quant::quantize_f32_to_q4_0(&src_data);
                    }
                    DType::Q8_0 => {
                        buffer = crate::quant::quantize_f32_to_q8_0(&src_data);
                    }
                }

                // Seek to correct offset and write
                let needed_pad = t.offset as usize - bytes_written;
                if needed_pad > 0 {
                    writer.write_all(&vec![0u8; needed_pad])?;
                    bytes_written += needed_pad;
                }

                writer.write_all(&buffer)?;
                bytes_written += buffer.len();

                // Pad to next aligned boundary
                let next_aligned = ((bytes_written as u64 + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
                let pad = (next_aligned - bytes_written as u64) as usize;
                if pad > 0 && pad < ALIGNMENT as usize {
                    writer.write_all(&vec![0u8; pad])?;
                    bytes_written += pad;
                }
            }
        }
    }

    writer.flush()?;
    println!("Wrote {} bytes to {}", bytes_written, out_path.display());

    Ok(())
}

/// Validate a `.hypno` file: read back the header, verify magic bytes, check offsets.
pub fn validate(path: &Path) -> anyhow::Result<()> {
    let model = crate::loader::HypnoModel::open(path)
        .map_err(|e| anyhow::anyhow!("Validation failed: {}", e))?;

    let header = model.manifest.header;
    println!("  Magic bytes: {:?}", &header.magic);
    println!("  Version: {}", header.version);
    println!("  Metadata KVs: {}", model.manifest.metadata.len());
    println!("  Tensors: {}", model.manifest.tensors.len());

    for kv in &model.manifest.metadata {
        println!("    {} = {}", kv.key, kv.value);
    }

    // Check that all tensor offsets are within bounds
    let file_size = fs::metadata(path)?.len();
    for t in &model.manifest.tensors {
        if t.offset + t.data_len > file_size {
            anyhow::bail!(
                "Tensor '{}' offset {} + len {} exceeds file size {}",
                t.name, t.offset, t.data_len, file_size
            );
        }
    }

    // Check that all tensor data pointers are readable
    for t in &model.manifest.tensors {
        if model.get_tensor_data(&t.name).is_none() {
            anyhow::bail!("Tensor '{}' data not accessible", t.name);
        }
    }

    println!("  All {} tensor offsets valid", model.manifest.tensors.len());

    Ok(())
}

/// Convert BF16 (brain floating point) to f32 (also used by lora module).
pub fn bf16_to_f32(bf: u16) -> f32 {
    // BF16 has the same exponent range as f32 but only 7 mantissa bits
    // To convert: shift left by 16 bits (fill mantissa with zeros)
    f32::from_bits((bf as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal safetensors file with fake weights for testing.
    fn create_test_safetensor(path: &Path) {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype as SfDtype;

        let data_f32: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let data_bytes: &[u8] = bytemuck::cast_slice(&data_f32);

        let tensor_view = TensorView::new(
            SfDtype::F32,
            vec![64],
            data_bytes,
        ).unwrap();

        let mut tensors: std::collections::HashMap<String, TensorView> = std::collections::HashMap::new();
        tensors.insert("weight".to_string(), tensor_view);

        let sf_data = safetensors::serialize(&tensors, &None).unwrap();
        fs::write(path, sf_data).unwrap();
    }

    #[test]
    fn test_convert_and_validate() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let model_dir = tmp_dir.path();

        // Create config.json
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 172,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        fs::write(model_dir.join("config.json"), config.to_string()).unwrap();

        // Create safetensors file
        create_test_safetensor(&model_dir.join("model.safetensors"));

        // Convert
        let out_path = tmp_dir.path().join("test.hypno");
        convert(model_dir, &out_path, DType::FP32, false).unwrap();

        // Validate
        validate(&out_path).unwrap();

        // Check with loader
        let model = crate::loader::HypnoModel::open(&out_path).unwrap();
        assert_eq!(model.get_metadata("architecture"), Some("LlamaForCausalLM"));
        assert_eq!(model.get_metadata("hidden_size"), Some("64"));

        let (data, dtype) = model.get_tensor_data("weight").unwrap();
        assert_eq!(dtype, DType::FP32);
        let floats: &[f32] = bytemuck::cast_slice(data);
        assert!((floats[0] - 0.0).abs() < 0.001);
    }

    fn create_large_test_safetensor(path: &Path) {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype as SfDtype;

        // Create a large enough tensor to actually get quantized (>4096 elements)
        let n = 128 * 64; // 8192 elements
        let data_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let data_bytes: &[u8] = bytemuck::cast_slice(&data_f32);

        let tensor_view = TensorView::new(
            SfDtype::F32,
            vec![128, 64],
            data_bytes,
        ).unwrap();

        let mut tensors: std::collections::HashMap<String, TensorView> = std::collections::HashMap::new();
        tensors.insert("weight".to_string(), tensor_view);

        let sf_data = safetensors::serialize(&tensors, &None).unwrap();
        fs::write(path, sf_data).unwrap();
    }

    #[test]
    fn test_convert_q4_0() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let model_dir = tmp_dir.path();

        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 172,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5
        });
        fs::write(model_dir.join("config.json"), config.to_string()).unwrap();
        create_large_test_safetensor(&model_dir.join("model.safetensors"));

        let out_path = tmp_dir.path().join("test_q4.hypno");
        convert(model_dir, &out_path, DType::Q4_0, false).unwrap();
        validate(&out_path).unwrap();

        // Large tensor gets quantized, small ones stay FP32
        let model = crate::loader::HypnoModel::open(&out_path).unwrap();
        let (data, dtype) = model.get_tensor_data("weight").unwrap();
        assert_eq!(dtype, DType::Q4_0);
        // 128*64 = 8192 elements → 256 blocks * 18 = 4608 bytes
        assert_eq!(data.len(), 4608);
    }
}
