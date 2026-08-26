//! LoRA adapter conversion: convert standalone adapters or merge into base weights.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// A parsed LoRA adapter from HuggingFace PEFT format.
pub struct LoraAdapter {
    /// Rank of the LoRA decomposition.
    pub r: usize,
    /// Alpha scaling factor (scale = alpha / r).
    pub lora_alpha: f64,
    /// Which modules this adapter targets (e.g. ["q_proj", "v_proj"]).
    pub target_modules: Vec<String>,
    /// LoRA A matrices: keyed by module name, shape [r, in_features].
    pub lora_a: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// LoRA B matrices: keyed by module name, shape [out_features, r].
    pub lora_b: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// Base model this adapter was trained on (optional).
    pub base_model_name: Option<String>,
}

/// Load a LoRA adapter from a PEFT directory.
pub fn load_lora_adapter(lora_dir: &Path) -> anyhow::Result<LoraAdapter> {
    // ── 1. Load adapter_config.json ─────────────────────────────
    let config_path = lora_dir.join("adapter_config.json");
    let config: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read adapter_config.json: {}", e))?,
    )?;

    let r = config["r"].as_u64().unwrap_or(8) as usize;
    let lora_alpha = config["lora_alpha"].as_f64().unwrap_or(r as f64);
    let target_modules: Vec<String> = config["target_modules"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let base_model_name = config["base_model_name_or_path"]
        .as_str()
        .map(String::from);

    // ── 2. Load adapter_model.safetensors ───────────────────────
    let sf_path = lora_dir
        .join("adapter_model.safetensors")
        .exists()
        .then(|| lora_dir.join("adapter_model.safetensors"))
        .or_else(|| {
            lora_dir.join("adapter_model.bin").exists()
                .then(|| lora_dir.join("adapter_model.bin"))
        });

    let sf_path = sf_path.ok_or_else(|| {
        anyhow::anyhow!("No adapter_model.safetensors or adapter_model.bin found in {}", lora_dir.display())
    })?;

    // Try safetensors first, fall back to PyTorch .bin
    let tensor_data = if sf_path.extension().map_or(false, |e| e == "safetensors") {
        load_safetensor_adapter(&sf_path)?
    } else {
        return Err(anyhow::anyhow!("PyTorch .bin adapter files not yet supported. Use safetensors."));
    };

    // ── 3. Organize into lora_A / lora_B ────────────────────────
    let mut lora_a: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    let mut lora_b: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();

    for (name, (shape, data)) in &tensor_data {
        if name.contains("lora_A") {
            let module = extract_module_name(name);
            lora_a.insert(module, (shape.clone(), data.clone()));
        } else if name.contains("lora_B") {
            let module = extract_module_name(name);
            lora_b.insert(module, (shape.clone(), data.clone()));
        }
    }

    if lora_a.is_empty() {
        anyhow::bail!("No lora_A tensors found in adapter — is this a LoRA adapter?");
    }

    println!("  LoRA rank r={}, alpha={:.1}, scale={:.4}", r, lora_alpha, lora_alpha / r as f64);
    println!("  Target modules: {:?}", target_modules);
    println!("  Found {} LoRA layer pairs", lora_a.len());

    Ok(LoraAdapter {
        r,
        lora_alpha,
        target_modules,
        lora_a,
        lora_b,
        base_model_name,
    })
}

fn load_safetensor_adapter(path: &Path) -> anyhow::Result<BTreeMap<String, (Vec<usize>, Vec<f32>)>> {
    let file_data = fs::read(path)?;
    let sf = safetensors::SafeTensors::deserialize(&file_data)?;
    let mut result = BTreeMap::new();

    for name in sf.names() {
        let view = sf.tensor(&name)?;
        let shape = view.shape().to_vec();
        let data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
            safetensors::Dtype::F16 => {
                use half::f16;
                let f16_data: &[f16] = bytemuck::cast_slice(view.data());
                f16_data.iter().map(|v| v.to_f32()).collect()
            }
            safetensors::Dtype::BF16 => {
                view.data()
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                        crate::sft_convert::bf16_to_f32(bf)
                    })
                    .collect()
            }
            _ => {
                eprintln!("Warning: unsupported dtype in LoRA tensor '{}'", name);
                continue;
            }
        };
        result.insert(name.to_string(), (shape, data));
    }

    Ok(result)
}

/// Extract module name from a LoRA tensor name like "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
fn extract_module_name(full_name: &str) -> String {
    // Strip "base_model.model." prefix if present
    let name = full_name
        .strip_prefix("base_model.model.")
        .unwrap_or(full_name);
    // Strip ".lora_A.weight" or ".lora_B.weight" suffix
    let name = name
        .strip_suffix(".lora_A.weight")
        .or_else(|| name.strip_suffix(".lora_B.weight"))
        .unwrap_or(name);
    name.to_string()
}

/// Merge LoRA weights into a base weight tensor.
///
/// `lora_a`: shape [r, in_features]
/// `lora_b`: shape [out_features, r]
/// `base_weight`: shape [out_features, in_features] (row-major)
/// `scale`: lora_alpha / r
///
/// Returns: `base_weight + scale * (lora_b @ lora_a)`
pub fn merge_lora_weights(
    lora_a: &[f32],
    lora_a_shape: &[usize],
    lora_b: &[f32],
    lora_b_shape: &[usize],
    base_weight: &[f32],
    base_shape: &[usize],
    scale: f32,
) -> Vec<f32> {
    let r = lora_a_shape[0];
    let in_features = lora_a_shape[1];
    let out_features = lora_b_shape[0];

    assert_eq!(lora_b_shape[1], r, "LoRA B second dim must match rank");
    assert_eq!(base_shape[0], out_features, "Base weight rows must match LoRA B rows");
    assert_eq!(base_shape[1], in_features, "Base weight cols must match LoRA A cols");

    let mut merged = base_weight.to_vec();

    // delta = lora_b @ lora_a  (out_features × r) @ (r × in_features) → out_features × in_features
    for o in 0..out_features {
        for i in 0..in_features {
            let mut dot = 0.0f32;
            for k in 0..r {
                dot += lora_b[o * r + k] * lora_a[k * in_features + i];
            }
            merged[o * in_features + i] += scale * dot;
        }
    }

    merged
}

/// Convert a standalone LoRA adapter to .hypno format (without merging).
/// Tensors are stored with their original LoRA names.
pub fn convert_lora_standalone(
    lora: &LoraAdapter,
    out_path: &Path,
    target_dtype: crate::dtype::DType,
) -> anyhow::Result<()> {
    use crate::dtype::DType;
use crate::format::{HypnoHeader, MetaKV, TensorMeta, ALIGNMENT};
    use std::io::{BufWriter, Write};

    let mut metadata_kvs: Vec<MetaKV> = Vec::new();
    metadata_kvs.push(MetaKV { key: "lora_rank".into(), value: lora.r.to_string() });
    metadata_kvs.push(MetaKV { key: "lora_alpha".into(), value: lora.lora_alpha.to_string() });
    metadata_kvs.push(MetaKV {
        key: "target_modules".into(),
        value: lora.target_modules.join(","),
    });
    if let Some(ref bm) = lora.base_model_name {
        metadata_kvs.push(MetaKV { key: "base_model".into(), value: bm.clone() });
    }

    let header_size = 16u64;
    let metadata_size: u64 = metadata_kvs.iter().map(|kv| kv.serialized_size() as u64).sum();

    // Collect all lora tensors
    let mut tensor_info: Vec<(String, Vec<usize>, &[f32])> = Vec::new();
    for (name, (shape, data)) in &lora.lora_a {
        tensor_info.push((format!("lora_A.{}", name), shape.clone(), data));
    }
    for (name, (shape, data)) in &lora.lora_b {
        tensor_info.push((format!("lora_B.{}", name), shape.clone(), data));
    }

    let tensor_table_size: u64 = tensor_info.iter()
        .map(|(name, shape, _data)| {
            let ndim = shape.len() as u32;
            4 + name.len() as u64 + 4 + (ndim as u64) * 8 + 4 + 8 + 8
        })
        .sum();

    let metadata_end = header_size + metadata_size + tensor_table_size;
    let data_start = ((metadata_end + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

    let mut tensors: Vec<TensorMeta> = Vec::new();
    let mut current_offset = data_start;

    for (name, shape, _data) in &tensor_info {
        let ndim = shape.len() as u32;
        let n_elems: usize = shape.iter().product();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let edt = crate::sft_convert::effective_dtype(target_dtype, n_elems);
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
        current_offset = ((current_offset + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }

    // Write
    let out_file = fs::File::create(out_path)?;
    let mut writer = BufWriter::new(out_file);
    let header = HypnoHeader::new(metadata_kvs.len() as u32, tensors.len() as u32);
    writer.write_all(bytemuck::bytes_of(&header))?;

    for kv in &metadata_kvs {
        writer.write_all(&(kv.key.len() as u32).to_le_bytes())?;
        writer.write_all(kv.key.as_bytes())?;
        writer.write_all(&(kv.value.len() as u32).to_le_bytes())?;
        writer.write_all(kv.value.as_bytes())?;
    }

    for t in &tensors {
        writer.write_all(&(t.name.len() as u32).to_le_bytes())?;
        writer.write_all(t.name.as_bytes())?;
        writer.write_all(&t.ndim.to_le_bytes())?;
        for &dim in &t.shape { writer.write_all(&dim.to_le_bytes())?; }
        writer.write_all(&(t.dtype as u32).to_le_bytes())?;
        writer.write_all(&t.offset.to_le_bytes())?;
        writer.write_all(&t.data_len.to_le_bytes())?;
    }

    let pos = header_size + metadata_size + tensor_table_size;
    let aligned = ((pos + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    for _ in pos..aligned { writer.write_all(&[0u8])?; }

    let mut bytes_written = aligned as usize;
    let mut buffer = Vec::new();

    for (i, (_name, _shape, data)) in tensor_info.iter().enumerate() {
        let t = &tensors[i];
        let needed_pad = t.offset as usize - bytes_written;
        if needed_pad > 0 {
            writer.write_all(&vec![0u8; needed_pad])?;
            bytes_written += needed_pad;
        }

        buffer.clear();
        match t.dtype {
            DType::FP32 => {
                buffer.extend_from_slice(bytemuck::cast_slice(data));
            }
            DType::FP16 => {
                use half::f16;
                let f16_data: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                buffer.extend_from_slice(bytemuck::cast_slice(&f16_data));
            }
            DType::Q4_0 => {
                buffer = crate::quant::quantize_f32_to_q4_0(data);
            }
            DType::Q8_0 => {
                buffer = crate::quant::quantize_f32_to_q8_0(data);
            }
        }

        writer.write_all(&buffer)?;
        bytes_written += buffer.len();

        let next_aligned = ((bytes_written as u64 + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        let pad = (next_aligned - bytes_written as u64) as usize;
        if pad > 0 && pad < ALIGNMENT as usize {
            writer.write_all(&vec![0u8; pad])?;
            bytes_written += pad;
        }
    }

    writer.flush()?;
    println!("Wrote standalone LoRA adapter: {} bytes → {}", bytes_written, out_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_module_name() {
        assert_eq!(
            extract_module_name("base_model.model.layers.0.self_attn.q_proj.lora_A.weight"),
            "layers.0.self_attn.q_proj"
        );
        assert_eq!(
            extract_module_name("model.layers.5.mlp.down_proj.lora_B.weight"),
            "model.layers.5.mlp.down_proj"
        );
    }

    #[test]
    fn test_merge_lora_weights_small() {
        // r=2, in=3, out=2
        let lora_a = vec![
            1.0, 0.0, 0.0,  // row 0 of A
            0.0, 1.0, 0.0,  // row 1 of A
        ];
        let lora_b = vec![
            1.0, 0.0,  // row 0 of B
            0.0, 1.0,  // row 1 of B
        ];
        let base = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ];

        // lora_b @ lora_a = identity for this case → merged = base + scale * I
        let merged = merge_lora_weights(
            &lora_a, &[2, 3],
            &lora_b, &[2, 2],
            &base, &[2, 3],
            1.0,
        );

        assert!((merged[0] - 2.0).abs() < 0.001);  // 1.0 + 1.0
        assert!((merged[3] - 0.0).abs() < 0.001);  // unchanged
        assert!((merged[4] - 2.0).abs() < 0.001);  // 1.0 + 1.0
    }
}
