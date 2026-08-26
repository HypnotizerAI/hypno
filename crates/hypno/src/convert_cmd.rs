//! `hypno convert` — Convert safetensors, GGUF, and LoRA to .hypno format.

use clap::Parser;
use crate::dtype::DType;
use std::path::PathBuf;

#[derive(Parser)]
pub struct Args {
    /// Input directory containing safetensors files + config.json, or a .gguf file
    #[arg(short = 'm', long, default_value = ".")]
    pub model_dir: PathBuf,

    /// Output .hypno file path
    #[arg(short = 'o', long, default_value = "model.hypno")]
    pub out: PathBuf,

    /// Quantization format: FP32, FP16, Q8_0, Q4_0
    #[arg(short = 'q', long, default_value = "FP16")]
    pub quantize: String,

    /// Input is a GGUF file (llama.cpp format)
    #[arg(long)]
    pub gguf: bool,

    /// Validate the output file after conversion
    #[arg(short = 'v', long, default_value = "true")]
    pub validate: bool,

    /// LoRA adapter directory (adapter_config.json + adapter_model.safetensors)
    #[arg(long)]
    pub lora_dir: Option<PathBuf>,

    /// Convert only the LoRA adapter (standalone mode)
    #[arg(long)]
    pub lora_only: bool,

    /// Override LoRA scaling factor
    #[arg(long)]
    pub lora_scale: Option<f32>,

    /// Store weight matrices in column-major order (transposed).
    /// Gives 2-3× faster matmul by enabling sequential memory access
    /// and perfect Q4_0 block alignment.
    #[arg(long)]
    pub col_major: bool,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let target = parse_dtype(&args.quantize)?;

    // LoRA-only: load adapter, convert standalone
    if args.lora_only {
        let lora_dir = args.lora_dir.clone().unwrap_or_else(|| args.model_dir.clone());
        let adapter = crate::lora::load_lora_adapter(&lora_dir)?;
        return crate::lora::convert_lora_standalone(&adapter, &args.out, target);
    }

    // LoRA merged into base model
    if let Some(ref lora_dir) = args.lora_dir {
        let adapter = crate::lora::load_lora_adapter(lora_dir)?;
        let scale = args.lora_scale.unwrap_or_else(|| adapter.lora_alpha as f32 / adapter.r as f32);

        // Copy base model to temp dir, merge LoRA into safetensor files
        let tmp = tempfile::tempdir()?;
        copy_model_files(&args.model_dir, tmp.path())?;
        merge_lora_into_safetensors(tmp.path(), &adapter, scale)?;

        let result = crate::sft_convert::convert(tmp.path(), &args.out, target, args.col_major);
        if result.is_ok() && args.validate {
            crate::sft_convert::validate(&args.out)?;
        }
        return result;
    }

    // GGUF conversion
    if args.gguf {
        let result = crate::gguf::convert_gguf_to_hypno(&args.model_dir, &args.out, target);
        if result.is_ok() && args.validate {
            crate::sft_convert::validate(&args.out)?;
        }
        return result;
    }

    // Standard safetensors conversion
    let result = crate::sft_convert::convert(&args.model_dir, &args.out, target, args.col_major);
    if result.is_ok() && args.validate {
        crate::sft_convert::validate(&args.out)?;
    }
    result
}

fn parse_dtype(s: &str) -> anyhow::Result<DType> {
    match s.to_uppercase().as_str() {
        "FP32" => Ok(DType::FP32),
        "FP16" => Ok(DType::FP16),
        "Q8_0" => Ok(DType::Q8_0),
        "Q4_0" => Ok(DType::Q4_0),
        other => anyhow::bail!("Unknown quantize format: {}. Use FP32, FP16, Q8_0, or Q4_0", other),
    }
}

fn copy_model_files(src: &std::path::Path, dst: &std::path::Path) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str.ends_with(".safetensors")
            || fname_str == "config.json"
            || fname_str.starts_with("tokenizer")
            || fname_str == "special_tokens_map.json"
        {
            std::fs::copy(entry.path(), dst.join(&*fname_str))?;
        }
    }
    Ok(())
}

fn merge_lora_into_safetensors(
    model_dir: &std::path::Path,
    adapter: &crate::lora::LoraAdapter,
    scale: f32,
) -> anyhow::Result<()> {
    use crate::lora::merge_lora_weights;
    use std::collections::BTreeMap;

    // Find all safetensor files
    let mut sf_paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            sf_paths.push(entry.path());
        }
    }
    sf_paths.sort();

    // Build a map of target module base names from lora adapter
    let lora_module_set: std::collections::HashSet<String> =
        adapter.target_modules.iter().cloned().collect();

    for sf_path in &sf_paths {
        let data = std::fs::read(sf_path)?;
        let sf = safetensors::SafeTensors::deserialize(&data)?;
        let mut merged: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        for name in sf.names() {
            let view = sf.tensor(&name)?;
            let shape: Vec<usize> = view.shape().to_vec();
            let base_data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(view.data()).to_vec(),
                safetensors::Dtype::F16 => {
                    use half::f16;
                    let f16d: &[f16] = bytemuck::cast_slice(view.data());
                    f16d.iter().map(|v| v.to_f32()).collect()
                }
                safetensors::Dtype::BF16 => {
                    view.data().chunks_exact(2)
                        .map(|c| crate::sft_convert::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect()
                }
                _ => continue,
            };

            // Check if any LoRA module targets this weight
            let module_name = extract_module_name(&name);
            if let Some(abbrev) = &module_name {
                if lora_module_set.contains(abbrev) {
                    // Fetch LoRA A and B
                    let key_a = format!("{}.lora_A.weight", abbrev);
                    let key_b = format!("{}.lora_B.weight", abbrev);
                    if let (Some((ashape, a_data)), Some((bshape, b_data))) =
                        (adapter.lora_a.get(&key_a), adapter.lora_b.get(&key_b))
                    {
                        let merged_weight = merge_lora_weights(
                            a_data, ashape, b_data, bshape,
                            &base_data, &shape, scale,
                        );
                        let bytes: &[u8] = bytemuck::cast_slice(&merged_weight);
                        merged.insert(name.to_string(), bytes.to_vec());
                        continue;
                    }
                }
            }
            // Pass through unchanged
            merged.insert(name.to_string(), view.data().to_vec());
        }

        // Write merged safetensor
        let mut tensors: BTreeMap<String, safetensors::tensor::TensorView> = BTreeMap::new();
        let mut raw_data: Vec<Vec<u8>> = Vec::new();
        for (_n, d) in &merged {
            raw_data.push(d.clone());
        }
        for (i, (n, _)) in merged.iter().enumerate() {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![raw_data[i].len() / 4],
                &raw_data[i],
            )?;
            tensors.insert(n.clone(), view);
        }

        let out_data = safetensors::serialize(&tensors, &None)?;
        std::fs::write(sf_path, &out_data)?;
    }

    Ok(())
}

fn extract_module_name(name: &str) -> Option<String> {
    // Strip "base_model.model." prefix and ".weight" suffix
    let stripped = name
        .strip_prefix("base_model.model.")
        .unwrap_or(name);
    let stripped = stripped.strip_suffix(".weight").unwrap_or(stripped);
    // Extract module base: "model.layers.0.self_attn.q_proj" → "q_proj"
    let parts: Vec<&str> = stripped.rsplitn(2, '.').collect();
    if parts.len() == 2 {
        Some(parts[0].to_string())
    } else {
        Some(stripped.to_string())
    }
}
