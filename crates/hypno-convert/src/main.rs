//! `hypno-convert` — Convert models and LoRA adapters to `.hypno` format.
//!
//! ## Full model conversion
//! ```bash
//! hypno-convert --model-dir ./llama-2-7b --out model.hypno
//! hypno-convert --model-dir ./llama-2-7b --out model-q4.hypno --quantize Q4_0
//! ```
//!
//! ## LoRA adapter conversion
//! ```bash
//! # Standalone adapter (just the LoRA weights)
//! hypno-convert --lora-only ./my-lora --out adapter.hypno
//!
//! # Merge LoRA into base model then convert
//! hypno-convert --model-dir ./llama-2-7b --lora-dir ./my-lora --out merged.hypno
//! ```

mod converter;
mod gguf;
mod lora;

use clap::Parser;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "hypno-convert")]
#[command(about = "Convert HuggingFace safetensors + LoRA adapters to .hypno format")]
struct Args {
    /// Input directory containing safetensors model files and config.json
    #[arg(short = 'm', long, default_value = ".")]
    model_dir: PathBuf,

    /// Output .hypno file path
    #[arg(short = 'o', long, default_value = "model.hypno")]
    out: PathBuf,

    /// Quantize weights: FP32, FP16, Q4_0, Q8_0
    #[arg(short = 'q', long, default_value = "FP16")]
    quantize: String,

    /// Validate the output file after conversion
    #[arg(short = 'v', long, default_value = "true")]
    validate: bool,

    /// LoRA adapter directory (adapter_config.json + adapter_model.safetensors).
    /// When combined with --model-dir, merges LoRA weights into the base model
    /// before converting.
    #[arg(long)]
    lora_dir: Option<PathBuf>,

    /// Convert only the LoRA adapter (standalone, no base model needed).
    #[arg(long)]
    lora_only: Option<PathBuf>,

    /// LoRA merge scale override (default: lora_alpha / r from adapter_config.json).
    #[arg(long)]
    lora_scale: Option<f32>,

    /// Convert from GGUF format instead of safetensors.
    #[arg(long)]
    gguf: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let dtype = match args.quantize.to_uppercase().as_str() {
        "FP32" => hypno_core::DType::FP32,
        "FP16" => hypno_core::DType::FP16,
        "Q4_0" => hypno_core::DType::Q4_0,
        "Q8_0" => hypno_core::DType::Q8_0,
        other => anyhow::bail!("Unknown quantize type: {}. Use FP32, FP16, Q4_0, or Q8_0", other),
    };

    // ── GGUF mode: convert from GGUF format ──────────────────
    if args.gguf {
        println!("Converting GGUF model from: {}", args.model_dir.display());
        println!("Output: {}", args.out.display());
        println!("Dtype: {:?}", dtype);
        gguf::convert_gguf_to_hypno(&args.model_dir, &args.out, dtype)?;

        if args.validate {
            println!("\nValidating...");
            converter::validate(&args.out)?;
            println!("Validation passed!");
        }
        println!("\nDone!");
        return Ok(());
    }

    // ── LoRA-only mode: convert adapter standalone ───────────
    if let Some(ref lora_only) = args.lora_only {
        println!("Converting LoRA adapter from: {}", lora_only.display());
        let adapter = lora::load_lora_adapter(lora_only)?;
        lora::convert_lora_standalone(&adapter, &args.out, dtype)?;

        if args.validate {
            println!("\nValidating...");
            converter::validate(&args.out)?;
            println!("Validation passed!");
        }
        println!("\nDone!");
        return Ok(());
    }

    // ── LoRA merge mode: merge adapter into base, then convert ─
    if let Some(ref lora_dir) = args.lora_dir {
        println!("Loading LoRA adapter from: {}", lora_dir.display());
        let adapter = lora::load_lora_adapter(lora_dir)?;
        let scale = args.lora_scale
            .unwrap_or_else(|| (adapter.lora_alpha / adapter.r as f64) as f32);

        println!("Loading base model from: {}", args.model_dir.display());
        println!("Merge scale: {:.4}", scale);

        let merged_dir = merge_lora_into_model(&adapter, &args.model_dir, scale)?;

        println!("Converting merged model...");
        converter::convert(&merged_dir, &args.out, dtype)?;

        // Clean up temp dir
        let _ = std::fs::remove_dir_all(&merged_dir);

        if args.validate {
            println!("\nValidating...");
            converter::validate(&args.out)?;
            println!("Validation passed!");
        }
        println!("\nDone!");
        return Ok(());
    }

    // ── Normal conversion ─────────────────────────────────────
    println!("Converting model from: {}", args.model_dir.display());
    println!("Output: {}", args.out.display());
    println!("Dtype: {:?}", dtype);

    converter::convert(&args.model_dir, &args.out, dtype)?;

    if args.validate {
        println!("\nValidating output file...");
        converter::validate(&args.out)?;
        println!("Validation passed!");
    }

    println!("\nDone!");
    Ok(())
}

/// Merge a LoRA adapter into base model weights in a temporary directory.
fn merge_lora_into_model(
    adapter: &lora::LoraAdapter,
    model_dir: &Path,
    scale: f32,
) -> anyhow::Result<PathBuf> {
    use std::collections::HashMap;

    let tmp_dir = tempfile::tempdir()?;
    let merged_dir = tmp_dir.path().to_path_buf();

    // Copy config.json (and tokenizer.json if present)
    std::fs::copy(model_dir.join("config.json"), merged_dir.join("config.json"))?;
    if model_dir.join("tokenizer.json").exists() {
        std::fs::copy(model_dir.join("tokenizer.json"), merged_dir.join("tokenizer.json"))?;
    }
    if model_dir.join("tokenizer_config.json").exists() {
        std::fs::copy(model_dir.join("tokenizer_config.json"), merged_dir.join("tokenizer_config.json"))?;
    }

    // Build a map: module_name → (lora_a_shape, lora_a_data, lora_b_shape, lora_b_data)
    let mut lora_map: HashMap<String, (&[usize], &[f32], &[usize], &[f32])> = HashMap::new();
    for (mod_name, (shape_a, data_a)) in &adapter.lora_a {
        if let Some((shape_b, data_b)) = adapter.lora_b.get(mod_name) {
            lora_map.insert(mod_name.clone(), (shape_a, data_a, shape_b, data_b));
        }
    }

    // Find all safetensors shards and merge LoRA into matching tensors
    let mut safetensor_files: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") { Some(entry.path()) } else { None }
        })
        .collect();
    safetensor_files.sort();

    let mut merged_index = 0;
    let mut merged_tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    for sf_path in &safetensor_files {
        let file_data = std::fs::read(sf_path)?;
        let sf = safetensors::SafeTensors::deserialize(&file_data)?;

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
                    view.data().chunks_exact(2)
                        .map(|chunk| converter::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                        .collect()
                }
                _ => continue,
            };

            // Normalize name: strip "model." prefix for matching
            let clean_name = name.strip_prefix("model.").unwrap_or(&name);

            // Try to match with a LoRA module
            let mut merged_data = data;
            for (mod_name, (sha, da, shb, db)) in &lora_map {
                // Match: the clean tensor name should match the LoRA module pattern
                // e.g. "layers.0.self_attn.q_proj.weight" matches "layers.0.self_attn.q_proj"
                if clean_name.ends_with(&format!("{}.weight", mod_name))
                    || clean_name == format!("{}.weight", mod_name)
                {
                    let expected_rows = sha[1]; // LoRA A's in_features = weight cols
                    let expected_cols = shb[0]; // LoRA B's out_features = weight rows

                    if shape.len() == 2 && shape[0] == expected_cols && shape[1] == expected_rows {
                        merged_data = lora::merge_lora_weights(
                            da, sha, db, shb,
                            &merged_data, &shape,
                            scale,
                        );
                        println!("  Merged LoRA into: {}", name);
                    }
                    break;
                }
            }

            merged_tensors.insert(name.to_string(), (shape, merged_data));
        }
    }

    // Write merged tensors to new safetensors file
    // Build owned data first, then create views that borrow from it
    let mut owned_data: Vec<(String, Vec<usize>, Vec<f32>)> = merged_tensors
        .into_iter()
        .map(|(n, (s, d))| (n, s, d))
        .collect();

    // We need to keep the float vecs alive while building views
    let float_bufs: Vec<Vec<f32>> = owned_data.iter().map(|(_, _, d)| d.clone()).collect();

    let mut sf_map: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
    for (i, (name, shape, _)) in owned_data.iter().enumerate() {
        let bytes: &[u8] = bytemuck::cast_slice(&float_bufs[i]);
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            shape.clone(),
            bytes,
        )?;
        sf_map.insert(name.clone(), tv);
    }

    let out_sf = safetensors::serialize(&sf_map, &None)?;
    let out_path = merged_dir.join(format!("model-{:05}-of-00001.safetensors", merged_index));
    std::fs::write(&out_path, out_sf)?;
    merged_index += 1;

    // Don't clean up — we return the path
    let persistent_dir = std::env::temp_dir().join(format!("hypno-lora-merged-{}", std::process::id()));
    std::fs::create_dir_all(&persistent_dir)?;
    for entry in std::fs::read_dir(&merged_dir)? {
        let entry = entry?;
        std::fs::copy(entry.path(), persistent_dir.join(entry.file_name()))?;
    }

    Ok(persistent_dir)
}

