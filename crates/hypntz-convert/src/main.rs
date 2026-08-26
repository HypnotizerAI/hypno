//! `hypntz-convert` — Convert HuggingFace safetensors models to `.hypno` format.
//!
//! Usage:
//! ```bash
//! hypntz-convert --model-dir ./llama-2-7b --out model.hypno
//! hypntz-convert --model-dir ./llama-2-7b --out model-q4.hypno --quantize Q4_0
//! ```

mod converter;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hypntz-convert")]
#[command(about = "Convert HuggingFace safetensors models to .hypno format")]
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
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let dtype = match args.quantize.to_uppercase().as_str() {
        "FP32" => hypntz_core::DType::FP32,
        "FP16" => hypntz_core::DType::FP16,
        "Q4_0" => hypntz_core::DType::Q4_0,
        "Q8_0" => hypntz_core::DType::Q8_0,
        other => anyhow::bail!("Unknown quantize type: {}. Use FP32, FP16, Q4_0, or Q8_0", other),
    };

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
