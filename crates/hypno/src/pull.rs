//! `hypno pull` — Download models from HuggingFace Hub and convert to .hypno.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Parser)]
pub struct Args {
    /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B")
    pub model_id: String,

    /// Output .hypno file path (default: derived from model name)
    #[arg(short, long)]
    pub out: Option<String>,

    /// Quantization format: FP32, FP16, Q8_0, Q4_0
    #[arg(short, long, default_value = "Q4_0")]
    pub quantize: String,

    /// HuggingFace API token (or set HF_TOKEN env var)
    #[arg(long)]
    pub token: Option<String>,

    /// Branch/revision
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Download cache directory
    #[arg(long)]
    pub cache_dir: Option<String>,

    /// Keep downloaded files after conversion
    #[arg(long)]
    pub keep: bool,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    path: String,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let token = args.token
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());

    let cache_dir = args.cache_dir.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        format!("{}/.cache/hypno/models", home)
    });

    let client = Client::builder().user_agent("hypno/0.1.0").build()?;

    println!("🔍 Listing files for {}...", args.model_id);
    let siblings = list_files(&client, &args.model_id, &args.revision, token.as_deref()).await?;

    let to_download: Vec<&HfSibling> = siblings.iter().filter(|f| {
        let p = &f.path;
        p == "config.json" || p == "tokenizer.json" || p == "tokenizer_config.json"
            || p == "special_tokens_map.json" || p == "generation_config.json"
            || (p.ends_with(".safetensors") && !p.contains("adapter"))
            || p.ends_with(".model")
    }).collect();

    if to_download.is_empty() {
        anyhow::bail!("No safetensors files found for {}", args.model_id);
    }

    let model_dir = PathBuf::from(&cache_dir).join(sanitize_name(&args.model_id));
    println!("📥 Downloading {} files to {}...", to_download.len(), model_dir.display());
    tokio::fs::create_dir_all(&model_dir).await?;

    for file in &to_download {
        let url = format!("https://huggingface.co/{}/resolve/{}/{}", args.model_id, args.revision, file.path);
        let dest = model_dir.join(&file.path);
        if let Some(parent) = dest.parent() { tokio::fs::create_dir_all(parent).await?; }
        download_file(&client, &url, &dest, token.as_deref()).await?;
    }
    println!("✅ Download complete.");

    let out_path = args.out.unwrap_or_else(|| format!("{}.hypno", sanitize_name(&args.model_id)));
    println!("🔄 Converting to {} ({})...", out_path, args.quantize);

    let target = match args.quantize.to_uppercase().as_str() {
        "FP32" => crate::dtype::DType::FP32, "FP16" => crate::dtype::DType::FP16,
        "Q8_0" => crate::dtype::DType::Q8_0, "Q4_0" => crate::dtype::DType::Q4_0,
        other => anyhow::bail!("Unknown quantize format: {}", other),
    };

    crate::sft_convert::convert(&model_dir, Path::new(&out_path), target, false)?;
    crate::sft_convert::validate(Path::new(&out_path))?;

    if !args.keep {
        println!("🧹 Cleaning up...");
        let _ = tokio::fs::remove_dir_all(&model_dir).await;
    }

    println!("✨ Done! Model saved to {}", out_path);
    println!("   Run: hypno run --model {}", out_path);
    println!("   Or:  hypno serve --model {} --port 8080", out_path);
    Ok(())
}

async fn list_files(client: &Client, model_id: &str, revision: &str, token: Option<&str>) -> anyhow::Result<Vec<HfSibling>> {
    let url = format!("https://huggingface.co/api/models/{}/tree/{}", model_id, revision);
    let mut req = client.get(&url);
    if let Some(t) = token { req = req.bearer_auth(t); }
    let siblings: Vec<HfSibling> = req.send().await?.json().await?;
    Ok(siblings)
}

async fn download_file(client: &Client, url: &str, dest: &Path, token: Option<&str>) -> anyhow::Result<()> {
    let mut req = client.get(url);
    if let Some(t) = token { req = req.bearer_auth(t); }
    let response = req.send().await?;
    if !response.status().is_success() {
        anyhow::bail!("HTTP {} downloading {}", response.status(), url);
    }
    let total_size = response.content_length().unwrap_or(0);
    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})").unwrap()
            .progress_chars("━▶ "));
        pb.set_message(dest.file_name().unwrap_or_default().to_string_lossy().to_string());
        pb
    } else {
        ProgressBar::new_spinner()
    };

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }
    pb.finish_with_message(format!("✅ {}", dest.file_name().unwrap_or_default().to_string_lossy()));
    file.flush().await?;
    Ok(())
}

fn sanitize_name(model_id: &str) -> String {
    model_id.split('/').last().unwrap_or(model_id).replace([' ', '.'], "-").to_lowercase()
}
