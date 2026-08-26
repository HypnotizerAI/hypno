//! hypno-pull — Download models from HuggingFace Hub and convert to .hypno.
//!
//! ```bash
//! hypno-pull Qwen/Qwen2.5-3B                    # download + convert to Q4_0
//! hypno-pull mistralai/Mistral-7B-Instruct-v0.3  # with auto-detected quant
//! hypno-pull TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quantize Q8_0 --out mymodel.hypno
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Download models from HuggingFace Hub and convert to .hypno format.
#[derive(Parser, Debug)]
#[command(name = "hypno-pull", version, about)]
struct Args {
    /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B")
    model_id: String,

    /// Output .hypno file path (default: derived from model name)
    #[arg(short, long)]
    out: Option<String>,

    /// Quantization format: FP32, FP16, Q8_0, Q4_0 (default: Q4_0)
    #[arg(short, long, default_value = "Q4_0")]
    quantize: String,

    /// HuggingFace API token (or set HF_TOKEN env var)
    #[arg(long)]
    token: Option<String>,

    /// Revision/branch to download (default: main)
    #[arg(long, default_value = "main")]
    revision: String,

    /// Download directory (default: ~/.cache/hypno/models)
    #[arg(long)]
    cache_dir: Option<String>,

    /// Keep downloaded files after conversion
    #[arg(long)]
    keep: bool,
}

#[derive(Debug, Deserialize)]
struct HfFile {
    #[serde(rename = "rfilename")]
    path: String,
    #[allow(dead_code)]
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(rename = "rfilename")]
    path: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let token = args
        .token
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok());

    let cache_dir = args.cache_dir.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        format!("{}/.cache/hypno/models", home)
    });

    let client = Client::builder()
        .user_agent("hypno-pull/0.1.0")
        .build()?;

    // Step 1: List model files
    println!("🔍 Listing files for {}...", args.model_id);
    let files = list_model_files(&client, &args.model_id, &args.revision, token.as_deref())
        .await?;

    // Select files to download: config.json, tokenizer*, *.safetensors
    let to_download: Vec<&HfFile> = files
        .iter()
        .filter(|f| {
            let p = &f.path;
            p == "config.json"
                || p == "tokenizer.json"
                || p == "tokenizer_config.json"
                || p == "special_tokens_map.json"
                || p == "generation_config.json"
                || (p.ends_with(".safetensors") && !p.contains("adapter"))
                || p.ends_with(".model") // sentencepiece
        })
        .collect();

    if to_download.is_empty() {
        anyhow::bail!("No safetensors files found for {}", args.model_id);
    }

    let model_dir = PathBuf::from(&cache_dir).join(sanitize_name(&args.model_id));

    println!(
        "📥 Downloading {} files to {}...",
        to_download.len(),
        model_dir.display()
    );

    tokio::fs::create_dir_all(&model_dir).await?;

    for file in &to_download {
        let url = hf_download_url(&args.model_id, &args.revision, &file.path);
        let dest = model_dir.join(&file.path);
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        download_file(&client, &url, &dest, token.as_deref()).await?;
    }

    println!("✅ Download complete.");

    // Step 2: Convert to .hypno
    let out_path = args.out.unwrap_or_else(|| {
        let name = sanitize_name(&args.model_id);
        format!("{}.hypno", name)
    });

    println!("🔄 Converting to {} ({})...", out_path, args.quantize);
    convert_to_hypno(&model_dir, &out_path, &args.quantize)?;

    // Step 3: Cleanup
    if !args.keep {
        println!("🧹 Cleaning up downloaded files...");
        let _ = tokio::fs::remove_dir_all(&model_dir).await;
    }

    println!("✨ Done! Model saved to {}", out_path);
    println!("   Run: hypno-cli --model {}", out_path);
    println!("   Or:  hypno-server --model {} --port 8080", out_path);

    Ok(())
}

async fn list_model_files(
    client: &Client,
    model_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<Vec<HfFile>> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/{}",
        model_id, revision
    );

    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let siblings: Vec<HfSibling> = req
        .send()
        .await
        .context("Failed to query HuggingFace API")?
        .json()
        .await
        .context("Failed to parse API response")?;

    Ok(siblings
        .into_iter()
        .map(|s| HfFile {
            path: s.path,
            size: None,
        })
        .collect())
}

fn hf_download_url(model_id: &str, revision: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model_id, revision, filename
    )
}

async fn download_file(
    client: &Client,
    url: &str,
    dest: &Path,
    token: Option<&str>,
) -> Result<()> {
    let mut req = client.get(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }

    let response = req.send().await.context("Download failed")?;
    let total_size = response.content_length().unwrap_or(0);
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {} downloading {}", status, url);
    }

    let pb = if total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")
                .unwrap()
                .progress_chars("━▶ "),
        );
        pb.set_message(
            dest.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
        );
        pb
    } else {
        ProgressBar::new_spinner()
    };

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;

    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream error")?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message(format!(
        "✅ {}",
        dest.file_name().unwrap_or_default().to_string_lossy()
    ));
    file.flush().await?;

    Ok(())
}

fn convert_to_hypno(model_dir: &Path, out_path: &str, quantize: &str) -> Result<()> {
    let mut cmd = std::process::Command::new("hypno-convert");
    cmd.arg("--model-dir").arg(model_dir);
    cmd.arg("--out").arg(out_path);
    cmd.arg("--quantize").arg(quantize);

    let status = cmd
        .status()
        .context("Failed to run hypno-convert. Is it installed? (cargo install hypno-convert)")?;

    if !status.success() {
        anyhow::bail!("hypno-convert exited with status: {}", status);
    }

    Ok(())
}

fn sanitize_name(model_id: &str) -> String {
    model_id
        .split('/')
        .last()
        .unwrap_or(model_id)
        .replace([' ', '.'], "-")
        .to_lowercase()
}
