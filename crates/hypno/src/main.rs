//! `hypno` — Fast local LLM inference in a single binary.
//!
//! ```bash
//! hypno pull Qwen/Qwen2.5-3B              # download from HuggingFace
//! hypno convert model.gguf -o model.hypno  # GGUF/safetensors → .hypno
//! hypno run --model model.hypno            # interactive chat
//! hypno serve --model model.hypno          # OpenAI-compatible API server
//! hypno bench                              # kernel benchmarks
//! ```

// ── Library modules ──
pub mod dtype;
pub mod format;
pub mod quant;
pub mod q4_0;
pub mod q8_0;
pub mod loader;
pub mod mmap;
pub mod kernels;
pub mod ops;
pub mod transformer;
pub mod tokenizer;
pub mod sft_convert;
pub mod gguf;
pub mod lora;

// ── CLI subcommands ──
mod run;
mod serve;
mod convert_cmd;
mod pull;
mod bench;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "hypno", version, about = "Fast local LLM inference", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a model from HuggingFace Hub and convert to .hypno
    Pull(pull::Args),
    /// Convert safetensors, GGUF, or LoRA adapters to .hypno format
    Convert(convert_cmd::Args),
    /// Interactive chat with a .hypno model
    Run(run::Args),
    /// Start an OpenAI-compatible HTTP API server
    Serve(serve::Args),
    /// Run kernel benchmarks and output results
    Bench(bench::Args),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Pull(args) => pull::run(args).await,
        Commands::Convert(args) => convert_cmd::run(args),
        Commands::Run(args) => run::run(args),
        Commands::Serve(args) => serve::run(args).await,
        Commands::Bench(args) => bench::run(args),
    }
}
