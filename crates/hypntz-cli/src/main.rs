//! `hypnotizer-cli` — Interactive LLM chat engine using `.hypno` models.
//!
//! Usage:
//! ```bash
//! hypnotizer-cli --model path/to/model.hypno --prompt "Hello, world!"
//! hypnotizer-cli --model model.hypno                     # interactive mode
//! ```

use clap::Parser;
use hypntz_inference::{
    ForwardBuffers, HypnoConfig, model_forward,
};
use hypntz_loader::HypnoModel;
use hypntz_tokenizer::HypnoTokenizer;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "hypnotizer-cli")]
#[command(about = "Interactive LLM inference using .hypno models")]
struct Args {
    /// Path to .hypno model file
    #[arg(short = 'm', long)]
    model: String,

    /// Prompt text (non-interactive mode)
    #[arg(short = 'p', long, default_value = "")]
    prompt: String,

    /// Maximum new tokens to generate
    #[arg(short = 'n', long, default_value = "128")]
    max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(short = 't', long, default_value = "0.8")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.9")]
    top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "40")]
    top_k: usize,

    /// Random seed
    #[arg(short = 's', long, default_value = "42")]
    seed: u64,

    /// Number of threads for inference
    #[arg(long, default_value = "4")]
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("╔══════════════════════════════════════════╗");
    println!("║       🌀 Hypnotizer LLM Engine v0.1     ║");
    println!("╚══════════════════════════════════════════╝");
    println!();

    // Set thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap_or(());

    // 1. Load model
    println!("Loading model: {}", args.model);
    let load_start = Instant::now();
    let model = HypnoModel::open(&args.model)?;
    let config = HypnoConfig::from_model(&model)
        .ok_or_else(|| anyhow::anyhow!("Failed to extract model config from .hypno metadata"))?;
    let load_elapsed = load_start.elapsed();

    println!("  Architecture:  {}", model.get_metadata("architecture").unwrap_or("unknown"));
    println!("  Hidden size:   {}", config.hidden_size);
    println!("  Layers:        {}", config.num_hidden_layers);
    println!("  Attention:     {} heads ({} KV)", config.num_attention_heads, config.num_key_value_heads);
    println!("  Vocab size:    {}", config.vocab_size);
    println!("  Max position:  {}", config.max_position_embeddings);
    println!("  Load time:     {:.2}ms", load_elapsed.as_secs_f64() * 1000.0);
    println!();

    // Check tensor count and total size
    let total_tensor_bytes: u64 = model.manifest.tensors.iter()
        .map(|t| t.data_len)
        .sum();
    println!("  Tensors:       {}", model.manifest.tensors.len());
    println!("  Total weights: {:.2} MB", total_tensor_bytes as f64 / 1_048_576.0);
    println!();

    // 2. Load tokenizer
    let tokenizer = HypnoTokenizer::from_hypno_metadata(&model.manifest.metadata)?;
    println!("Tokenizer loaded: {} vocabulary entries", tokenizer.vocab_size());

    // 3. Pre-allocate buffers
    println!("Allocating inference buffers...");
    let mut buffers = ForwardBuffers::new(&config);
    println!("  Hidden dim: {} floats", buffers.hidden.len());
    println!();

    // 4. Run inference
    if args.prompt.is_empty() {
        // Interactive mode
        interactive_mode(&model, &config, &tokenizer, &mut buffers, &args)?;
    } else {
        // Single generation
        generate(&model, &config, &tokenizer, &mut buffers, &args)?;
    }

    Ok(())
}

fn generate(
    model: &HypnoModel,
    config: &HypnoConfig,
    tokenizer: &HypnoTokenizer,
    buffers: &mut ForwardBuffers,
    args: &Args,
) -> anyhow::Result<()> {
    // Tokenize input
    let token_ids = tokenizer.encode(&args.prompt);

    print!("Prompt ({} tokens): ", token_ids.len());
    io::stdout().flush()?;

    // Process prompt tokens (prefill)
    let prefill_start = Instant::now();
    buffers.reset_cache();

    let mut logits = Vec::new();
    for (i, &tid) in token_ids.iter().enumerate() {
        let _is_last = i == token_ids.len() - 1;
        logits = model_forward(model, config, tid, buffers, true);
    }
    let prefill_elapsed = prefill_start.elapsed();
    let prompt_tps = token_ids.len() as f64 / prefill_elapsed.as_secs_f64().max(0.001);

    println!("{}", args.prompt);
    println!();
    println!("--- Prompt processed in {:.2}ms ({:.1} tok/s) ---", 
        prefill_elapsed.as_secs_f64() * 1000.0, prompt_tps);

    // Generate
    let gen_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let mut rng = XorShift64::new(args.seed);

    print!("> ");
    io::stdout().flush()?;

    for _ in 0..args.max_tokens {
        // Sample from logits
        let next_token = sample_token(&logits, args.temperature, args.top_p, args.top_k, &mut rng);

        // Check for EOS
        if Some(next_token) == tokenizer.eos_token_id() {
            break;
        }

        generated_tokens.push(next_token);

        // Decode and print
        let text = tokenizer.decode(&[next_token]);
        print!("{}", text);
        io::stdout().flush()?;

        // Forward for next token
        logits = model_forward(model, config, next_token, buffers, true);
    }

    let gen_elapsed = gen_start.elapsed();
    let gen_tps = generated_tokens.len() as f64 / gen_elapsed.as_secs_f64().max(0.001);

    println!();
    println!("--- Generated {} tokens in {:.2}s ({:.1} tok/s) ---",
        generated_tokens.len(),
        gen_elapsed.as_secs_f64(),
        gen_tps
    );

    Ok(())
}

fn interactive_mode(
    model: &HypnoModel,
    config: &HypnoConfig,
    tokenizer: &HypnoTokenizer,
    buffers: &mut ForwardBuffers,
    args: &Args,
) -> anyhow::Result<()> {
    println!("Interactive mode. Type 'exit' to quit, 'clear' to reset context.");
    println!();

    loop {
        print!("🧠 > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "exit" | "quit" => break,
            "clear" => {
                buffers.reset_cache();
                println!("Context cleared.");
                continue;
            }
            "" => continue,
            _ => {}
        }

        // Tokenize
        let token_ids = tokenizer.encode(input);
        println!("  [{} tokens]", token_ids.len());

        // Processing
        let mut logits = Vec::new();
        for (_i, &tid) in token_ids.iter().enumerate() {
            logits = model_forward(model, config, tid, buffers, true);
        }

        // Generate response
        let mut rng = XorShift64::new(args.seed.wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        ));

        print!("🤖 ");
        io::stdout().flush()?;

        let mut gen_count = 0;
        for _ in 0..args.max_tokens {
            let next_token = sample_token(&logits, args.temperature, args.top_p, args.top_k, &mut rng);

            if Some(next_token) == tokenizer.eos_token_id() {
                break;
            }

            gen_count += 1;
            let text = tokenizer.decode(&[next_token]);
            print!("{}", text);
            io::stdout().flush()?;

            logits = model_forward(model, config, next_token, buffers, true);
        }

        println!();
        println!("  [{} tokens generated]", gen_count);
        println!();
    }

    Ok(())
}

/// Sample a token from logits using temperature, top-p, and top-k.
fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    rng: &mut XorShift64,
) -> u32 {
    let n = logits.len();
    let mut probs: Vec<f32> = logits.to_vec();

    // Temperature scaling
    if temperature > 0.0 {
        let inv_temp = 1.0 / temperature;
        for p in &mut probs {
            *p = (*p * inv_temp).exp();
        }
    }

    // Top-k filtering
    if top_k > 0 && top_k < n {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[top_k.min(indexed.len()) - 1].1;
        // Zero out below threshold
        for p in &mut probs {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    // Top-p (nucleus) filtering
    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let mut cumsum = 0.0f32;
            let mut threshold = f32::MAX;
            for (_, p) in &indexed {
                cumsum += p / sum;
                if cumsum > top_p {
                    threshold = *p;
                    break;
                }
            }
            for p in &mut probs {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }
    }

    // Normalize
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        // Greedy fallback
        return logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Sample
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p / sum;
        if r < cumsum {
            return i as u32;
        }
    }

    // Fallback: argmax
    probs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Simple XorShift64 PRNG for deterministic sampling.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Use upper 24 bits for good float distribution
        let bits = self.next() >> 40;
        bits as f32 / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_greedy() {
        let logits = vec![0.1f32, 0.5, 2.0, 0.3];
        let mut rng = XorShift64::new(42);
        // With temperature=0, should be greedy (argmax=2)
        let token = sample_token(&logits, 0.01, 1.0, 0, &mut rng);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_rng_determinism() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }
}
