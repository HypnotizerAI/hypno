//! `hypno run` — Interactive LLM chat engine.

use clap::Parser;
use hypno_inference::{ForwardBuffers, HypnoConfig, model_forward};
use hypno_loader::HypnoModel;
use hypno_tokenizer::HypnoTokenizer;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Parser)]
pub struct Args {
    /// Path to .hypno model file
    #[arg(short = 'm', long)]
    pub model: String,

    /// Prompt text (non-interactive mode)
    #[arg(short = 'p', long, default_value = "")]
    pub prompt: String,

    /// Maximum new tokens to generate
    #[arg(short = 'n', long, default_value = "128")]
    pub max_tokens: usize,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(short = 't', long, default_value = "0.8")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "40")]
    pub top_k: usize,

    /// Random seed
    #[arg(short = 's', long, default_value = "42")]
    pub seed: u64,

    /// Number of threads for inference
    #[arg(long, default_value = "4")]
    pub threads: usize,
}

pub fn run(args: Args) -> anyhow::Result<()> {
    println!("╔══════════════════════════════════════════╗");
    println!("║         🌀 Hypno LLM Engine v0.1         ║");
    println!("╚══════════════════════════════════════════╝\n");

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap_or(());

    // Load model
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

    let total_tensor_bytes: u64 = model.manifest.tensors.iter().map(|t| t.data_len).sum();
    println!("  Tensors:       {}", model.manifest.tensors.len());
    println!("  Total weights: {:.2} MB\n", total_tensor_bytes as f64 / 1_048_576.0);

    // Load tokenizer
    let tokenizer = HypnoTokenizer::from_hypno_metadata(&model.manifest.metadata)?;
    println!("Tokenizer loaded: {} vocabulary entries\n", tokenizer.vocab_size());

    // Allocate buffers
    let mut buffers = ForwardBuffers::new(&config);

    if args.prompt.is_empty() {
        interactive_mode(&model, &config, &tokenizer, &mut buffers, &args)?;
    } else {
        generate(&model, &config, &tokenizer, &mut buffers, &args)?;
    }

    Ok(())
}

fn generate(
    model: &HypnoModel, config: &HypnoConfig, tokenizer: &HypnoTokenizer,
    buffers: &mut ForwardBuffers, args: &Args,
) -> anyhow::Result<()> {
    let token_ids = tokenizer.encode(&args.prompt);
    print!("Prompt ({} tokens): ", token_ids.len());
    io::stdout().flush()?;

    let prefill_start = Instant::now();
    buffers.reset_cache();

    let mut logits = Vec::new();
    for &tid in &token_ids {
        logits = model_forward(model, config, tid, buffers, true);
    }
    let prefill_elapsed = prefill_start.elapsed();
    let prompt_tps = token_ids.len() as f64 / prefill_elapsed.as_secs_f64().max(0.001);

    println!("{}", args.prompt);
    println!("\n--- Prompt processed in {:.2}ms ({:.1} tok/s) ---",
        prefill_elapsed.as_secs_f64() * 1000.0, prompt_tps);

    let gen_start = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(args.max_tokens);
    let mut rng = XorShift64::new(args.seed);

    print!("> ");
    io::stdout().flush()?;

    for _ in 0..args.max_tokens {
        let next_token = sample_token(&logits, args.temperature, args.top_p, args.top_k, &mut rng);
        if Some(next_token) == tokenizer.eos_token_id() { break; }
        generated_tokens.push(next_token);
        print!("{}", tokenizer.decode(&[next_token]));
        io::stdout().flush()?;
        logits = model_forward(model, config, next_token, buffers, true);
    }

    let gen_elapsed = gen_start.elapsed();
    let gen_tps = generated_tokens.len() as f64 / gen_elapsed.as_secs_f64().max(0.001);
    println!("\n--- Generated {} tokens in {:.2}s ({:.1} tok/s) ---",
        generated_tokens.len(), gen_elapsed.as_secs_f64(), gen_tps);

    Ok(())
}

fn interactive_mode(
    model: &HypnoModel, config: &HypnoConfig, tokenizer: &HypnoTokenizer,
    buffers: &mut ForwardBuffers, args: &Args,
) -> anyhow::Result<()> {
    println!("Interactive mode. Type 'exit' to quit, 'clear' to reset context.\n");

    loop {
        print!("🧠 > ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        match input {
            "exit" | "quit" => break,
            "clear" => { buffers.reset_cache(); println!("Context cleared."); continue; }
            "" => continue,
            _ => {}
        }

        let token_ids = tokenizer.encode(input);
        println!("  [{} tokens]", token_ids.len());

        let mut logits = Vec::new();
        for &tid in &token_ids {
            logits = model_forward(model, config, tid, buffers, true);
        }

        let mut rng = XorShift64::new(args.seed.wrapping_add(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64
        ));

        print!("🤖 ");
        io::stdout().flush()?;
        let mut gen_count = 0;
        for _ in 0..args.max_tokens {
            let next_token = sample_token(&logits, args.temperature, args.top_p, args.top_k, &mut rng);
            if Some(next_token) == tokenizer.eos_token_id() { break; }
            gen_count += 1;
            print!("{}", tokenizer.decode(&[next_token]));
            io::stdout().flush()?;
            logits = model_forward(model, config, next_token, buffers, true);
        }
        println!("\n  [{} tokens generated]\n", gen_count);
    }
    Ok(())
}

fn sample_token(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut XorShift64) -> u32 {
    let n = logits.len();
    let mut probs: Vec<f32> = logits.to_vec();

    if temperature > 0.0 {
        let inv_temp = 1.0 / temperature;
        for p in &mut probs { *p = (*p * inv_temp).exp(); }
    }

    if top_k > 0 && top_k < n {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = indexed[top_k.min(indexed.len()) - 1].1;
        for p in &mut probs { if *p < threshold { *p = 0.0; } }
    }

    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            let mut cumsum = 0.0f32;
            let mut threshold = f32::MAX;
            for (_, p) in &indexed {
                cumsum += p / sum;
                if cumsum > top_p { threshold = *p; break; }
            }
            for p in &mut probs { if *p < threshold { *p = 0.0; } }
        }
    }

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32).unwrap_or(0);
    }

    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p / sum;
        if r < cumsum { return i as u32; }
    }

    probs.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32).unwrap_or(0)
}

struct XorShift64 { state: u64 }

impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f32(&mut self) -> f32 { (self.next() >> 40) as f32 / 16_777_216.0 }
}
