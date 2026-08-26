//! `hypno-server` — OpenAI-compatible HTTP API server for Hypno inference.
//!
//! Exposes a small subset of the OpenAI REST surface backed by `.hypno`
//! models loaded with `hypno-loader`, tokenized with `hypno-tokenizer`, and
//! run through the `hypno-inference` engine (flat KV cache forward pass).
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — chat completions (JSON or SSE stream)
//! - `GET  /v1/models`          — list `.hypno` files in the models directory
//! - `GET  /health`             — liveness probe
//!
//! Usage:
//! ```bash
//! hypno-server --model path/to/model.hypno --port 8080 --host 0.0.0.0 --threads 4
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use hypno_inference::{model_forward, ForwardBuffers, HypnoConfig};
use hypno_loader::HypnoModel;
use hypno_tokenizer::HypnoTokenizer;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

// ───────────────────────────── CLI ─────────────────────────────

#[derive(Parser)]
#[command(name = "hypno-server")]
#[command(about = "OpenAI-compatible HTTP API server for Hypno inference")]
#[command(version)]
struct Args {
    /// Path to .hypno model file
    #[arg(short = 'm', long)]
    model: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Host / interface to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Number of threads for inference
    #[arg(long, default_value = "4")]
    threads: usize,
}

// ───────────────────────────── State ─────────────────────────────

/// Shared application state, cloned inside an `Arc` and handed to every handler.
///
/// The loaded model, config and tokenizer are immutable and safe to share
/// across threads. The inference `ForwardBuffers` (which hold the mutable KV
/// cache) are guarded by a mutex so only one generation runs at a time — the
/// KV cache is reset between requests.
struct AppState {
    model: Arc<HypnoModel>,
    config: Arc<HypnoConfig>,
    tokenizer: Arc<HypnoTokenizer>,
    buffers: Mutex<ForwardBuffers>,
    /// Directory scanned by `/v1/models` (the parent of the `--model` path).
    models_dir: PathBuf,
    /// Name of the loaded model (filename without extension).
    model_name: String,
}

// ──────────────────────── Request / Response ────────────────────────

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// Top-k sampling (Hypno extension; OpenAI clients may omit it).
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    stream: bool,
    /// Optional deterministic seed (time-based when omitted).
    #[serde(default)]
    seed: Option<u64>,
}

fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> usize { 40 }

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// Streaming chunk (https://platform.openai.com/docs/api-reference/chat/streaming)
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ModelsList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

/// Messages pushed from the blocking generation task to the SSE stream task.
enum StreamChunk {
    /// First chunk: announce the assistant role.
    Role,
    /// A decoded token fragment.
    Content(String),
    /// Terminal chunk carrying the finish reason.
    Finish(String),
    /// `data: [DONE]` marker.
    End,
}

/// Parameters captured per request and moved into the blocking task.
struct GenParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: u64,
}

/// Result of a non-streaming generation.
struct Generation {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: String,
}

// ───────────────────────────── main ─────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    // Match the CLI: a fixed-size rayon pool backs the SIMD matmul kernels.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap_or(());

    // 1. Load the .hypno model (memory-mapped, zero-copy).
    let model = Arc::new(HypnoModel::open(&args.model)?);

    // 2. Extract architecture config from embedded metadata.
    let config = HypnoConfig::from_model(model.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Failed to extract model config from .hypno metadata"))?;

    // 3. Load the tokenizer embedded in the .hypno metadata.
    let tokenizer = HypnoTokenizer::from_hypno_metadata(&model.manifest.metadata)?;

    // 4. Pre-allocate inference buffers (hidden states + flat KV cache).
    let buffers = ForwardBuffers::new(&config);

    // Models directory = parent of the --model path; model name = file stem.
    let model_path = PathBuf::from(&args.model);
    let models_dir = model_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "model".to_string());

    log::info!("Loaded model: {} (\"{}\")", args.model, model_name);
    log::info!(
        "  hidden_size={} layers={} attention_heads={} kv_heads={} vocab={} max_pos={} threads={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
        config.max_position_embeddings,
        args.threads
    );

    let state = Arc::new(AppState {
        model,
        config: Arc::new(config),
        tokenizer: Arc::new(tokenizer),
        buffers: Mutex::new(buffers),
        models_dir,
        model_name,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("hypno-server listening on http://{}", addr);
    axum::serve(listener, app).await?;

    Ok(())
}

// ──────────────────────────── Handlers ────────────────────────────

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok".to_string() })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsList> {
    let mut ids: Vec<String> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&state.models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("hypno") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    ids.push(stem.to_string());
                }
            }
        }
    }

    // Always include the loaded model, even if its file vanished from disk.
    if !ids.contains(&state.model_name) {
        ids.push(state.model_name.clone());
    }

    ids.sort();
    ids.dedup();

    let data = ids
        .into_iter()
        .map(|id| ModelInfo { id, object: "model".to_string() })
        .collect();

    Json(ModelsList { object: "list".to_string(), data })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let prompt = build_prompt(&req.messages);
    let model_name = if req.model.is_empty() {
        state.model_name.clone()
    } else {
        req.model.clone()
    };
    let created = now_secs();
    let completion_id = make_completion_id();

    if req.stream {
        stream_chat(state, req, prompt, model_name, created, completion_id).await
    } else {
        json_chat(state, req, prompt, model_name, created, completion_id).await
    }
}

/// Non-streaming completion: run the whole generation on a blocking thread and
/// return a single `chat.completion` JSON object.
async fn json_chat(
    state: Arc<AppState>,
    req: ChatRequest,
    prompt: String,
    model_name: String,
    created: u64,
    completion_id: String,
) -> Response {
    let st = state.clone();
    let params = GenParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: resolve_seed(req.seed),
    };

    let result = tokio::task::spawn_blocking(move || {
        let mut buffers = st.buffers.lock();
        generate_completion(&st, &prompt, &params, &mut buffers)
    })
    .await;

    match result {
        Ok(gen) => {
            let resp = ChatCompletionResponse {
                id: completion_id,
                object: "chat.completion".to_string(),
                created,
                model: model_name,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: gen.text,
                    },
                    finish_reason: gen.finish_reason,
                }],
                usage: Usage {
                    prompt_tokens: gen.prompt_tokens,
                    completion_tokens: gen.completion_tokens,
                    total_tokens: gen.prompt_tokens + gen.completion_tokens,
                },
            };
            Json(resp).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("inference task failed: {e}"),
        ),
    }
}

/// Streaming completion: a blocking task generates token-by-token and pushes
/// decoded fragments through an mpsc channel; the async side converts them into
/// SSE `data:` events. Holding the buffer mutex only inside the blocking task
/// keeps the async executor free and lets client disconnects cancel generation.
async fn stream_chat(
    state: Arc<AppState>,
    req: ChatRequest,
    prompt: String,
    model_name: String,
    created: u64,
    completion_id: String,
) -> Response {
    let (tx, rx) = mpsc::channel::<StreamChunk>(64);

    let st = state.clone();
    let params = GenParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: resolve_seed(req.seed),
    };

    // Fire-and-forget: the channel drives completion; when the SSE response is
    // dropped (client gone) `blocking_send` errors and the loop exits.
    let _task = tokio::task::spawn_blocking(move || {
        let mut buffers = st.buffers.lock();
        let (mut logits, _prompt_tokens) = prefill(&st, &prompt, &mut buffers);
        let mut rng = XorShift64::new(params.seed);

        // Lead with the assistant role delta.
        let _ = tx.blocking_send(StreamChunk::Role);

        let mut finish_reason = "stop".to_string();
        let mut generated = 0usize;

        for _ in 0..params.max_tokens {
            if logits.is_empty() {
                break;
            }
            let next = sample_token(&logits, params.temperature, params.top_p, params.top_k, &mut rng);

            if Some(next) == st.tokenizer.eos_token_id() {
                finish_reason = "stop".to_string();
                break;
            }

            generated += 1;
            let text = st.tokenizer.decode(&[next]);
            if !text.is_empty()
                && tx.blocking_send(StreamChunk::Content(text)).is_err()
            {
                // Client disconnected — stop spending compute.
                return;
            }

            logits = model_forward(st.model.as_ref(), st.config.as_ref(), next, &mut buffers, true);
        }

        if generated == params.max_tokens && params.max_tokens > 0 {
            finish_reason = "length".to_string();
        }

        let _ = tx.blocking_send(StreamChunk::Finish(finish_reason));
        let _ = tx.blocking_send(StreamChunk::End);
        // `tx` drops here → ReceiverStream terminates.
    });

    let stream = ReceiverStream::new(rx).map(move |chunk| {
        Ok::<Event, Infallible>(build_chunk_event(
            &chunk,
            &completion_id,
            created,
            &model_name,
        ))
    });

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ──────────────────────────── Inference ────────────────────────────

/// Build a single prompt string from the OpenAI chat `messages` array.
///
/// Uses a minimal role-prefixed template suitable for base LLMs, ending with an
/// `assistant:` cue so the model continues as the assistant.
fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(m.role.as_str());
        s.push_str(": ");
        s.push_str(&m.content);
        s.push('\n');
    }
    s.push_str("assistant:");
    s
}

/// Run the prompt prefill through the model with a freshly reset KV cache and
/// return the logits for the final prompt token (plus the prompt token count).
fn prefill(state: &AppState, prompt: &str, buffers: &mut ForwardBuffers) -> (Vec<f32>, usize) {
    let token_ids = state.tokenizer.encode(prompt);
    let prompt_tokens = token_ids.len();

    buffers.reset_cache();
    let mut logits = Vec::new();
    for &tid in &token_ids {
        logits = model_forward(state.model.as_ref(), state.config.as_ref(), tid, buffers, true);
    }
    (logits, prompt_tokens)
}

/// Generate a full completion (non-streaming). Must be called while holding the
/// buffer lock.
fn generate_completion(
    state: &AppState,
    prompt: &str,
    params: &GenParams,
    buffers: &mut ForwardBuffers,
) -> Generation {
    let (mut logits, prompt_tokens) = prefill(state, prompt, buffers);
    let mut rng = XorShift64::new(params.seed);

    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop".to_string();

    for _ in 0..params.max_tokens {
        if logits.is_empty() {
            break;
        }
        let next = sample_token(&logits, params.temperature, params.top_p, params.top_k, &mut rng);

        if Some(next) == state.tokenizer.eos_token_id() {
            finish_reason = "stop".to_string();
            break;
        }

        completion_tokens += 1;
        text.push_str(&state.tokenizer.decode(&[next]));
        logits = model_forward(state.model.as_ref(), state.config.as_ref(), next, buffers, true);
    }

    if completion_tokens == params.max_tokens && params.max_tokens > 0 {
        finish_reason = "length".to_string();
    }

    Generation {
        text,
        prompt_tokens,
        completion_tokens,
        finish_reason,
    }
}

// ──────────────────────────── Helpers ────────────────────────────

/// Render a streaming chunk as an axum SSE `Event` with a `data:` JSON payload.
fn build_chunk_event(chunk: &StreamChunk, id: &str, created: u64, model: &str) -> Event {
    match chunk {
        StreamChunk::Role => chunk_event(
            id,
            created,
            model,
            Delta { role: Some("assistant".to_string()), content: None },
            None,
        ),
        StreamChunk::Content(s) => chunk_event(
            id,
            created,
            model,
            Delta { role: None, content: Some(s.clone()) },
            None,
        ),
        StreamChunk::Finish(reason) => chunk_event(
            id,
            created,
            model,
            Delta { role: None, content: None },
            Some(reason.clone()),
        ),
        StreamChunk::End => Event::default().data("[DONE]"),
    }
}

fn chunk_event(id: &str, created: u64, model: &str, delta: Delta, finish_reason: Option<String>) -> Event {
    let payload = ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice { index: 0, delta, finish_reason }],
    };
    let json = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    Event::default().data(json)
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({
        "error": { "message": message, "type": "server_error" }
    });
    (status, Json(body)).into_response()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn make_completion_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    format!("chatcmpl-{nanos:x}")
}

/// Use the client-provided seed when present, otherwise a time-based one so
/// repeated identical requests vary.
fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    })
}

// ──────────────────────── Sampling (from hypno-cli) ────────────────────────
//
// Copied verbatim from `hypno-cli/src/main.rs`: XorShift64 PRNG with
// temperature, top-k and top-p (nucleus) filtering.

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

    #[test]
    fn test_build_prompt() {
        let msgs = vec![
            ChatMessage { role: "system".to_string(), content: "be nice".to_string() },
            ChatMessage { role: "user".to_string(), content: "hi".to_string() },
        ];
        let p = build_prompt(&msgs);
        assert_eq!(p, "system: be nice\nuser: hi\nassistant:");
    }
}
