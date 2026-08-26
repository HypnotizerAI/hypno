//! `hypno serve` — OpenAI-compatible HTTP API server.

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use crate::transformer::{ForwardBuffers, HypnoConfig};
use crate::transformer::model_forward;
use crate::loader::HypnoModel;
use crate::tokenizer::HypnoTokenizer;
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

#[derive(Parser)]
pub struct Args {
    #[arg(short = 'm', long)]
    pub model: String,
    #[arg(long, default_value = "8080")]
    pub port: u16,
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    #[arg(long, default_value = "4")]
    pub threads: usize,
}

struct AppState {
    model: Arc<HypnoModel>,
    config: Arc<HypnoConfig>,
    tokenizer: Arc<HypnoTokenizer>,
    buffers: Mutex<ForwardBuffers>,
    models_dir: PathBuf,
    model_name: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)] model: String,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")] max_tokens: usize,
    #[serde(default = "default_temperature")] temperature: f32,
    #[serde(default = "default_top_p")] top_p: f32,
    #[serde(default = "default_top_k")] top_k: usize,
    #[serde(default)] stream: bool,
    #[serde(default)] seed: Option<u64>,
}
fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> usize { 40 }

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage { role: String, content: String }

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String, object: String, created: u64, model: String,
    choices: Vec<ChatChoice>, usage: Usage,
}
#[derive(Serialize)]
struct ChatChoice { index: usize, message: ChatMessage, finish_reason: String }
#[derive(Serialize)]
struct Usage { prompt_tokens: usize, completion_tokens: usize, total_tokens: usize }

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String, object: String, created: u64, model: String,
    choices: Vec<ChunkChoice>,
}
#[derive(Serialize)]
struct ChunkChoice { index: usize, delta: Delta, finish_reason: Option<String> }
#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")] role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] content: Option<String>,
}
#[derive(Serialize)]
struct ModelsList { object: String, data: Vec<ModelInfo> }
#[derive(Serialize)]
struct ModelInfo { id: String, object: String }
#[derive(Serialize)]
struct HealthResponse { status: String }

enum StreamChunk {
    Role, Content(String), Finish(String), End,
}
struct GenParams { max_tokens: usize, temperature: f32, top_p: f32, top_k: usize, seed: u64 }
struct Generation { text: String, prompt_tokens: usize, completion_tokens: usize, finish_reason: String }

pub async fn run(args: Args) -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().unwrap_or(());

    let model = Arc::new(HypnoModel::open(&args.model)?);
    let config = HypnoConfig::from_model(model.as_ref())
        .ok_or_else(|| anyhow::anyhow!("Failed to extract model config from .hypno metadata"))?;
    let tokenizer = HypnoTokenizer::from_hypno_metadata(&model.manifest.metadata)?;
    let buffers = ForwardBuffers::new(&config);

    let model_path = PathBuf::from(&args.model);
    let models_dir = model_path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
    let model_name = model_path.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();

    log::info!("Loaded model: {} (\"{}\")", args.model, model_name);
    log::info!("  hidden_size={} layers={} attention_heads={} kv_heads={} vocab={} max_pos={}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.vocab_size, config.max_position_embeddings);

    let state = Arc::new(AppState {
        model, config: Arc::new(config), tokenizer: Arc::new(tokenizer),
        buffers: Mutex::new(buffers), models_dir, model_name,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    log::info!("hypno server listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> { Json(HealthResponse { status: "ok".to_string() }) }

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
    if !ids.contains(&state.model_name) { ids.push(state.model_name.clone()); }
    ids.sort(); ids.dedup();
    let data = ids.into_iter().map(|id| ModelInfo { id, object: "model".to_string() }).collect();
    Json(ModelsList { object: "list".to_string(), data })
}

async fn chat_completions(State(state): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let prompt = build_prompt(&req.messages);
    let model_name = if req.model.is_empty() { state.model_name.clone() } else { req.model.clone() };
    let created = now_secs();
    let completion_id = make_completion_id();
    if req.stream {
        stream_chat(state, req, prompt, model_name, created, completion_id).await
    } else {
        json_chat(state, req, prompt, model_name, created, completion_id).await
    }
}

async fn json_chat(state: Arc<AppState>, req: ChatRequest, prompt: String, model_name: String, created: u64, completion_id: String) -> Response {
    let st = state.clone();
    let params = GenParams { max_tokens: req.max_tokens, temperature: req.temperature, top_p: req.top_p, top_k: req.top_k, seed: resolve_seed(req.seed) };
    let result = tokio::task::spawn_blocking(move || {
        let mut buffers = st.buffers.lock();
        generate_completion(&st, &prompt, &params, &mut buffers)
    }).await;
    match result {
        Ok(gen) => Json(ChatCompletionResponse {
            id: completion_id, object: "chat.completion".to_string(), created, model: model_name,
            choices: vec![ChatChoice { index: 0, message: ChatMessage { role: "assistant".to_string(), content: gen.text }, finish_reason: gen.finish_reason }],
            usage: Usage { prompt_tokens: gen.prompt_tokens, completion_tokens: gen.completion_tokens, total_tokens: gen.prompt_tokens + gen.completion_tokens },
        }).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("inference task failed: {e}")),
    }
}

async fn stream_chat(state: Arc<AppState>, req: ChatRequest, prompt: String, model_name: String, created: u64, completion_id: String) -> Response {
    let (tx, rx) = mpsc::channel::<StreamChunk>(64);
    let st = state.clone();
    let params = GenParams { max_tokens: req.max_tokens, temperature: req.temperature, top_p: req.top_p, top_k: req.top_k, seed: resolve_seed(req.seed) };
    let _task = tokio::task::spawn_blocking(move || {
        let mut buffers = st.buffers.lock();
        let (mut logits, _) = prefill(&st, &prompt, &mut buffers);
        let mut rng = XorShift64::new(params.seed);
        let _ = tx.blocking_send(StreamChunk::Role);
        let mut finish_reason = "stop".to_string();
        let mut generated = 0usize;
        for _ in 0..params.max_tokens {
            if logits.is_empty() { break; }
            let next = sample_token(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
            if Some(next) == st.tokenizer.eos_token_id() { finish_reason = "stop".to_string(); break; }
            generated += 1;
            let text = st.tokenizer.decode(&[next]);
            if !text.is_empty() && tx.blocking_send(StreamChunk::Content(text)).is_err() { return; }
            logits = model_forward(st.model.as_ref(), st.config.as_ref(), next, &mut buffers, true);
        }
        if generated == params.max_tokens && params.max_tokens > 0 { finish_reason = "length".to_string(); }
        let _ = tx.blocking_send(StreamChunk::Finish(finish_reason));
        let _ = tx.blocking_send(StreamChunk::End);
    });
    let stream = ReceiverStream::new(rx).map(move |chunk| {
        Ok::<Event, Infallible>(build_chunk_event(&chunk, &completion_id, created, &model_name))
    });
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in messages { s.push_str(&m.role); s.push_str(": "); s.push_str(&m.content); s.push('\n'); }
    s.push_str("assistant:");
    s
}

fn prefill(state: &AppState, prompt: &str, buffers: &mut ForwardBuffers) -> (Vec<f32>, usize) {
    let token_ids = state.tokenizer.encode(prompt);
    let prompt_tokens = token_ids.len();
    buffers.reset_cache();
    let mut logits = Vec::new();
    for &tid in &token_ids { logits = model_forward(state.model.as_ref(), state.config.as_ref(), tid, buffers, true); }
    (logits, prompt_tokens)
}

fn generate_completion(state: &AppState, prompt: &str, params: &GenParams, buffers: &mut ForwardBuffers) -> Generation {
    let (mut logits, prompt_tokens) = prefill(state, prompt, buffers);
    let mut rng = XorShift64::new(params.seed);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop".to_string();
    for _ in 0..params.max_tokens {
        if logits.is_empty() { break; }
        let next = sample_token(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
        if Some(next) == state.tokenizer.eos_token_id() { finish_reason = "stop".to_string(); break; }
        completion_tokens += 1;
        text.push_str(&state.tokenizer.decode(&[next]));
        logits = model_forward(state.model.as_ref(), state.config.as_ref(), next, buffers, true);
    }
    if completion_tokens == params.max_tokens && params.max_tokens > 0 { finish_reason = "length".to_string(); }
    Generation { text, prompt_tokens, completion_tokens, finish_reason }
}

fn build_chunk_event(chunk: &StreamChunk, id: &str, created: u64, model: &str) -> Event {
    match chunk {
        StreamChunk::Role => chunk_event(id, created, model, Delta { role: Some("assistant".to_string()), content: None }, None),
        StreamChunk::Content(s) => chunk_event(id, created, model, Delta { role: None, content: Some(s.clone()) }, None),
        StreamChunk::Finish(reason) => chunk_event(id, created, model, Delta { role: None, content: None }, Some(reason.clone())),
        StreamChunk::End => Event::default().data("[DONE]"),
    }
}

fn chunk_event(id: &str, created: u64, model: &str, delta: Delta, finish_reason: Option<String>) -> Event {
    let payload = ChatCompletionChunk { id: id.to_string(), object: "chat.completion.chunk".to_string(), created, model: model.to_string(), choices: vec![ChunkChoice { index: 0, delta, finish_reason }] };
    Event::default().data(serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string()))
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({ "error": { "message": message, "type": "server_error" } });
    (status, Json(body)).into_response()
}
fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }
fn make_completion_id() -> String { format!("chatcmpl-{:x}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64) }
fn resolve_seed(seed: Option<u64>) -> u64 { seed.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64) }

fn sample_token(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut XorShift64) -> u32 {
    let n = logits.len();
    let mut probs: Vec<f32> = logits.to_vec();
    if temperature > 0.0 { let inv = 1.0 / temperature; for p in &mut probs { *p = (*p * inv).exp(); } }
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
            let mut cumsum = 0.0f32; let mut threshold = f32::MAX;
            for (_, p) in &indexed { cumsum += p / sum; if cumsum > top_p { threshold = *p; break; } }
            for p in &mut probs { if *p < threshold { *p = 0.0; } }
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 { return logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i as u32).unwrap_or(0); }
    let r = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, p) in probs.iter().enumerate() { cumsum += p / sum; if r < cumsum { return i as u32; } }
    probs.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i as u32).unwrap_or(0)
}

struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed.max(1) } }
    fn next(&mut self) -> u64 { let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; x }
    fn next_f32(&mut self) -> f32 { (self.next() >> 40) as f32 / 16_777_216.0 }
}
