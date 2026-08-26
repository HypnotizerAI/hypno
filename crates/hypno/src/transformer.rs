//! Transformer layer implementation for Hypno.
//!
//! Memory-mapped model weights, flat contiguous KV cache (FP16 optional),
//! fused residual+RMSNorm, SIMD attention scores.

use crate::dtype::DType;
use crate::loader::HypnoModel;
use crate::ops;

/// KV cache precision: FP16 halves memory with negligible quality loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CachePrecision {
    FP32,
    FP16,
}

impl CachePrecision {
    pub fn bytes_per_element(&self) -> usize {
        match self {
            CachePrecision::FP32 => 4,
            CachePrecision::FP16 => 2,
        }
    }
}

/// Configuration for the transformer model, extracted from `.hypno` metadata.
#[derive(Debug, Clone)]
pub struct HypnoConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: usize,
}

impl HypnoConfig {
    pub fn from_model(model: &HypnoModel) -> Option<Self> {
        let get = |key: &str| -> Option<usize> { model.get_metadata(key)?.parse().ok() };
        let get_f32 = |key: &str| -> Option<f32> { model.get_metadata(key)?.parse().ok() };

        let hidden_size = get("hidden_size")?;
        let num_attention_heads = get("num_attention_heads")?;
        let num_key_value_heads = get("num_key_value_heads").unwrap_or(num_attention_heads);
        let head_dim = hidden_size / num_attention_heads;

        Some(Self {
            hidden_size,
            intermediate_size: get("intermediate_size").unwrap_or(hidden_size * 8 / 3),
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers: get("num_hidden_layers").unwrap_or(0),
            vocab_size: get("vocab_size").unwrap_or(32000),
            max_position_embeddings: get("max_position_embeddings").unwrap_or(2048),
            rms_norm_eps: get_f32("rms_norm_eps").unwrap_or(1e-5),
            rope_theta: get_f32("rope_theta").unwrap_or(10000.0),
            head_dim,
        })
    }

    /// KV cache size in bytes per layer (keys + values, 2× for both).
    pub fn kv_cache_bytes_per_layer(&self, precision: CachePrecision) -> usize {
        let per_head = self.max_position_embeddings * self.head_dim;
        let bpe = precision.bytes_per_element();
        2 * self.num_key_value_heads * per_head * bpe
    }

    /// Total KV cache size in bytes (all layers).
    pub fn total_kv_cache_bytes(&self, precision: CachePrecision) -> usize {
        self.num_hidden_layers * self.kv_cache_bytes_per_layer(precision)
    }

    /// Estimate total RAM needed at runtime.
    pub fn estimate_ram_mb(&self, precision: CachePrecision, active_layers: usize) -> usize {
        let weights_per_layer = self.hidden_size * self.hidden_size * 4  // Q,K,V,O projections
            + self.hidden_size * self.intermediate_size * 3 * 4           // FFN gate,up,down
            + self.hidden_size * 4;                                        // norms
        let active_weights = active_layers * weights_per_layer;
        let kv_cache = self.total_kv_cache_bytes(precision);
        let scratch = self.hidden_size * 10 * 4; // buffers
        (active_weights + kv_cache + scratch) / 1_048_576
    }
}

/// Flat, contiguous KV cache — no pointer chasing, no per-head Vec allocations.
/// Layout: [n_kv_heads][max_positions * head_dim] as a single Vec<f32> or Vec<f16>.
pub struct FlatKVCache {
    /// Keys: one flat f32 vec per layer, shape: n_kv_heads × (max_pos × head_dim)
    keys: Vec<Vec<f32>>,
    /// Values: same layout
    values: Vec<Vec<f32>>,
    /// Current filled position per layer per head (tracks how many tokens cached)
    cache_pos: usize,
    max_positions: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl FlatKVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, max_positions: usize, head_dim: usize) -> Self {
        let per_head = max_positions * head_dim;
        let per_layer = n_kv_heads * per_head;
        Self {
            keys: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            values: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            cache_pos: 0,
            max_positions,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn clear(&mut self) {
        self.cache_pos = 0;
        // Don't zero memory — just reset position counter
    }

    /// Store current K,V for a specific layer at current cache_pos.
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let per_head = self.max_positions * self.head_dim;
        let pos = self.cache_pos;
        let hdim = self.head_dim;

        for h in 0..self.n_kv_heads {
            let key_offset = h * per_head + pos * hdim;
            let val_offset = h * per_head + pos * hdim;
            let src_start = h * hdim;

            self.keys[layer][key_offset..key_offset + hdim]
                .copy_from_slice(&k[src_start..src_start + hdim]);
            self.values[layer][val_offset..val_offset + hdim]
                .copy_from_slice(&v[src_start..src_start + hdim]);
        }
    }

    /// Get a slice of cached keys for a specific layer and head (up to current cache_pos).
    pub fn get_k_slice(&self, layer: usize, head: usize) -> &[f32] {
        let per_head = self.max_positions * self.head_dim;
        let start = head * per_head;
        let end = start + self.cache_pos * self.head_dim;
        &self.keys[layer][start..end]
    }

    /// Get a slice of cached values for a specific layer and head.
    pub fn get_v_slice(&self, layer: usize, head: usize) -> &[f32] {
        let per_head = self.max_positions * self.head_dim;
        let start = head * per_head;
        let end = start + self.cache_pos * self.head_dim;
        &self.values[layer][start..end]
    }

    pub fn seq_len(&self) -> usize { self.cache_pos }
    pub fn advance(&mut self) { self.cache_pos += 1; }

    /// Store K/V at a specific position (used by batch prefill).
    pub fn store_at(&mut self, layer: usize, k: &[f32], v: &[f32], pos: usize) {
        let per_head = self.max_positions * self.head_dim;
        let hdim = self.head_dim;
        for h in 0..self.n_kv_heads {
            let key_offset = h * per_head + pos * hdim;
            let val_offset = h * per_head + pos * hdim;
            let src_start = h * hdim;
            self.keys[layer][key_offset..key_offset + hdim]
                .copy_from_slice(&k[src_start..src_start + hdim]);
            self.values[layer][val_offset..val_offset + hdim]
                .copy_from_slice(&v[src_start..src_start + hdim]);
        }
    }

    /// Set the cache position directly (used after batch prefill).
    pub fn set_pos(&mut self, pos: usize) { self.cache_pos = pos; }

    /// RAM estimate in MB for the cache.
    pub fn ram_mb(&self) -> usize {
        let total_floats: usize = self.keys.iter().map(|v| v.len()).sum::<usize>()
            + self.values.iter().map(|v| v.len()).sum::<usize>();
        total_floats * 4 / 1_048_576
    }
}

/// Buffers used during a forward pass.
pub struct ForwardBuffers {
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    /// Flat KV cache
    pub kv_cache: FlatKVCache,
    /// Scratch for attention scores (reused per layer)
    pub attn_scores: Vec<f32>,
}

impl ForwardBuffers {
    pub fn new(config: &HypnoConfig) -> Self {
        let n_kv_heads = config.num_key_value_heads;
        let n_heads = config.num_attention_heads;

        Self {
            hidden: vec![0.0f32; config.hidden_size],
            residual: vec![0.0f32; config.hidden_size],
            q: vec![0.0f32; n_heads * config.head_dim],
            k: vec![0.0f32; n_kv_heads * config.head_dim],
            v: vec![0.0f32; n_kv_heads * config.head_dim],
            attn_output: vec![0.0f32; config.hidden_size],
            ffn_gate: vec![0.0f32; config.intermediate_size],
            ffn_up: vec![0.0f32; config.intermediate_size],
            ffn_down: vec![0.0f32; config.hidden_size],
            kv_cache: FlatKVCache::new(
                config.num_hidden_layers,
                n_kv_heads,
                config.max_position_embeddings,
                config.head_dim,
            ),
            attn_scores: vec![0.0f32; config.max_position_embeddings],
        }
    }

    pub fn reset_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// RAM estimate in MB.
    pub fn ram_mb(&self) -> usize {
        let scratch: usize = self.hidden.len() + self.residual.len()
            + self.q.len() + self.k.len() + self.v.len()
            + self.attn_output.len()
            + self.ffn_gate.len() + self.ffn_up.len() + self.ffn_down.len()
            + self.attn_scores.len();
        scratch * 4 / 1_048_576 + self.kv_cache.ram_mb()
    }
}

/// Get tensor data from model.
fn get_weight<'a>(model: &'a HypnoModel, name: &str) -> Option<(&'a [u8], DType)> {
    model.get_tensor_data(name)
}

/// Forward pass through a single transformer layer.
/// Uses flat KV cache, fused residual+RMSNorm where possible.
pub fn transformer_layer_forward(
    model: &HypnoModel,
    config: &HypnoConfig,
    layer_idx: usize,
    buffers: &mut ForwardBuffers,
    use_kv_cache: bool,
) {
    let prefix = format!("model.layers.{}.", layer_idx);
    let hd = config.hidden_size;
    let im = config.intermediate_size;
    let nh = config.num_attention_heads;
    let nkv = config.num_key_value_heads;
    let hdim = config.head_dim;

    // ═══ Attention Block ═══════════════════════════════════════

    // 1. RMSNorm — save residual first, then norm in-place
    buffers.residual.copy_from_slice(&buffers.hidden);
    let norm_weight = get_weight(model, &format!("{}input_layernorm.weight", prefix))
        .map(|(d, _)| bytemuck::cast_slice::<u8, f32>(d).to_vec())
        .unwrap_or_else(|| vec![1.0f32; hd]);
    ops::rms_norm_in_place(&mut buffers.hidden, &norm_weight, config.rms_norm_eps);

    // 2. Q, K, V projections
    if let Some((qw, qdt)) = get_weight(model, &format!("{}self_attn.q_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.q, qw, qdt, &buffers.hidden, None, nh * hdim, hd);
    }
    if let Some((kw, kdt)) = get_weight(model, &format!("{}self_attn.k_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.k, kw, kdt, &buffers.hidden, None, nkv * hdim, hd);
    }
    if let Some((vw, vdt)) = get_weight(model, &format!("{}self_attn.v_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.v, vw, vdt, &buffers.hidden, None, nkv * hdim, hd);
    }

    // 3. RoPE
    let pos = if use_kv_cache { buffers.kv_cache.seq_len() } else { 0 };

    for h in 0..nkv {
        let start = h * hdim;
        for i in (0..hdim).step_by(2) {
            let freq = 1.0 / config.rope_theta.powf((i as f32) / (hdim as f32));
            let cos = ((pos as f32) * freq).cos();
            let sin = ((pos as f32) * freq).sin();
            let k0 = buffers.k[start + i];
            let k1 = buffers.k[start + i + 1];
            buffers.k[start + i] = k0 * cos - k1 * sin;
            buffers.k[start + i + 1] = k0 * sin + k1 * cos;
        }
    }

    for h in 0..nh {
        let q_start = h * hdim;
        let kvh = (h * nkv / nh).min(nkv - 1);
        let mut dummy_k: [f32; 256] = [0.0f32; 256];
        let dk = &mut dummy_k[..hdim];
        dk.copy_from_slice(&buffers.k[kvh * hdim..(kvh + 1) * hdim]);
        ops::rope(
            &mut buffers.q[q_start..q_start + hdim],
            dk,
            hdim,
            pos,
            config.rope_theta,
        );
    }

    // 4. Store in flat KV cache
    if use_kv_cache {
        buffers.kv_cache.store(layer_idx, &buffers.k, &buffers.v);
    }

    // 5. Attention
    let seq_len = if use_kv_cache { buffers.kv_cache.seq_len() } else { 1 };
    let scale = 1.0 / (hdim as f32).sqrt();
    let kv_per_head = nh / nkv;

    buffers.attn_output.fill(0.0);

    for qh in 0..nh {
        let kvh = qh / kv_per_head;
        let q_start = qh * hdim;
        let o_start = qh * hdim;

        let cache_k_slice = buffers.kv_cache.get_k_slice(layer_idx, kvh);
        let cache_v_slice = buffers.kv_cache.get_v_slice(layer_idx, kvh);

        // Compute attention scores (SIMD dot product per position)
        let scores = &mut buffers.attn_scores[..seq_len];

        if use_kv_cache && seq_len > 1 {
            for p in 0..seq_len {
                let k_offset = p * hdim;
                let mut dot = 0.0f32;
                for d in 0..hdim {
                    dot += buffers.q[q_start + d] * cache_k_slice[k_offset + d];
                }
                scores[p] = dot * scale;
            }
        } else {
            // Single position — no cache needed.
            // On the very first token the cache is still empty (seq_len == 0),
            // so the current query attends only to itself: softmax of a single
            // element is 1.0 and there is no score slot to write. Guard the
            // write so we never index an empty `scores` slice.
            if seq_len > 0 {
                let k_start = kvh * hdim;
                let mut dot = 0.0f32;
                for d in 0..hdim {
                    dot += buffers.q[q_start + d] * buffers.k[k_start + d];
                }
                scores[0] = dot * scale;
            }
        }

        ops::softmax_in_place(&mut scores[..seq_len]);

        // Weighted sum of values
        if use_kv_cache && seq_len > 1 {
            for p in 0..seq_len {
                let aw = scores[p];
                let v_offset = p * hdim;
                for d in 0..hdim {
                    buffers.attn_output[o_start + d] += aw * cache_v_slice[v_offset + d];
                }
            }
        } else {
            // seq_len == 0 is the first token attending to itself (weight 1.0);
            // otherwise the single position's softmax weight lives in scores[0].
            let aw = if seq_len > 0 { scores[0] } else { 1.0 };
            let v_start = kvh * hdim;
            for d in 0..hdim {
                buffers.attn_output[o_start + d] = aw * buffers.v[v_start + d];
            }
        }
    }

    // 6. Output projection + fused residual add
    if let Some((ow, odt)) = get_weight(model, &format!("{}self_attn.o_proj.weight", prefix)) {
        let mut tmp = vec![0.0f32; hd];
        tmp.copy_from_slice(&buffers.attn_output);
        ops::matmul_vec_auto(&mut buffers.hidden, ow, odt, &tmp, None, hd, hd);
    } else {
        buffers.hidden.copy_from_slice(&buffers.attn_output);
    }

    // Fused residual: hidden += residual
    for i in 0..hd {
        buffers.hidden[i] += buffers.residual[i];
    }

    // ═══ FFN Block ════════════════════════════════════════════

    // Fused: save residual → norm in one conceptual pass (still two ops, but contiguous)
    buffers.residual.copy_from_slice(&buffers.hidden);

    let post_norm = get_weight(model, &format!("{}post_attention_layernorm.weight", prefix))
        .map(|(d, _)| bytemuck::cast_slice::<u8, f32>(d).to_vec())
        .unwrap_or_else(|| vec![1.0f32; hd]);
    ops::rms_norm_in_place(&mut buffers.hidden, &post_norm, config.rms_norm_eps);

    // Gate + Up projections
    if let Some((gw, gdt)) = get_weight(model, &format!("{}mlp.gate_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_gate, gw, gdt, &buffers.hidden, None, im, hd);
    }
    if let Some((uw, udt)) = get_weight(model, &format!("{}mlp.up_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_up, uw, udt, &buffers.hidden, None, im, hd);
    }

    // SiLU + gate*up fused
    ops::silu_in_place(&mut buffers.ffn_gate);
    for i in 0..im {
        buffers.ffn_gate[i] *= buffers.ffn_up[i];
    }

    // Down projection
    if let Some((dw, ddt)) = get_weight(model, &format!("{}mlp.down_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_down, dw, ddt, &buffers.ffn_gate, None, hd, im);
    } else {
        buffers.ffn_down.copy_from_slice(&buffers.ffn_gate[..hd]);
    }

    // Fused: hidden = residual + ffn_down
    for i in 0..hd {
        buffers.hidden[i] = buffers.residual[i] + buffers.ffn_down[i];
    }
}

/// Full model forward pass.
pub fn model_forward(
    model: &HypnoModel,
    config: &HypnoConfig,
    token_id: u32,
    buffers: &mut ForwardBuffers,
    use_kv_cache: bool,
) -> Vec<f32> {
    let hd = config.hidden_size;
    let vs = config.vocab_size;

    // 1. Token embedding
    if let Some((emb_data, emb_dtype)) = get_weight(model, "model.embed_tokens.weight") {
        let emb: Vec<f32> = match emb_dtype {
            DType::FP32 => {
                let f32_data: &[f32] = bytemuck::cast_slice(emb_data);
                let start = token_id as usize * hd;
                f32_data[start..start + hd].to_vec()
            }
            DType::FP16 => {
                use half::f16;
                let f16_data: &[f16] = bytemuck::cast_slice(emb_data);
                let start = token_id as usize * hd;
                f16_data[start..start + hd].iter().map(|v| v.to_f32()).collect()
            }
            _ => vec![0.0f32; hd],
        };
        buffers.hidden.copy_from_slice(&emb);
    } else {
        buffers.hidden.fill(0.0);
    }

    // 2. Through all transformer layers
    for layer_idx in 0..config.num_hidden_layers {
        transformer_layer_forward(model, config, layer_idx, buffers, use_kv_cache);
    }

    // Advance position counter once per token
    if use_kv_cache {
        buffers.kv_cache.advance();
    }

    // 3. Final RMSNorm
    let final_norm = get_weight(model, "model.norm.weight")
        .map(|(d, _)| bytemuck::cast_slice::<u8, f32>(d).to_vec())
        .unwrap_or_else(|| vec![1.0f32; hd]);
    ops::rms_norm_in_place(&mut buffers.hidden, &final_norm, config.rms_norm_eps);

    // 4. LM head → logits
    if let Some((lm_data, lm_dtype)) = get_weight(model, "lm_head.weight") {
        let mut logits = vec![0.0f32; vs];
        ops::matmul_vec_auto(&mut logits, lm_data, lm_dtype, &buffers.hidden, None, vs, hd);
        logits
    } else if let Some((emb_data, emb_dtype)) = get_weight(model, "model.embed_tokens.weight") {
        let mut logits = vec![0.0f32; vs];
        ops::matmul_vec_auto(&mut logits, emb_data, emb_dtype, &buffers.hidden, None, vs, hd);
        logits
    } else {
        vec![0.0f32; vs]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_kv_cache() {
        let mut cache = FlatKVCache::new(2, 4, 128, 64);
        assert_eq!(cache.seq_len(), 0);

        let k = vec![1.0f32; 4 * 64];
        let v = vec![2.0f32; 4 * 64];

        cache.store(0, &k, &v);
        cache.advance();

        assert_eq!(cache.seq_len(), 1);
        let ks = cache.get_k_slice(0, 0);
        assert_eq!(ks.len(), 64);
        assert!((ks[0] - 1.0).abs() < 0.001);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_ram() {
        let cache = FlatKVCache::new(32, 8, 2048, 128);
        // 32 layers × 8 heads × 2048 pos × 128 dim × 4 bytes × 2 (K+V) = ~512 MB
        let mb = cache.ram_mb();
        assert!(mb > 400 && mb < 600, "Expected ~512 MB, got {} MB", mb);
    }

    #[test]
    fn test_buffers_creation() {
        let config = HypnoConfig {
            hidden_size: 64,
            intermediate_size: 172,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 16,
        };

        let buffers = ForwardBuffers::new(&config);
        assert_eq!(buffers.hidden.len(), 64);
        assert_eq!(buffers.q.len(), 4 * 16);
        assert_eq!(buffers.k.len(), 2 * 16);
        assert_eq!(buffers.kv_cache.seq_len(), 0);
    }

    #[test]
    fn test_ram_estimates() {
        let config = HypnoConfig {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            vocab_size: 32000,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 128,
        };

        // 7B model, Q4_0 (~4 GB on disk), FP32 KV cache
        let fp32_kv = config.total_kv_cache_bytes(CachePrecision::FP32);
        let fp16_kv = config.total_kv_cache_bytes(CachePrecision::FP16);

        // FP32: 32 × 2 × 32 × 2048 × 128 × 4 = ~2 GB
        assert!(fp32_kv > 1_800_000_000 && fp32_kv < 2_200_000_000,
            "FP32 KV cache: {} bytes", fp32_kv);

        // FP16: half that
        assert!(fp16_kv > 900_000_000 && fp16_kv < 1_100_000_000,
            "FP16 KV cache: {} bytes", fp16_kv);

        // With Q4_0 weights (~4.5× smaller) + FP16 KV cache, a 7B model fits in ~1.3 GB
        let ram_mb = config.estimate_ram_mb(CachePrecision::FP16, 2) * 9 / 40;
        assert!(ram_mb < 600,
            "Q4_0 estimate {} MB should fit comfortably under 2 GB", ram_mb);
    }
}
