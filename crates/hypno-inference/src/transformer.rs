//! Transformer layer implementation for Hypno.
//!
//! Implements a single decoder-only transformer layer (like Llama/GPT)
//! that reads weights directly from memory-mapped `.hypno` memory.
//!
//! Architecture:
//! ```text
//! x → RMSNorm → Q,K,V Projections → RoPE → Attention → Output Proj → Residual
//!   → RMSNorm → FFN Gate+Up → SiLU → Down Proj → Residual
//! ```

use hypno_core::DType;
use hypno_loader::HypnoModel;
use crate::ops;

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
    /// Extract configuration from `.hypno` metadata.
    pub fn from_model(model: &HypnoModel) -> Option<Self> {
        let get = |key: &str| -> Option<usize> {
            model.get_metadata(key)?.parse().ok()
        };
        let get_f32 = |key: &str| -> Option<f32> {
            model.get_metadata(key)?.parse().ok()
        };

        let hidden_size = get("hidden_size")?;
        let num_attention_heads = get("num_attention_heads")?;
        let num_key_value_heads = get("num_key_value_heads")
            .unwrap_or(num_attention_heads);
        let head_dim = hidden_size / num_attention_heads;

        Some(Self {
            hidden_size,
            intermediate_size: get("intermediate_size").unwrap_or(hidden_size * 8 / 3),
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers: get("num_hidden_layers").unwrap_or(0),
            vocab_size: get("vocab_size").unwrap_or(32000),
            max_position_embeddings: get("max_position_embeddings")
                .unwrap_or(2048),
            rms_norm_eps: get_f32("rms_norm_eps").unwrap_or(1e-5),
            rope_theta: get_f32("rope_theta").unwrap_or(10000.0),
            head_dim,
        })
    }
}

/// Buffers used during a forward pass (pre-allocated to avoid allocations).
#[derive(Clone)]
pub struct ForwardBuffers {
    pub hidden: Vec<f32>,       // [hidden_size]
    pub residual: Vec<f32>,     // [hidden_size]
    pub q: Vec<f32>,            // [num_heads * head_dim]
    pub k: Vec<f32>,            // [num_kv_heads * head_dim]
    pub v: Vec<f32>,            // [num_kv_heads * head_dim]
    pub attn_output: Vec<f32>,  // [hidden_size]
    pub ffn_gate: Vec<f32>,     // [intermediate_size]
    pub ffn_up: Vec<f32>,       // [intermediate_size]
    pub ffn_down: Vec<f32>,     // [hidden_size]
    /// KV cache for attention: (layer, head, position, dim)
    pub kv_cache_keys: Vec<Vec<Vec<f32>>>,   // [layers][kv_heads][pos * head_dim]
    pub kv_cache_values: Vec<Vec<Vec<f32>>>,
    pub cache_pos: usize,
}

impl ForwardBuffers {
    pub fn new(config: &HypnoConfig) -> Self {
        let n_kv_heads = config.num_key_value_heads;
        let n_heads = config.num_attention_heads;

        // Initialize empty KV caches
        let kv_cache_keys: Vec<Vec<Vec<f32>>> = (0..config.num_hidden_layers)
            .map(|_| (0..n_kv_heads).map(|_| Vec::new()).collect())
            .collect();
        let kv_cache_values = kv_cache_keys.clone();

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
            kv_cache_keys,
            kv_cache_values,
            cache_pos: 0,
        }
    }

    /// Reset KV caches for a new sequence.
    pub fn reset_cache(&mut self) {
        for layer_keys in &mut self.kv_cache_keys {
            for head_keys in layer_keys {
                head_keys.clear();
            }
        }
        for layer_vals in &mut self.kv_cache_values {
            for head_vals in layer_vals {
                head_vals.clear();
            }
        }
        self.cache_pos = 0;
    }
}

/// Gets tensor data from the model, returning (bytes, dtype).
fn get_weight<'a>(model: &'a HypnoModel, name: &str) -> Option<(&'a [u8], DType)> {
    model.get_tensor_data(name)
}

/// Forward pass through a single transformer layer.
/// Uses `buffers.hidden` as input and output for the hidden state.
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

    // --- Attention Block ---

    // 1. RMSNorm (input_layernorm)
    let norm_weight = get_weight(model, &format!("{}input_layernorm.weight", prefix))
        .map(|(d, _)| bytemuck::cast_slice::<u8, f32>(d).to_vec())
        .unwrap_or_else(|| vec![1.0f32; hd]);
    ops::rms_norm_in_place(&mut buffers.hidden, &norm_weight, config.rms_norm_eps);

    // Save residual
    buffers.residual.copy_from_slice(&buffers.hidden);

    // 2. Q projection
    if let Some((qw, qdt)) = get_weight(model, &format!("{}self_attn.q_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.q, qw, qdt, &buffers.hidden, None, nh * hdim, hd);
    }

    // 3. K projection
    if let Some((kw, kdt)) = get_weight(model, &format!("{}self_attn.k_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.k, kw, kdt, &buffers.hidden, None, nkv * hdim, hd);
    }

    // 4. V projection
    if let Some((vw, vdt)) = get_weight(model, &format!("{}self_attn.v_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.v, vw, vdt, &buffers.hidden, None, nkv * hdim, hd);
    }

    // 5. RoPE: rotate each KV head exactly once, then rotate all Q heads
    let pos = if use_kv_cache { buffers.cache_pos } else { 0 };

    // First, rotate each KV head once (avoids double-rotation from GQA sharing)
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

    // Then rotate each Q head (RoPE on Q alone: pass a dummy K that won't be accessed)
    for h in 0..nh {
        let q_start = h * hdim;
        let kvh = (h * nkv / nh).min(nkv - 1);
        // Use a stack copy of already-rotated K as dummy — rope() rotates both args
        // but the second rotation on an already-rotated K is harmless since we don't
        // read it back after this call (cache has already captured the correct K)
        let mut dummy_k: [f32; 256] = [0.0f32; 256]; // max head_dim supported
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

    // 6. KV Cache: add current K,V to cache (position counter managed by caller)
    if use_kv_cache {
        let layer = layer_idx;
        // Append current K and V to the cache
        for h in 0..nkv {
            let k_start = h * hdim;
            buffers.kv_cache_keys[layer][h].extend_from_slice(&buffers.k[k_start..k_start + hdim]);
            buffers.kv_cache_values[layer][h].extend_from_slice(&buffers.v[k_start..k_start + hdim]);
        }
    }

    // 7. Attention: scaled dot-product attention over ALL cached positions
    let seq_len = if use_kv_cache { buffers.cache_pos } else { 1 };
    let scale = 1.0 / (hdim as f32).sqrt();

    buffers.attn_output.fill(0.0);
    let kv_per_head = nh / nkv; // Grouped Query Attention

    for qh in 0..nh {
        let kvh = qh / kv_per_head;
        let q_start = qh * hdim;
        let o_start = qh * hdim;

        let cache_k = if use_kv_cache {
            &buffers.kv_cache_keys[layer_idx][kvh]
        } else {
            // Dummy reference — won't be used
            &buffers.kv_cache_keys[layer_idx][0]
        };

        // Compute attention scores over all cached positions
        let mut scores = vec![0.0f32; seq_len];
        for p in 0..seq_len {
            let k_offset = p * hdim;
            let mut dot = 0.0f32;

            if use_kv_cache {
                // Use KV cache for all positions
                for d in 0..hdim {
                    dot += buffers.q[q_start + d] * cache_k[k_offset + d];
                }
            } else {
                // No cache: single position, use current K
                let k_start = kvh * hdim;
                for d in 0..hdim {
                    dot += buffers.q[q_start + d] * buffers.k[k_start + d];
                }
            }
            scores[p] = dot * scale;
        }

        // Softmax
        ops::softmax_in_place(&mut scores);

        // Weighted sum of values from cache
        let cache_v = if use_kv_cache {
            &buffers.kv_cache_values[layer_idx][kvh]
        } else {
            &buffers.kv_cache_values[layer_idx][0]
        };
        for p in 0..seq_len {
            let attn_w = scores[p];
            if use_kv_cache {
                let v_offset = p * hdim;
                for d in 0..hdim {
                    buffers.attn_output[o_start + d] += attn_w * cache_v[v_offset + d];
                }
            } else {
                let v_start = kvh * hdim;
                for d in 0..hdim {
                    buffers.attn_output[o_start + d] += attn_w * buffers.v[v_start + d];
                }
            }
        }
    }

    // 8. Output projection
    if let Some((ow, odt)) = get_weight(model, &format!("{}self_attn.o_proj.weight", prefix)) {
        let mut tmp = vec![0.0f32; hd];
        tmp.copy_from_slice(&buffers.attn_output);
        ops::matmul_vec_auto(&mut buffers.hidden, ow, odt, &tmp, None, hd, hd);
    } else {
        buffers.hidden.copy_from_slice(&buffers.attn_output);
    }

    // Residual connection
    for i in 0..hd {
        buffers.hidden[i] += buffers.residual[i];
    }

    // --- FFN Block ---

    // Save residual
    buffers.residual.copy_from_slice(&buffers.hidden);

    // 9. RMSNorm (post_attention_layernorm)
    let post_norm = get_weight(model, &format!("{}post_attention_layernorm.weight", prefix))
        .map(|(d, _)| bytemuck::cast_slice::<u8, f32>(d).to_vec())
        .unwrap_or_else(|| vec![1.0f32; hd]);
    ops::rms_norm_in_place(&mut buffers.hidden, &post_norm, config.rms_norm_eps);

    // 10. FFN Gate projection
    if let Some((gw, gdt)) = get_weight(model, &format!("{}mlp.gate_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_gate, gw, gdt, &buffers.hidden, None, im, hd);
    }

    // 11. FFN Up projection
    if let Some((uw, udt)) = get_weight(model, &format!("{}mlp.up_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_up, uw, udt, &buffers.hidden, None, im, hd);
    }

    // 12. SiLU activation on gate
    ops::silu_in_place(&mut buffers.ffn_gate);

    // 13. Element-wise multiply gate * up
    for i in 0..im {
        buffers.ffn_gate[i] *= buffers.ffn_up[i];
    }

    // 14. Down projection
    if let Some((dw, ddt)) = get_weight(model, &format!("{}mlp.down_proj.weight", prefix)) {
        ops::matmul_vec_auto(&mut buffers.ffn_down, dw, ddt, &buffers.ffn_gate, None, hd, im);
    } else {
        buffers.ffn_down.copy_from_slice(&buffers.ffn_gate[..hd]);
    }

    // Residual connection
    for i in 0..hd {
        buffers.hidden[i] = buffers.residual[i] + buffers.ffn_down[i];
    }
}

/// Full model forward pass: input token IDs → output logits.
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
            _ => vec![0.0f32; hd], // Quantized embeddings not supported yet
        };
        buffers.hidden.copy_from_slice(&emb);
    } else {
        buffers.hidden.fill(0.0);
    }

    // 2. Through all transformer layers
    for layer_idx in 0..config.num_hidden_layers {
        transformer_layer_forward(model, config, layer_idx, buffers, use_kv_cache);
    }

    // Advance position counter once per token (after all layers cached)
    if use_kv_cache {
        buffers.cache_pos += 1;
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
        // Tie weights: use embedding as LM head
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
    // Note: full transformer tests require a real .hypno model file.
    // These tests verify the buffers and config parsing logic.

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
        assert_eq!(buffers.v.len(), 2 * 16);
        assert_eq!(buffers.ffn_gate.len(), 172);
        assert_eq!(buffers.kv_cache_keys.len(), 2);
    }
}
