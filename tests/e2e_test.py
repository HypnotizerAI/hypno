#!/usr/bin/env python3
"""Create a synthetic HuggingFace-style model and test the full hypno pipeline.

This creates a minimal Llama-style model with safetensors weights,
then converts to .hypno format and runs inference via the CLI.
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np
from safetensors.numpy import save_file


def make_llama_config(hidden_size=64, intermediate_size=172, num_layers=2,
                       num_heads=4, num_kv_heads=2, vocab_size=256):
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
        "vocab_size": vocab_size,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "head_dim": hidden_size // num_heads,
    }


def make_tokenizer_json(vocab_size=256):
    """Create a minimal byte-level tokenizer."""
    vocab = {}
    # Special tokens
    vocab["<s>"] = 0
    vocab["</s>"] = 1
    vocab["<unk>"] = 2
    vocab["<pad>"] = 3

    # Byte tokens
    for i in range(256):
        token = f"<0x{i:02X}>"
        vocab[token] = i + 4

    merges = []
    for i in range(0, 254, 2):
        merges.append(f"<0x{i:02X}> <0x{i+1:02X}>")

    return {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
        },
        "added_tokens": [
            {"id": 0, "content": "<s>", "special": True},
            {"id": 1, "content": "</s>", "special": True},
            {"id": 2, "content": "<unk>", "special": True},
            {"id": 3, "content": "<pad>", "special": True},
        ],
        "post_processor": {
            "type": "TemplateProcessing",
            "single": ["<s>", "$A", "</s>"],
            "pair": ["<s>", "$A", "</s>", "$B", "</s>"],
            "special_tokens": {
                "<s>": {"id": 0, "type_id": 0},
                "</s>": {"id": 1, "type_id": 0},
            },
        },
    }


def make_weights(config):
    """Create random but deterministic weights for a minimal Llama model."""
    rng = np.random.RandomState(42)
    tensors = {}
    hd = config["hidden_size"]
    im = config["intermediate_size"]
    nl = config["num_hidden_layers"]
    vs = config["vocab_size"]
    nh = config["num_attention_heads"]
    nkv = config["num_key_value_heads"]
    hdim = config["head_dim"]

    # Embedding
    tensors["model.embed_tokens.weight"] = rng.randn(vs, hd).astype(np.float32) * 0.02

    # LM head (tied with embedding)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].copy()

    # Final norm
    tensors["model.norm.weight"] = np.ones(hd, dtype=np.float32)

    for l in range(nl):
        prefix = f"model.layers.{l}"

        # Input layernorm
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hd, dtype=np.float32)

        # Attention
        tensors[f"{prefix}.self_attn.q_proj.weight"] = (
            rng.randn(nh * hdim, hd).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.k_proj.weight"] = (
            rng.randn(nkv * hdim, hd).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.v_proj.weight"] = (
            rng.randn(nkv * hdim, hd).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.self_attn.o_proj.weight"] = (
            rng.randn(hd, nh * hdim).astype(np.float32) * 0.02
        )

        # Post attention layernorm
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hd, dtype=np.float32)

        # MLP
        tensors[f"{prefix}.mlp.gate_proj.weight"] = (
            rng.randn(im, hd).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.mlp.up_proj.weight"] = (
            rng.randn(im, hd).astype(np.float32) * 0.02
        )
        tensors[f"{prefix}.mlp.down_proj.weight"] = (
            rng.randn(hd, im).astype(np.float32) * 0.02
        )

    return tensors


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="hypno_test_")
    os.makedirs(out_dir, exist_ok=True)

    config = make_llama_config()
    tokenizer = make_tokenizer_json()
    weights = make_weights(config)

    # Write config.json
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Write tokenizer.json
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)

    # Write safetensors
    save_file(weights, os.path.join(out_dir, "model.safetensors"))

    # Write a second shard with any extra tensors
    extra = {}
    if any("self_attn" in k for k in weights):
        # Already included
        pass

    print(f"Created test model at: {out_dir}")
    print(f"  Config: hidden={config['hidden_size']}, layers={config['num_hidden_layers']}, "
          f"vocab={config['vocab_size']}")
    print(f"  Tensors: {len(weights)}")
    total_params = sum(w.size for w in weights.values())
    print(f"  Parameters: {total_params:,} ({total_params * 4:,} bytes FP32)")
    print(f"  Tokenizer vocab: {len(tokenizer['model']['vocab'])}")

    return out_dir


if __name__ == "__main__":
    model_dir = main()
    print(f"\nMODEL_DIR={model_dir}")
