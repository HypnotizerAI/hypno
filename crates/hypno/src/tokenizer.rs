//! BPE / SentencePiece tokenizer for Hypno.
//!
//! Supports the HuggingFace `tokenizer.json` format. During conversion,
//! the vocabulary and merges are embedded in `.hypno` metadata and loaded
//! from there at runtime.
//!
//! Uses a trie-based approach for efficient byte-pair encoding.

use std::collections::HashMap;

/// A single entry in the tokenizer vocabulary.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub id: u32,
    pub token: String,
}

/// The full tokenizer model, loaded from either `tokenizer.json` or `.hypno` metadata.
#[derive(Debug, Clone)]
pub struct HypnoTokenizer {
    /// Map from token string → token id.
    vocab: HashMap<String, u32>,
    /// Map from token id → token string.
    id_to_token: Vec<String>,
    /// Byte-pair merge ranks: (token_a, token_b) → priority (lower = higher priority).
    merges: HashMap<(String, String), u32>,
    /// Special tokens.
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
}

impl HypnoTokenizer {
    /// Create a tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json(&data)
    }

    /// Create a tokenizer from the contents of `tokenizer.json`.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let root: serde_json::Value = serde_json::from_str(json)?;

        // Extract the model section
        let model = root.get("model").ok_or_else(|| anyhow::anyhow!("Missing 'model' in tokenizer.json"))?;

        // Extract vocab
        let vocab_obj = model.get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("Missing 'model.vocab'"))?;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut max_id = 0u32;
        for (token, id_val) in vocab_obj {
            let id = id_val.as_u64().ok_or_else(|| anyhow::anyhow!("Non-integer vocab id"))? as u32;
            vocab.insert(token.clone(), id);
            if id > max_id {
                max_id = id;
            }
        }

        let mut id_to_token = vec![String::new(); max_id as usize + 1];
        for (token, &id) in &vocab {
            id_to_token[id as usize] = token.clone();
        }

        // Extract merges
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        if let Some(merges_arr) = model.get("merges").and_then(|m| m.as_array()) {
            for (rank, merge_entry) in merges_arr.iter().enumerate() {
                if let Some(merge_str) = merge_entry.as_str() {
                    let parts: Vec<&str> = merge_str.split(' ').collect();
                    if parts.len() == 2 {
                        merges.insert(
                            (parts[0].to_string(), parts[1].to_string()),
                            rank as u32,
                        );
                    }
                }
            }
        }

        // Fix: if no merges array, try the "fuse_unk" or other fields
        // Some models use SentencePiece and don't have BPE merges
        // For those, we store merges as empty and rely on pre-tokenization

        // Extract special token IDs from added_tokens or special_tokens
        let added_tokens = root.get("added_tokens").and_then(|a| a.as_array());
        let mut bos_token_id: Option<u32> = None;
        let mut eos_token_id: Option<u32> = None;
        let mut pad_token_id: Option<u32> = None;
        let mut unk_token_id: Option<u32> = None;

        // Check post_processor or special tokens for BOS/EOS
        if let Some(pp) = root.get("post_processor") {
            if let Some(cls) = pp.get("cls_token") {
                if let Some(name) = cls.as_str() {
                    bos_token_id = vocab.get(name).copied();
                }
            }
            if let Some(sep) = pp.get("sep_token") {
                if let Some(name) = sep.as_str() {
                    eos_token_id = vocab.get(name).copied();
                }
            }
        }

        // Also check added_tokens for special tokens
        if let Some(tokens) = added_tokens {
            for token_entry in tokens {
                let special = token_entry.get("special").and_then(|s| s.as_bool()).unwrap_or(false);
                if special {
                    if let (Some(content), Some(id)) = (
                        token_entry.get("content").and_then(|c| c.as_str()),
                        token_entry.get("id").and_then(|i| i.as_u64()),
                    ) {
                        let tid = id as u32;
                        if content == "<s>" || content == "<bos>" {
                            bos_token_id = Some(tid);
                        }
                        if content == "</s>" || content == "<eos>" {
                            eos_token_id = Some(tid);
                        }
                        if content == "<pad>" {
                            pad_token_id = Some(tid);
                        }
                        if content == "<unk>" {
                            unk_token_id = Some(tid);
                        }
                    }
                }
            }
        }

        // Fallback: check raw vocab for common special tokens
        if bos_token_id.is_none() {
            bos_token_id = vocab.get("<s>").copied()
                .or_else(|| vocab.get("<bos>").copied());
        }
        if eos_token_id.is_none() {
            eos_token_id = vocab.get("</s>").copied()
                .or_else(|| vocab.get("<eos>").copied());
        }
        if pad_token_id.is_none() {
            pad_token_id = vocab.get("<pad>").copied();
        }
        if unk_token_id.is_none() {
            unk_token_id = vocab.get("<unk>").copied();
        }

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            unk_token_id,
        })
    }

    /// Create a tokenizer from metadata embedded in the `.hypno` file.
    /// The metadata should contain "tokenizer_json" with the full tokenizer.json content.
    pub fn from_hypno_metadata(metadata: &[crate::format::MetaKV]) -> anyhow::Result<Self> {
        let json = metadata.iter()
            .find(|kv| kv.key == "tokenizer_json")
            .map(|kv| kv.value.clone())
            .ok_or_else(|| anyhow::anyhow!("Missing 'tokenizer_json' in metadata"))?;
        Self::from_json(&json)
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            // Fallback: word-level tokenization using vocab
            return self.encode_word_level(text);
        }
        self.encode_bpe(text)
    }

    /// Word-level tokenization (used when no BPE merges are available).
    fn encode_word_level(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // Add BOS if available
        if let Some(bos) = self.bos_token_id {
            ids.push(bos);
        }

        // Split on whitespace and punctuation
        let words = tokenize_words(text);
        for word in words {
            // Try the full word first
            if let Some(&id) = self.vocab.get(&word) {
                ids.push(id);
            } else {
                // Try byte-level fallback
                let bytes = word.as_bytes();
                for &b in bytes {
                    let token = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.vocab.get(&token) {
                        ids.push(id);
                    } else if let Some(unk) = self.unk_token_id {
                        ids.push(unk);
                    }
                }
            }
        }

        ids
    }

    /// Full BPE encoding.
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        if let Some(bos) = self.bos_token_id {
            ids.push(bos);
        }

        // Split text into byte-level characters
        let mut symbols: Vec<String> = text.as_bytes().iter().map(|&b| {
            format!("<0x{:02X}>", b)
        }).collect();

        if symbols.is_empty() {
            return ids;
        }

        // Perform BPE merges
        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = 0usize;

            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(
                    symbols[i].clone(),
                    symbols[i + 1].clone(),
                )) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == u32::MAX {
                break; // No more merges possible
            }

            // Merge the pair
            let merged = format!("{}{}",
                symbols[best_idx],
                symbols[best_idx + 1]
            );
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        // Convert symbols to token IDs
        for sym in &symbols {
            if let Some(&id) = self.vocab.get(sym) {
                ids.push(id);
            } else if let Some(&id) = self.vocab.get(sym.as_str()) {
                ids.push(id);
            } else if let Some(unk) = self.unk_token_id {
                ids.push(unk);
            }
            // else: skip unknown symbols
        }

        ids
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in ids {
            if (id as usize) < self.id_to_token.len() {
                let token = &self.id_to_token[id as usize];
                // Handle byte-level tokens
                if token.starts_with("<0x") && token.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        result.push(byte as char);
                    }
                } else if token == "<s>" || token == "<bos>" || token == "</s>" || token == "<eos>" || token == "<unk>" || token == "<pad>" {
                    // Skip special tokens in output
                } else {
                    // Replace the HuggingFace Ġ character (space marker) with actual space
                    result.push_str(&token.replace('\u{0120}', " "));
                }
            }
        }
        result
    }

    pub fn bos_token_id(&self) -> Option<u32> { self.bos_token_id }
    pub fn eos_token_id(&self) -> Option<u32> { self.eos_token_id }
    pub fn pad_token_id(&self) -> Option<u32> { self.pad_token_id }
    pub fn vocab_size(&self) -> usize { self.id_to_token.len() }
}

/// Simple word tokenizer splitting on whitespace and punctuation boundaries.
fn tokenize_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            // Add space as its own token
            if let Some(&_id) = None::<&u32> {
                // Skip — spaces are handled differently
            }
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                words.push(current.clone());
                current.clear();
            }
            current.push(ch);
            words.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        words.push(current);
    }

    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_from_minimal_json() {
        // Minimal tokenizer.json with just vocab
        let json = r#"{
            "model": {
                "vocab": {
                    "<s>": 0,
                    "</s>": 1,
                    "<unk>": 2,
                    "<pad>": 3,
                    "hello": 4,
                    "world": 5,
                    "!": 6
                },
                "merges": []
            },
            "added_tokens": [
                {"id": 0, "content": "<s>", "special": true},
                {"id": 1, "content": "</s>", "special": true},
                {"id": 2, "content": "<unk>", "special": true},
                {"id": 3, "content": "<pad>", "special": true}
            ]
        }"#;

        let tok = HypnoTokenizer::from_json(json).unwrap();
        assert_eq!(tok.bos_token_id, Some(0));
        assert_eq!(tok.eos_token_id, Some(1));
        assert_eq!(tok.vocab_size(), 7);

        let ids = tok.encode("hello world !");
        assert!(ids.contains(&4));
        assert!(ids.contains(&5));
        assert!(ids.contains(&6));
    }

    #[test]
    fn test_decode() {
        let json = r#"{
            "model": {
                "vocab": {
                    "<s>": 0,
                    "</s>": 1,
                    "hello": 2,
                    "\u0120world": 3
                },
                "merges": []
            }
        }"#;

        let tok = HypnoTokenizer::from_json(json).unwrap();
        let text = tok.decode(&[2, 3]);
        assert_eq!(text, "hello world");
    }
}
