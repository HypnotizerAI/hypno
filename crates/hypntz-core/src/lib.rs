//! Hypnotizer core types and binary format definitions.
//!
//! The `.hypno` binary format layout:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │  Header (16 bytes)                  │
//! │  ├─ magic:     [u8; 4] = b"HYPN"   │
//! │  ├─ version:   u32_le = 1          │
//! │  ├─ meta_kv_count: u32_le          │
//! │  └─ tensor_count:  u32_le          │
//! ├─────────────────────────────────────┤
//! │  Metadata KVs (variable)            │
//! │  For each KV:                       │
//! │  ├─ key_len:   u32_le              │
//! │  ├─ key:       [u8; key_len]       │
//! │  ├─ value_len: u32_le              │
//! │  └─ value:     [u8; value_len]     │
//! ├─────────────────────────────────────┤
//! │  Tensor Metadata Table (variable)   │
//! │  For each tensor:                   │
//! │  ├─ name_len:   u32_le             │
//! │  ├─ name:       [u8; name_len]     │
//! │  ├─ ndim:       u32_le             │
//! │  ├─ shape:      [u64_le; ndim]     │
//! │  ├─ dtype:      u32_le             │
//! │  ├─ offset:     u64_le             │
//! │  └─ data_len:   u64_le             │
//! ├─────────────────────────────────────┤
//! │  Data Payload (64-byte aligned)     │
//! │  [padding to 64-byte boundary]      │
//! │  tensor_0: [u8; data_len_0]        │
//! │  [padding to 64-byte boundary]      │
//! │  tensor_1: [u8; data_len_1]        │
//! │  ...                               │
//! └─────────────────────────────────────┘
//! ```

pub mod format;
pub mod quantization;
pub mod dtype;

pub use format::*;
pub use quantization::*;
pub use dtype::*;
