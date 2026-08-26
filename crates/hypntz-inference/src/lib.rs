//! Hypnotizer inference engine.
//!
//! Provides tensor operators (MatMul, RMSNorm, RoPE, Softmax, SiLU) and
//! a Transformer layer implementation that reads weights directly from
//! memory-mapped `.hypno` files.

pub mod ops;
pub mod transformer;
pub mod kernels;

pub use ops::*;
pub use transformer::*;
pub use kernels::cpu_features;
