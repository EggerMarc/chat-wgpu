//! WGSL compute kernels — **GPU only**, organized as discrete functional
//! building blocks by category. Each kernel does one thing
//! (`activation::swiglu`, `norm::rmsnorm_unit`, …); families assemble their
//! architecture from these blocks (see `crate::arch` + `crate::families`), and
//! a fusion pass collapses common sequences into fused kernels. Nothing
//! branches on a runtime "family config" flag.
//!
//! Verified with GPU known-answer tests (`src/main.rs`): structural inputs whose
//! outputs are closed-form. No CPU reimplementation of any kernel.
//!
//!   matmul                 ✅
//!   norm::rmsnorm          ✅   norm::rmsnorm_unit (Gemma)  ✅
//!   activation::silu/gelu/tanh   ✅
//!   activation::swiglu/geglu     ✅
//!   rope                   ✅
//!   softmax / attention    todo
//!   fused: norm+matmul, gate+up+act, q4 dequant-matmul   todo (the perf payoff)
//!
//! Convention: buffers bind at sequential `@binding` 0.., a uniform last.

pub mod activation;
pub mod attention;
pub mod matmul;
pub mod mlp_mega;
pub mod norm;
pub mod q4;
pub mod rope;

pub use matmul::matmul;
pub use rope::rope;
