//! chat-wgpu — a clean-room WebGPU LLM inference engine.
//!
//! Hand-rolled WGSL kernels, GGUF weights, HF tokenizer; no candle, no other ML
//! framework. Targets the browser (WebGPU via wasm) and runs natively (Metal /
//! Vulkan / DX12 via wgpu) for development and kernel verification.
//!
//! Status: foundation. The GPU context + matmul kernel are in place and
//! verified vs CPU (`cargo run --bin verify --features verify`). The remaining
//! kernels (rmsnorm, rope, swiglu, attention, q4 dequant-matmul) and the Qwen3
//! forward loop + GGUF loader + wasm API are the build-out — see ROADMAP.md.

pub mod context;
pub mod kernels;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
