//! chat-wgpu — a clean-room WebGPU LLM inference engine.
//!
//! Hand-rolled WGSL kernel building blocks (`kernels`), composed by per-model
//! `forward` functions (`model`). A model is a trait that loads its weights and
//! composes its architecture — no framework, no candle. Targets the browser
//! (WebGPU via wasm) and runs natively (Metal / Vulkan / DX12) for development
//! and kernel verification.

pub mod context;
pub mod kernels;
pub mod loader;
pub mod model;
pub mod profile;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
