//! Models.
//!
//! A model is a **trait**: it loads its weights and composes its architecture in
//! `forward`. The forward *is* the architecture (chat-mlx style) — built by
//! composing kernel building blocks, not by branching a generic loop on config
//! flags. Each model is its own composition; shared structure is shared through
//! the reusable components in `layers` (Linear, RmsNorm, …), not through a
//! config struct.
//!
//! Intermediates are local buffers, so they can be tapped mid-`forward` via a
//! [`Hook`] — e.g. sampling from a mid-layer hidden state for a meta-ML model.

use crate::context::GpuContext;

mod layers;
pub mod qwen3;

pub use layers::{Linear, RmsNorm};

/// A loaded, runnable model. Concrete models (`qwen3::Qwen3`, …) implement it.
pub trait Model: Sized {
    /// Load weights onto a fresh instance from a weight source.
    fn load(ctx: &GpuContext, w: &mut dyn Weights) -> Result<Self, String>;

    /// Forward one token: `x` is the input embedding (length = model dim) at
    /// position `pos`; returns the final hidden state. Composes the architecture
    /// from kernel building blocks and taps named intermediates through `hook`.
    fn forward(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        pos: usize,
        hook: &mut dyn Hook,
    ) -> wgpu::Buffer;
}

/// Where a model pulls its weights + metadata from — a GGUF file in production,
/// or anything else (random, for tests). The model asks for exactly the tensors
/// it needs, by name.
pub trait Weights {
    fn meta_u32(&self, key: &str) -> u32;
    fn meta_f32(&self, key: &str) -> f32;
    /// Is a tensor present? (e.g. to detect a bias or a tied lm-head.)
    fn has(&self, name: &str) -> bool;
    /// A weight matrix as `[in_f, out_f]` — the matmul B operand. (The loader
    /// transposes GGUF's `[out, in]` and dequantizes.)
    fn matrix(&mut self, ctx: &GpuContext, name: &str, in_f: usize, out_f: usize) -> wgpu::Buffer;
    /// A weight vector of length `len` (norm gains, biases).
    fn vector(&mut self, ctx: &GpuContext, name: &str, len: usize) -> wgpu::Buffer;
}

/// A tap on the forward pass. `forward` calls `tap` at each named intermediate,
/// handing over the GPU buffer (cheaply cloneable) so a caller can read it later
/// or feed it elsewhere — without the model knowing what for.
pub trait Hook {
    fn tap(&mut self, name: &str, layer: usize, buf: &wgpu::Buffer, len: usize);
}

/// No-op hook.
impl Hook for () {
    fn tap(&mut self, _: &str, _: usize, _: &wgpu::Buffer, _: usize) {}
}
