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

mod cache;
mod layers;
pub mod qwen3;

pub use cache::KvCache;
pub use layers::{Linear, Proj, RmsNorm};

/// A loaded, runnable model. Concrete models (`qwen3::Qwen3`, …) implement it.
pub trait Model: Sized {
    /// Load weights onto a fresh instance. `quantize` keeps matmul weights
    /// q4-packed in VRAM (smaller, faster decode) instead of dense f32.
    fn load(ctx: &GpuContext, w: &mut dyn Weights, quantize: bool) -> Result<Self, String>;

    /// Allocate a KV cache sized for this model and a `max_seq` context.
    fn new_cache(&self, ctx: &GpuContext, max_seq: usize) -> KvCache;

    /// Vocabulary size (logits length).
    fn vocab_size(&self) -> usize;

    /// Input embedding for `token` (length = model dim).
    fn embed(&self, ctx: &GpuContext, token: u32) -> wgpu::Buffer;

    /// Forward one token at `pos`: `x` is the input embedding; the new K/V are
    /// written into `cache` and attention reads the `pos+1` prefix. Returns the
    /// final hidden state. Composes the architecture from kernel building blocks
    /// and taps named intermediates through `hook`.
    fn forward(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        pos: usize,
        cache: &mut KvCache,
        hook: &mut dyn Hook,
    ) -> wgpu::Buffer;

    /// Project a hidden state to vocabulary logits (`[vocab]`).
    fn logits(&self, ctx: &GpuContext, hidden: &wgpu::Buffer) -> wgpu::Buffer;
}

/// Greedy generation: prefill the prompt, then decode `max_new` tokens by
/// argmax. Reads logits back each step (a future on-device argmax avoids the
/// sync). Returns the generated token ids.
pub async fn generate<M: Model>(
    ctx: &GpuContext,
    model: &M,
    prompt: &[u32],
    max_new: usize,
    hook: &mut dyn Hook,
) -> Vec<u32> {
    ctx.clear_cache(); // fresh program: drop any cache from prior kernels
    let mut cache = model.new_cache(ctx, prompt.len() + max_new);
    let vocab = model.vocab_size();
    let total = prompt.len() + max_new;
    let mut produced = Vec::with_capacity(max_new);
    let mut next_input = prompt[0];

    // One unified step per token (prompt *and* generated): reset the frame →
    // embed → forward → lm-head → read. The op sequence is identical every step,
    // so the bind-group cache stays aligned across tokens. (During the prompt
    // the logits are just discarded.)
    for pos in 0..total {
        ctx.reset_frame();
        // PROFILE: split CPU-record (all the ctx.run calls) from GPU+sync (read).
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let x = model.embed(ctx, next_input);
        let hidden = model.forward(ctx, &x, pos, &mut cache, hook);
        let logits = model.logits(ctx, &hidden);
        #[cfg(not(target_arch = "wasm32"))]
        let t_rec = t0.elapsed().as_secs_f64() * 1000.0;
        #[cfg(not(target_arch = "wasm32"))]
        let t1 = std::time::Instant::now();
        let lv = ctx.read(&logits, vocab).await; // flush + GPU sync
        #[cfg(not(target_arch = "wasm32"))]
        if pos >= prompt.len() && pos < prompt.len() + 4 {
            eprintln!(
                "[profile] token {}: record(CPU)={:.1}ms  read(GPU+sync)={:.1}ms",
                pos - prompt.len(),
                t_rec,
                t1.elapsed().as_secs_f64() * 1000.0
            );
        }
        let tok = argmax(&lv);
        if pos + 1 < prompt.len() {
            next_input = prompt[pos + 1];
        } else {
            produced.push(tok);
            next_input = tok;
            if produced.len() >= max_new {
                break;
            }
        }
    }
    produced
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Where a model pulls its weights + metadata from — a GGUF file in production,
/// or anything else (random, for tests). The model asks for exactly the tensors
/// it needs, by name.
pub trait Weights {
    fn meta_u32(&self, key: &str) -> u32;
    fn meta_f32(&self, key: &str) -> f32;
    /// Is a tensor present? (e.g. to detect a bias or a tied lm-head.)
    fn has(&self, name: &str) -> bool;
    /// A weight matrix as `[in_f, out_f]` host data — the matmul B operand. (The
    /// loader transposes GGUF's `[out, in]` and dequantizes.) Host-side so it can
    /// be concatenated (fused QKV / gate-up) before upload.
    fn matrix_data(&mut self, name: &str, in_f: usize, out_f: usize) -> Vec<f32>;
    /// A weight matrix in its **native** `[out_f, in_f]` orientation (no
    /// transpose) — what the q4 row-quantizer blocks along.
    fn matrix_raw(&mut self, name: &str, out_f: usize, in_f: usize) -> Vec<f32>;
    /// A weight vector of length `len` host data (norm gains, biases).
    fn vector_data(&mut self, name: &str, len: usize) -> Vec<f32>;

    /// Upload a matrix to a GPU buffer.
    fn matrix(&mut self, ctx: &GpuContext, name: &str, in_f: usize, out_f: usize) -> wgpu::Buffer {
        ctx.storage(&self.matrix_data(name, in_f, out_f))
    }
    /// Upload a vector to a GPU buffer.
    fn vector(&mut self, ctx: &GpuContext, name: &str, len: usize) -> wgpu::Buffer {
        ctx.storage(&self.vector_data(name, len))
    }
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
