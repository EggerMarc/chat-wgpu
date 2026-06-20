//! Qwen3.
//!
//! The `forward` is the architecture — read it top to bottom and you see Qwen3:
//! pre-norm, q/k/v projections, **per-head QK-Norm**, RoPE, attention, output
//! projection, residual; then pre-norm, SwiGLU MLP, residual. No config flags —
//! Qwen3 simply *is* this composition. Llama is the same minus the QK-Norm
//! lines; Gemma swaps the norm/activation building blocks and adds post-norms.
//! Each is its own `Model` impl over the shared components and kernels.

use super::layers::{Linear, RmsNorm};
use super::{Hook, KvCache, Model, Weights};
use crate::context::GpuContext;
use crate::kernels::{activation, attention, rope};

struct Layer {
    attn_norm: RmsNorm,
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: RmsNorm, // per-head QK-Norm — the Qwen3 distinctive
    k_norm: RmsNorm,
    ffn_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct Qwen3 {
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    /// Input embedding table, raw `[vocab, dim]` (row-major) for row gather.
    embed: wgpu::Buffer,
    /// LM head `[dim, vocab]` (the matmul operand). Tied to `embed` if the GGUF
    /// ships no `output.weight`.
    lm_head: Linear,
    vocab: usize,
    dim: usize,
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl Model for Qwen3 {
    fn load(ctx: &GpuContext, w: &mut dyn Weights) -> Result<Self, String> {
        let n_layers = w.meta_u32("qwen3.block_count") as usize;
        let dim = w.meta_u32("qwen3.embedding_length") as usize;
        let n_heads = w.meta_u32("qwen3.attention.head_count") as usize;
        let n_kv_heads = w.meta_u32("qwen3.attention.head_count_kv") as usize;
        let head_dim = w.meta_u32("qwen3.attention.key_length") as usize;
        let hidden = w.meta_u32("qwen3.feed_forward_length") as usize;
        let eps = w.meta_f32("qwen3.attention.layer_norm_rms_epsilon");
        let rope_theta = w.meta_f32("qwen3.rope.freq_base");
        let vocab = w.meta_u32("qwen3.vocab_size") as usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let layers = (0..n_layers)
            .map(|i| {
                let p = format!("blk.{i}");
                Layer {
                    attn_norm: RmsNorm::load(ctx, w, &format!("{p}.attn_norm"), dim, eps, false),
                    wq: Linear::load(ctx, w, &format!("{p}.attn_q"), dim, q_dim),
                    wk: Linear::load(ctx, w, &format!("{p}.attn_k"), dim, kv_dim),
                    wv: Linear::load(ctx, w, &format!("{p}.attn_v"), dim, kv_dim),
                    wo: Linear::load(ctx, w, &format!("{p}.attn_output"), q_dim, dim),
                    q_norm: RmsNorm::load(ctx, w, &format!("{p}.attn_q_norm"), head_dim, eps, false),
                    k_norm: RmsNorm::load(ctx, w, &format!("{p}.attn_k_norm"), head_dim, eps, false),
                    ffn_norm: RmsNorm::load(ctx, w, &format!("{p}.ffn_norm"), dim, eps, false),
                    gate: Linear::load(ctx, w, &format!("{p}.ffn_gate"), dim, hidden),
                    up: Linear::load(ctx, w, &format!("{p}.ffn_up"), dim, hidden),
                    down: Linear::load(ctx, w, &format!("{p}.ffn_down"), hidden, dim),
                }
            })
            .collect();

        // Embedding table raw `[vocab, dim]` for row gather; LM head transposed
        // `[dim, vocab]`, tied to the embedding when `output.weight` is absent.
        let embed = w.vector(ctx, "token_embd.weight", vocab * dim);
        let lm_head = if w.has("output.weight") {
            Linear::load(ctx, w, "output", dim, vocab)
        } else {
            Linear::load(ctx, w, "token_embd", dim, vocab)
        };

        Ok(Self {
            layers,
            final_norm: RmsNorm::load(ctx, w, "output_norm", dim, eps, false),
            embed,
            lm_head,
            vocab,
            dim,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
        })
    }

    fn new_cache(&self, ctx: &GpuContext, max_seq: usize) -> KvCache {
        KvCache::new(ctx, self.layers.len(), self.n_kv_heads * self.head_dim, max_seq)
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn embed(&self, ctx: &GpuContext, token: u32) -> wgpu::Buffer {
        // Gather row `token` of the `[vocab, dim]` table.
        let out = ctx.empty(self.dim);
        ctx.copy(&self.embed, token as usize * self.dim, &out, 0, self.dim);
        out
    }

    fn logits(&self, ctx: &GpuContext, hidden: &wgpu::Buffer) -> wgpu::Buffer {
        self.lm_head.forward(ctx, hidden, 1)
    }

    fn forward(
        &self,
        ctx: &GpuContext,
        x: &wgpu::Buffer,
        pos: usize,
        cache: &mut KvCache,
        hook: &mut dyn Hook,
    ) -> wgpu::Buffer {
        let mut hidden = clone_buf(ctx, x, self.dim);
        for (i, l) in self.layers.iter().enumerate() {
            // ── attention sublayer ──
            let normed = l.attn_norm.forward(ctx, &hidden, 1);
            let q = l.wq.forward(ctx, &normed, 1);
            let k = l.wk.forward(ctx, &normed, 1);
            let v = l.wv.forward(ctx, &normed, 1);

            // per-head QK-Norm (each head's head_dim vector), then RoPE
            let q = l.q_norm.forward(ctx, &q, self.n_heads);
            let k = l.k_norm.forward(ctx, &k, self.n_kv_heads);
            let q = rope::rope(ctx, &q, self.n_heads, self.head_dim, pos, self.rope_theta);
            let k = rope::rope(ctx, &k, self.n_kv_heads, self.head_dim, pos, self.rope_theta);
            hook.tap("q", i, &q, self.n_heads * self.head_dim);

            // append this token's K/V to the cache, attend over the prefix
            cache.write(ctx, i, pos, &k, &v);
            let attn = attention::attention(
                ctx,
                &q,
                cache.k(i),
                cache.v(i),
                self.n_heads,
                self.n_kv_heads,
                pos + 1,
                self.head_dim,
            );
            let attn_out = l.wo.forward(ctx, &attn, 1);
            hidden = attention::add(ctx, &hidden, &attn_out, self.dim);
            hook.tap("post_attn", i, &hidden, self.dim);

            // ── MLP sublayer (SwiGLU) ──
            let normed = l.ffn_norm.forward(ctx, &hidden, 1);
            let gate = l.gate.forward(ctx, &normed, 1);
            let up = l.up.forward(ctx, &normed, 1);
            let act = activation::swiglu(ctx, &gate, &up, self.hidden);
            let ff = l.down.forward(ctx, &act, 1);
            hidden = attention::add(ctx, &hidden, &ff, self.dim);
            hook.tap("layer_out", i, &hidden, self.dim);
        }
        self.final_norm.forward(ctx, &hidden, 1)
    }
}

/// Copy a buffer (the input embedding) so `forward` owns a mutable hidden state.
fn clone_buf(ctx: &GpuContext, src: &wgpu::Buffer, len: usize) -> wgpu::Buffer {
    // residual add with a zero buffer = a copy, using the existing kernel.
    let zero = ctx.storage(&vec![0.0f32; len]);
    attention::add(ctx, src, &zero, len)
}
