//! Family implementations — built next to the kernels, in terms of the
//! family-agnostic ops.
//!
//! The WGSL kernels and their Rust wrappers (`crate::kernels`) are generic over
//! the points where transformer families actually diverge: the RMSNorm gain
//! (`NormKind`), the MLP activation (`Activation`), whether attention applies
//! QK-Norm or QKV bias, embedding tying, RoPE base, and any sliding-window
//! masking. A [`FamilySpec`] is just that set of choices; one constructor per
//! family declares them. The forward loop (todo) reads the spec and dispatches
//! the right kernel variant — no per-family kernel code.
//!
//! This mirrors chat-mlx/chat-candle's config-driven architecture, but the
//! "config" here is an explicit per-family spec sitting beside the kernels.

use crate::kernels::{Activation, NormKind};

mod gemma;
mod llama;
mod qwen2;
mod qwen3;

/// The architecture knobs a family pins. Everything a generic forward loop
/// needs to dispatch the right kernel variants. Defaults that a specific GGUF
/// overrides (rope base, tying) are marked.
#[derive(Clone, Copy, Debug)]
pub struct FamilySpec {
    pub name: &'static str,
    /// RMSNorm gain variant.
    pub norm: NormKind,
    /// MLP gate activation.
    pub activation: Activation,
    /// Per-head RMSNorm on q/k before attention (Qwen3).
    pub use_qk_norm: bool,
    /// Bias on q/k/v projections (Qwen2).
    pub attn_qkv_bias: bool,
    /// Bias on the attention output projection.
    pub attn_o_bias: bool,
    /// Default: tie lm-head to the input embedding. A shipped `output.weight`
    /// (or `lm_head.weight`) in the GGUF overrides this.
    pub tie_word_embeddings: bool,
    /// Default RoPE base; a GGUF `*.rope.freq_base` overrides it.
    pub default_rope_theta: f32,
    /// Sliding-window attention span, if the family masks beyond it (Mistral,
    /// Gemma). `None` = full causal.
    pub sliding_window: Option<u32>,
    /// Final-logit soft-cap (Gemma); `None` = no capping.
    pub final_logit_softcap: Option<f32>,
}

impl FamilySpec {
    /// Resolve a family from a GGUF `general.architecture` string. Unknown
    /// architectures fall back to the Llama-shaped baseline, which covers most
    /// of the ecosystem; metadata + tensor probing still fill in the rest.
    pub fn from_arch(arch: &str) -> FamilySpec {
        match arch {
            "qwen3" => qwen3::spec(),
            "qwen2" => qwen2::spec(),
            "gemma" | "gemma2" | "gemma3" => gemma::spec(),
            _ => llama::spec(), // "llama", "mistral", "phi3", … (Llama-shaped)
        }
    }
}
