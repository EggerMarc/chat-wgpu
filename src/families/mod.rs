//! Family implementations — each one *plays out* its architecture as a list of
//! [`crate::arch::Block`] building blocks, next to the kernels. Not a shared
//! config-branched forward: the structure (which ops, in what order, which
//! norm/activation/masking variant) lives in the block list, so the fusion pass
//! can collapse it and the executor can just walk it.
//!
//! What's a *weight* detail vs a *structural* block: QKV bias (Qwen2) is a bias
//! tensor attached to a `Linear`, applied at execution if present — not a
//! separate block. QK-Norm (Qwen3), the RMSNorm gain (Gemma), the MLP
//! activation, post-sublayer norms (Gemma2), and sliding-window masking *are*
//! structural and show up as distinct blocks.

use crate::arch::Block;

mod gemma;
mod llama;
mod qwen2;
mod qwen3;

/// Scalars used while executing the blocks (not part of the structure).
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub eps: f32,
    pub rope_theta: f32,
    pub sliding_window: Option<u32>,
    pub final_logit_softcap: Option<f32>,
}

/// A resolved family: its name, its scalar params, and one decoder layer
/// assembled from building blocks. (Embedding + final norm + lm-head wrap the
/// `n_layers` repetition of `layer`; added with the forward loop.)
#[derive(Clone, Debug)]
pub struct Family {
    pub name: &'static str,
    pub params: Params,
    pub layer: Vec<Block>,
}

impl Family {
    /// Resolve from a GGUF `general.architecture` string. Unknown → Llama
    /// baseline (covers Mistral, Phi-3, and most of the ecosystem; metadata +
    /// tensor probing fill in weight-level details).
    pub fn from_arch(arch: &str) -> Family {
        match arch {
            "qwen3" => qwen3::family(),
            "qwen2" => qwen2::family(),
            "gemma" | "gemma2" | "gemma3" => gemma::family(),
            _ => llama::family(),
        }
    }
}
