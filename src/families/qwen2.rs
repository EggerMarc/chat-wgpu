//! Qwen2 / Qwen2.5 — Llama-shaped with QKV bias (the q/k/v projections carry a
//! bias; the output projection does not) and tied embeddings. No QK-Norm.

use super::FamilySpec;
use crate::kernels::{Activation, NormKind};

pub fn spec() -> FamilySpec {
    FamilySpec {
        name: "qwen2",
        norm: NormKind::Plain,
        activation: Activation::SwiGlu,
        use_qk_norm: false,
        attn_qkv_bias: true,
        attn_o_bias: false,
        tie_word_embeddings: true,
        default_rope_theta: 1_000_000.0,
        sliding_window: None,
        final_logit_softcap: None,
    }
}
