//! Qwen3 — the first bring-up target. Llama-shaped, but adds per-head QK-Norm
//! (an RMSNorm on q and k before attention), no QKV bias, and ships tied
//! embeddings (no `output.weight`).

use super::FamilySpec;
use crate::kernels::{Activation, NormKind};

pub fn spec() -> FamilySpec {
    FamilySpec {
        name: "qwen3",
        norm: NormKind::Plain,
        activation: Activation::SwiGlu,
        use_qk_norm: true,
        attn_qkv_bias: false,
        attn_o_bias: false,
        tie_word_embeddings: true,
        default_rope_theta: 1_000_000.0,
        sliding_window: None,
        final_logit_softcap: None,
    }
}
