//! Llama-shaped baseline — also the fallback for Mistral / Phi-3 and any
//! unknown architecture. Plain RMSNorm, SwiGLU, GQA, RoPE, no QK-Norm, no QKV
//! bias. (Mistral adds a sliding window; that's a metadata-driven override on
//! top of this, not a separate spec.)

use super::FamilySpec;
use crate::kernels::{Activation, NormKind};

pub fn spec() -> FamilySpec {
    FamilySpec {
        name: "llama",
        norm: NormKind::Plain,
        activation: Activation::SwiGlu,
        use_qk_norm: false,
        attn_qkv_bias: false,
        attn_o_bias: false,
        tie_word_embeddings: false,
        default_rope_theta: 10_000.0,
        sliding_window: None,
        final_logit_softcap: None,
    }
}
