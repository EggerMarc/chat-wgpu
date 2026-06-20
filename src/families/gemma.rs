//! Gemma 2/3 — the family that exercises the kernel variants: the unit-shift
//! RMSNorm gain (`1 + weight`) and GeGLU instead of SwiGLU. Also sliding-window
//! attention on alternating layers and final-logit soft-capping (handled in the
//! forward loop / sampler).

use super::FamilySpec;
use crate::kernels::{Activation, NormKind};

pub fn spec() -> FamilySpec {
    FamilySpec {
        name: "gemma",
        norm: NormKind::UnitShift,
        activation: Activation::GeGlu,
        use_qk_norm: false,
        attn_qkv_bias: false,
        attn_o_bias: false,
        tie_word_embeddings: true,
        default_rope_theta: 10_000.0,
        sliding_window: Some(4096),
        final_logit_softcap: Some(30.0),
    }
}
