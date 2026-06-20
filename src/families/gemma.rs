//! Gemma 2/3 — exercises the variant kernels and a different layer shape:
//! unit-shift RMSNorm (`1 + weight`), GeGLU, sliding-window attention, and
//! *post*-sublayer norms (a second RMSNorm on each sublayer output before the
//! residual add) in addition to the pre-norms. Final-logit soft-capping is a
//! sampler-side scalar in `params`.

use super::{Family, Params};
use crate::arch::{Block::*, Mask::*, Proj::*};

pub fn family() -> Family {
    Family {
        name: "gemma",
        params: Params {
            eps: 1e-6,
            rope_theta: 10_000.0,
            sliding_window: Some(4096),
            final_logit_softcap: Some(30.0),
        },
        layer: vec![
            // attention: pre-norm + sublayer + post-norm, then residual
            ResidualSave,
            RmsNormUnit,
            Linear(Q),
            Linear(K),
            Linear(V),
            Rope,
            Attention(Sliding(4096)),
            Linear(O),
            RmsNormUnit, // <- Gemma post-norm
            ResidualAdd,
            // MLP: pre-norm + GeGLU sublayer + post-norm, then residual
            ResidualSave,
            RmsNormUnit,
            Linear(Gate),
            Linear(Up),
            GeGlu,
            Linear(Down),
            RmsNormUnit, // <- Gemma post-norm
            ResidualAdd,
        ],
    }
}
