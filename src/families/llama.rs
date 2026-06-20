//! Llama-shaped baseline — also the fallback for Mistral / Phi-3 and unknown
//! archs. Plain RMSNorm, SwiGLU, GQA, RoPE; no QK-Norm. Pre-norm only.

use super::{Family, Params};
use crate::arch::{Block::*, Mask::*, Proj::*};

pub fn family() -> Family {
    Family {
        name: "llama",
        params: Params {
            eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: None,
            final_logit_softcap: None,
        },
        layer: vec![
            // attention sublayer
            ResidualSave,
            RmsNorm,
            Linear(Q),
            Linear(K),
            Linear(V),
            Rope,
            Attention(Causal),
            Linear(O),
            ResidualAdd,
            // MLP sublayer
            ResidualSave,
            RmsNorm,
            Linear(Gate),
            Linear(Up),
            SwiGlu,
            Linear(Down),
            ResidualAdd,
        ],
    }
}
