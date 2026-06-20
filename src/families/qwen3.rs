//! Qwen3 — Llama-shaped plus per-head QK-Norm (an RMSNorm on q and k after
//! their projections, before RoPE). No QKV bias; tied embeddings; large RoPE
//! base. The `QkNorm` block is the one structural addition over Llama.

use super::{Family, Params};
use crate::arch::{Block::*, Mask::*, Proj::*};

pub fn family() -> Family {
    Family {
        name: "qwen3",
        params: Params {
            eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            final_logit_softcap: None,
        },
        layer: vec![
            ResidualSave,
            RmsNorm,
            Linear(Q),
            Linear(K),
            Linear(V),
            QkNorm, // <- Qwen3 only
            Rope,
            Attention(Causal),
            Linear(O),
            ResidualAdd,
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
