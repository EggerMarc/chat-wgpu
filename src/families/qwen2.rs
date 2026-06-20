//! Qwen2 / Qwen2.5 — structurally identical to Llama; the difference is a bias
//! tensor on the q/k/v projections (a weight detail applied at execution, not a
//! block) and tied embeddings. Shares Llama's block assembly, its own params.

use super::{Family, Params};

pub fn family() -> Family {
    Family {
        name: "qwen2",
        params: Params {
            eps: 1e-6,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            final_logit_softcap: None,
        },
        layer: super::llama::family().layer,
    }
}
