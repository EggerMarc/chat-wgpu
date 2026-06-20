//! The op vocabulary families assemble from, and the fusion pass over it.
//!
//! A family describes a decoder layer as an ordered list of [`Block`]s — each a
//! reference to a functional kernel building block. Because the architecture is
//! *data*, not a branchy forward function, a [`fuse`] pass can pattern-rewrite
//! adjacent blocks into fused kernels (norm folded into the next matmul; the
//! gate/up matmuls + activation collapsed into one register-blocked dispatch).
//! That fusion is where the browser-GPU performance comes from.
//!
//! The executor that runs a (fused) block list on the GPU with real weights is
//! the forward-loop milestone (see ROADMAP). Today the assembly + fusion run as
//! data and are reported by `verify`.

/// A weight projection (matmul against a named weight).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

/// Attention masking the family applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    Causal,
    Sliding(u32),
}

/// One step in a decoder layer. Primitive variants map 1:1 to a kernel building
/// block; the `Fused*` variants are produced by [`fuse`] and dispatch a single
/// fused kernel in place of the sequence they replace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Block {
    ResidualSave,
    ResidualAdd,
    RmsNorm,
    RmsNormUnit,
    Linear(Proj),
    QkNorm,
    Rope,
    Attention(Mask),
    SwiGlu,
    GeGlu,

    // --- fused (post-`fuse`) ---
    /// RMSNorm folded into the input of the following projection.
    FusedNormLinear { unit: bool, proj: Proj },
    /// gate-proj + up-proj matmuls + gate activation, one register-blocked
    /// kernel that dequantizes each weight once across all token rows.
    FusedGatedMlp { geglu: bool },
}

impl Block {
    /// One dispatch per block (a fused block is still one dispatch — that's the
    /// point). Used to report the fusion win.
    pub fn dispatches(self) -> usize {
        1
    }
}

/// Pattern-rewrite a block list into its fused form. Conservative and local:
/// each rule matches a short window and only fires on an exact match, so it is
/// always a behavior-preserving rewrite.
pub fn fuse(blocks: &[Block]) -> Vec<Block> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut i = 0;
    while i < blocks.len() {
        // [RmsNorm{,Unit}, Linear(p)] -> FusedNormLinear
        if let (Some(&n), Some(&Block::Linear(proj))) = (blocks.get(i), blocks.get(i + 1)) {
            let unit = match n {
                Block::RmsNorm => Some(false),
                Block::RmsNormUnit => Some(true),
                _ => None,
            };
            if let Some(unit) = unit {
                out.push(Block::FusedNormLinear { unit, proj });
                i += 2;
                continue;
            }
        }
        // [Linear(Gate), Linear(Up), SwiGlu|GeGlu] -> FusedGatedMlp
        if let (Some(&Block::Linear(Proj::Gate)), Some(&Block::Linear(Proj::Up)), Some(&act)) =
            (blocks.get(i), blocks.get(i + 1), blocks.get(i + 2))
        {
            let geglu = match act {
                Block::SwiGlu => Some(false),
                Block::GeGlu => Some(true),
                _ => None,
            };
            if let Some(geglu) = geglu {
                out.push(Block::FusedGatedMlp { geglu });
                i += 3;
                continue;
            }
        }
        out.push(blocks[i]);
        i += 1;
    }
    out
}

/// Total dispatches for a block list (sum of per-block dispatches).
pub fn dispatch_count(blocks: &[Block]) -> usize {
    blocks.iter().map(|b| b.dispatches()).sum()
}
