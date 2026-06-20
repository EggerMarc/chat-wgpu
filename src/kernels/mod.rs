//! WGSL compute kernels and their host-side dispatch wrappers — **GPU only**.
//!
//! Kernels are added in dependency order and checked with GPU known-answer
//! tests (`src/main.rs`): structural inputs whose outputs are closed-form
//! (identity → passthrough, ones → sum, rope@pos0 → identity). No CPU
//! reimplementation of any kernel lives here — this is a pure wgpu provider.
//!
//! **Family-agnostic:** where transformer families diverge, the kernel takes a
//! variant (`NormKind`, `Activation`, …) selected from a `families::FamilySpec`
//! rather than forking the shader. One shader, branched on a uniform variant
//! field; the family layer picks which branch.
//!
//!   matmul (f32)  ✅
//!   rmsnorm       ✅  (Plain | UnitShift gain)
//!   glu           ✅  (SwiGLU | GeGLU)
//!   rope          ✅
//!   softmax / attention   todo
//!   q4 dequant-matmul (the hot path)  todo
//!
//! Convention: buffers bind at sequential `@binding` 0.., a uniform last.

use crate::context::GpuContext;

const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
const RMSNORM_WGSL: &str = include_str!("rmsnorm.wgsl");
const GLU_WGSL: &str = include_str!("glu.wgsl");
const ROPE_WGSL: &str = include_str!("rope.wgsl");

/// RMSNorm gain variant — the per-family divergence in how the learned weight
/// is applied. Selected by the family spec, not the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `gain = weight` (Llama / Qwen).
    Plain,
    /// `gain = 1 + weight` (Gemma).
    UnitShift,
}

impl NormKind {
    fn variant(self) -> u32 {
        match self {
            NormKind::Plain => 0,
            NormKind::UnitShift => 1,
        }
    }
}

/// Gated-MLP activation variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    /// `silu(gate) * up` (Llama / Qwen).
    SwiGlu,
    /// `gelu(gate) * up` (Gemma).
    GeGlu,
}

impl Activation {
    fn variant(self) -> u32 {
        match self {
            Activation::SwiGlu => 0,
            Activation::GeGlu => 1,
        }
    }
}

/// GPU matmul `C[m,n] = A[m,k] · B[k,n]` (row-major f32). Returns the result
/// buffer; read it with `ctx.read`.
pub fn matmul(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> wgpu::Buffer {
    let c = ctx.empty(m * n);
    let dims = [m as u32, k as u32, n as u32, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("matmul", MATMUL_WGSL, "main");
    let wg = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
    ctx.run(&pipeline, &[a, b, &c, &dims_buf], wg);
    c
}

/// RMSNorm over `rows × dim`: `y = x / sqrt(mean(x²) + eps) * gain(weight)`,
/// where the gain is the family's `NormKind`. `weight` has length `dim`.
pub fn rmsnorm(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    rows: usize,
    dim: usize,
    eps: f32,
    kind: NormKind,
) -> wgpu::Buffer {
    let y = ctx.empty(rows * dim);
    // Dims { rows: u32, dim: u32, eps: f32, variant: u32 } — mixed types, pack by hand.
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
    u[8..12].copy_from_slice(&eps.to_le_bytes());
    u[12..16].copy_from_slice(&kind.variant().to_le_bytes());
    let dims_buf = ctx.uniform(&u);
    let pipeline = ctx.pipeline("rmsnorm", RMSNORM_WGSL, "main");
    ctx.run(&pipeline, &[x, weight, &y, &dims_buf], ((rows as u32).div_ceil(64), 1, 1));
    y
}

/// Gated MLP: `out = act(gate) * up`, elementwise (length `n`), where `act` is
/// the family's `Activation` (SwiGLU / GeGLU).
pub fn glu(
    ctx: &GpuContext,
    gate: &wgpu::Buffer,
    up: &wgpu::Buffer,
    n: usize,
    act: Activation,
) -> wgpu::Buffer {
    let out = ctx.empty(n);
    let dims = [n as u32, act.variant(), 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("glu", GLU_WGSL, "main");
    ctx.run(&pipeline, &[gate, up, &out, &dims_buf], ((n as u32).div_ceil(256), 1, 1));
    out
}

/// RoPE (rotate-half) over `rows × head_dim`, all rows at position `pos`.
pub fn rope(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    rows: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) -> wgpu::Buffer {
    let y = ctx.empty(rows * head_dim);
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(head_dim as u32).to_le_bytes());
    u[8..12].copy_from_slice(&(pos as u32).to_le_bytes());
    u[12..16].copy_from_slice(&theta.to_le_bytes());
    let dims_buf = ctx.uniform(&u);
    let pipeline = ctx.pipeline("rope", ROPE_WGSL, "main");
    ctx.run(&pipeline, &[x, &y, &dims_buf], ((rows as u32).div_ceil(64), 1, 1));
    y
}
