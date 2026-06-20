//! WGSL compute kernels and their host-side dispatch wrappers — **GPU only**.
//!
//! Kernels are added in dependency order and checked with GPU known-answer
//! tests (`src/main.rs`): structural inputs whose outputs are closed-form
//! (identity → passthrough, ones → sum, rope@pos0 → identity). No CPU
//! reimplementation of any kernel lives here — this is a pure wgpu provider.
//!
//!   matmul (f32)  ✅
//!   rmsnorm       ✅
//!   swiglu        ✅
//!   rope          ✅
//!   softmax / attention   todo
//!   q4 dequant-matmul (the hot path)  todo
//!
//! Convention: buffers bind at sequential `@binding` 0.., a uniform last.

use crate::context::GpuContext;

const MATMUL_WGSL: &str = include_str!("matmul.wgsl");
const RMSNORM_WGSL: &str = include_str!("rmsnorm.wgsl");
const SWIGLU_WGSL: &str = include_str!("swiglu.wgsl");
const ROPE_WGSL: &str = include_str!("rope.wgsl");

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

/// RMSNorm over `rows × dim`: `y = x / sqrt(mean(x²) + eps) * weight`.
/// `weight` has length `dim`. Plain (Llama/Qwen) variant.
pub fn rmsnorm(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    weight: &wgpu::Buffer,
    rows: usize,
    dim: usize,
    eps: f32,
) -> wgpu::Buffer {
    let y = ctx.empty(rows * dim);
    // Dims { rows: u32, dim: u32, eps: f32, _pad: u32 } — mixed types, pack by hand.
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
    u[8..12].copy_from_slice(&eps.to_le_bytes());
    let dims_buf = ctx.uniform(&u);
    let pipeline = ctx.pipeline("rmsnorm", RMSNORM_WGSL, "main");
    ctx.run(&pipeline, &[x, weight, &y, &dims_buf], ((rows as u32).div_ceil(64), 1, 1));
    y
}

/// SwiGLU: `out = silu(gate) * up`, elementwise (length `n`).
pub fn swiglu(ctx: &GpuContext, gate: &wgpu::Buffer, up: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    let out = ctx.empty(n);
    let dims = [n as u32, 0, 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("swiglu", SWIGLU_WGSL, "main");
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
