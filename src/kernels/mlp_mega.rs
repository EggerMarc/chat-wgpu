//! MEGAKERNEL TEST: whole MLP sublayer in one dispatch (f32 weights).

use crate::context::GpuContext;

const WGSL: &str = include_str!("mlp_mega.wgsl");

/// `out = hidden + Wdown · (silu(Wgate·norm(hidden)) * (Wup·norm(hidden)))`,
/// one dispatch. `wg`/`wu` are `[dim, hidden]`, `wd` is `[hidden, dim]` (matmul
/// B-layout). `ffn_w` is the RMSNorm gain `[dim]`. Decode (one token).
#[allow(clippy::too_many_arguments)]
pub fn mlp(
    ctx: &GpuContext,
    hidden: &wgpu::Buffer,
    ffn_w: &wgpu::Buffer,
    wg: &wgpu::Buffer,
    wu: &wgpu::Buffer,
    wd: &wgpu::Buffer,
    dim: usize,
    hidden_dim: usize,
    eps: f32,
) -> wgpu::Buffer {
    let out = ctx.empty(dim);
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(dim as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(hidden_dim as u32).to_le_bytes());
    u[8..12].copy_from_slice(&eps.to_le_bytes());
    let dims_buf = ctx.uniform(&u);
    let pipeline = ctx.pipeline("mlp_mega", WGSL, "main");
    // ONE workgroup for the whole MLP of this token.
    ctx.run(&pipeline, &[hidden, ffn_w, wg, wu, wd, &out, &dims_buf], (1, 1, 1));
    out
}
