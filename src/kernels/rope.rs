//! RoPE building block.

use crate::context::GpuContext;

const WGSL: &str = include_str!("rope.wgsl");

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
    let pipeline = ctx.pipeline("rope", WGSL, "main");
    // Uniform carries `pos` (varies per token) → can't use the bind-group cache.
    ctx.run_uncached(&pipeline, &[x, &y, &dims_buf], ((rows as u32).div_ceil(64), 1, 1));
    y
}
