//! Normalization building blocks. Each gain rule is a distinct kernel.

use crate::context::GpuContext;

const WGSL: &str = include_str!("norm.wgsl");

fn dims(rows: usize, dim: usize, eps: f32) -> [u8; 16] {
    let mut u = [0u8; 16];
    u[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(dim as u32).to_le_bytes());
    u[8..12].copy_from_slice(&eps.to_le_bytes());
    u
}

fn run(ctx: &GpuContext, entry: &'static str, x: &wgpu::Buffer, w: &wgpu::Buffer,
       rows: usize, dim: usize, eps: f32) -> wgpu::Buffer {
    let y = ctx.empty(rows * dim);
    let dims_buf = ctx.uniform(&dims(rows, dim, eps));
    let pipeline = ctx.pipeline(entry, WGSL, entry);
    // One workgroup (= one 32-lane subgroup) per row. See norm.wgsl.
    ctx.run(&pipeline, &[x, w, &y, &dims_buf], (rows as u32, 1, 1));
    y
}

/// RMSNorm, plain gain (`y = x/rms * weight`). Llama / Qwen.
pub fn rmsnorm(ctx: &GpuContext, x: &wgpu::Buffer, w: &wgpu::Buffer,
               rows: usize, dim: usize, eps: f32) -> wgpu::Buffer {
    run(ctx, "rmsnorm", x, w, rows, dim, eps)
}

/// RMSNorm, unit-shift gain (`y = x/rms * (1 + weight)`). Gemma.
pub fn rmsnorm_unit(ctx: &GpuContext, x: &wgpu::Buffer, w: &wgpu::Buffer,
                    rows: usize, dim: usize, eps: f32) -> wgpu::Buffer {
    run(ctx, "rmsnorm_unit", x, w, rows, dim, eps)
}
