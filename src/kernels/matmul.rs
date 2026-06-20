//! Matmul building block.

use crate::context::GpuContext;

const WGSL: &str = include_str!("matmul.wgsl");

/// `C[m,n] = A[m,k] · B[k,n]` (row-major f32). Returns the result buffer.
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
    let pipeline = ctx.pipeline("matmul", WGSL, "main");
    let wg = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
    ctx.run(&pipeline, &[a, b, &c, &dims_buf], wg);
    c
}
