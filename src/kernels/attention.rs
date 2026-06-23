use crate::context::GpuContext;

const ATTN_WGSL: &str = include_str!("attention.wgsl");
const EWISE_WGSL: &str = include_str!("ewise.wgsl");

#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &GpuContext,
    q: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
    n_heads: usize,
    n_kv_heads: usize,
    seq: usize,
    head_dim: usize,
) -> wgpu::Buffer {
    let out = ctx.empty(n_heads * head_dim);
    let scale = (head_dim as f32).powf(-0.5);
    let mut u = [0u8; 32];
    u[0..4].copy_from_slice(&(n_heads as u32).to_le_bytes());
    u[4..8].copy_from_slice(&(n_kv_heads as u32).to_le_bytes());
    u[8..12].copy_from_slice(&(seq as u32).to_le_bytes());
    u[12..16].copy_from_slice(&(head_dim as u32).to_le_bytes());
    u[16..20].copy_from_slice(&scale.to_le_bytes());
    let dims_buf = ctx.uniform(&u);
    let pipeline = ctx.pipeline("attention", ATTN_WGSL, "main");
    // One workgroup (= one 32-lane subgroup) per head; the lanes split the KV
    // sequence. See attention.wgsl.
    ctx.run_uncached(
        &pipeline,
        &[q, k, v, &out, &dims_buf],
        (n_heads as u32, 1, 1),
    );
    out
}

pub fn add(ctx: &GpuContext, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    let c = ctx.empty(n);
    let dims = [n as u32, 0, 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("ewise_add", EWISE_WGSL, "add");
    ctx.run(
        &pipeline,
        &[a, b, &c, &dims_buf],
        ((n as u32).div_ceil(256), 1, 1),
    );
    c
}
