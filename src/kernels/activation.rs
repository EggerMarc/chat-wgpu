//! Activation building blocks — discrete kernels, composed per family.
//! Unary forms (`silu`, `gelu`, `tanh`) and the fused gated forms
//! (`swiglu`, `geglu`).

use crate::context::GpuContext;

const ACT_WGSL: &str = include_str!("activation.wgsl");
const GLU_WGSL: &str = include_str!("glu.wgsl");

/// Unary elementwise `y = f(x)`, length `n`.
fn unary(ctx: &GpuContext, entry: &'static str, x: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    let y = ctx.empty(n);
    let dims = [n as u32, 0, 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline(entry, ACT_WGSL, entry);
    ctx.run(&pipeline, &[x, &y, &dims_buf], ((n as u32).div_ceil(256), 1, 1));
    y
}

pub fn silu(ctx: &GpuContext, x: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    unary(ctx, "silu", x, n)
}
pub fn gelu(ctx: &GpuContext, x: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    unary(ctx, "gelu", x, n)
}
pub fn tanh(ctx: &GpuContext, x: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    unary(ctx, "tanh_act", x, n)
}

/// Gated `out = act(gate) * up`, length `n`.
fn gated(ctx: &GpuContext, entry: &'static str, gate: &wgpu::Buffer, up: &wgpu::Buffer,
         n: usize) -> wgpu::Buffer {
    let out = ctx.empty(n);
    let dims = [n as u32, 0, 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline(entry, GLU_WGSL, entry);
    ctx.run(&pipeline, &[gate, up, &out, &dims_buf], ((n as u32).div_ceil(256), 1, 1));
    out
}

/// SwiGLU: `silu(gate) * up`. Llama / Qwen.
pub fn swiglu(ctx: &GpuContext, gate: &wgpu::Buffer, up: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    gated(ctx, "swiglu", gate, up, n)
}
/// GeGLU: `gelu(gate) * up`. Gemma.
pub fn geglu(ctx: &GpuContext, gate: &wgpu::Buffer, up: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    gated(ctx, "geglu", gate, up, n)
}
