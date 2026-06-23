//! On-device argmax: logits `[n]` → a single u32 index, so decode reads back
//! 4 bytes instead of the whole vocabulary vector.

use crate::context::GpuContext;

const WGSL: &str = include_str!("argmax.wgsl");

/// Index of the maximum of `logits[0..n]`, written as a u32 into a 1-element
/// buffer. Read it back with `ctx.read(&buf, 1)` and `bits.to_bits()` (the f32
/// readback reinterprets the u32 bytes).
pub fn argmax(ctx: &GpuContext, logits: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    let out = ctx.empty(1);
    let dims = [n as u32, 0, 0, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("argmax", WGSL, "main");
    ctx.run(&pipeline, &[logits, &out, &dims_buf], (1, 1, 1));
    out
}
