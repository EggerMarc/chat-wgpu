//! q4-in-VRAM GEMV building block + the host-side re-quantizer.

use crate::context::GpuContext;

const WGSL: &str = include_str!("q4.wgsl");

/// q4 GEMV: `y[out] = sum_in x[in] * W`, where W is the packed q4 weight
/// (`scales` + `quants`). `x` is `[in_f]`. Decode only (one token).
pub fn gemv(
    ctx: &GpuContext,
    x: &wgpu::Buffer,
    scales: &wgpu::Buffer,
    quants: &wgpu::Buffer,
    in_f: usize,
    out_f: usize,
) -> wgpu::Buffer {
    let out = ctx.empty(out_f);
    // One subgroup computes ROWS (=4) output rows; grid tiled across x/y since the
    // dispatch dimension caps at 65535 (out_f reaches the vocab size). Must match
    // ROWS in q4.wgsl.
    const ROWS: u32 = 4;
    let groups = (out_f as u32).div_ceil(ROWS);
    let grid_x = groups.min(65535);
    let grid_y = groups.div_ceil(grid_x);
    let dims = [in_f as u32, out_f as u32, grid_x, 0u32];
    let dims_buf = ctx.uniform(bytemuck::cast_slice(&dims));
    let pipeline = ctx.pipeline("q4_gemv", WGSL, "main");
    ctx.run(&pipeline, &[x, scales, quants, &out, &dims_buf], (grid_x, grid_y, 1));
    out
}

/// Re-quantize a native `[out_f, in_f]` f32 weight to Q4_0-style blocks of 32
/// along `in_f`: per block a single f32 scale `d = absmax/8`, and 4-bit quants
/// `q = round(v/d)+8`. Each output row is contiguous (so the GEMV workgroup
/// streams it coalesced). Returns `(scales[out_f*in_f/32], quants[out_f*in_f/8])`
/// (8 nibbles per u32). `in_f` must be a multiple of 32.
pub fn quantize_q4_0(data: &[f32], out_f: usize, in_f: usize) -> (Vec<f32>, Vec<u32>) {
    assert!(in_f % 32 == 0, "q4 expects in_f multiple of 32");
    let nblocks = in_f / 32;
    let mut scales = vec![0f32; out_f * nblocks];
    let mut quants = vec![0u32; out_f * (in_f / 8)];
    for o in 0..out_f {
        for b in 0..nblocks {
            let base = o * in_f + b * 32;
            let absmax = (0..32).fold(0f32, |m, j| m.max(data[base + j].abs()));
            let d = absmax / 8.0;
            let inv = if d != 0.0 { 1.0 / d } else { 0.0 };
            scales[o * nblocks + b] = d;
            for w in 0..4 {
                let mut word = 0u32;
                for n in 0..8 {
                    let v = data[base + w * 8 + n];
                    let q = ((v * inv).round() as i32 + 8).clamp(0, 15) as u32;
                    word |= q << (n * 4);
                }
                quants[o * (in_f / 8) + b * 4 + w] = word;
            }
        }
    }
    (scales, quants)
}
