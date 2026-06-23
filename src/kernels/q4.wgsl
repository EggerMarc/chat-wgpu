// q4-in-VRAM GEMV. y[out] = sum_in x[in]*W, W kept Q4_0-packed (blocks of 32
// along in: f32 scale + 4-bit quants), dequantized inline.
//
// One SUBGROUP (32 lanes on Apple/Metal) per output → out_dim workgroups, high
// occupancy. The 32 lanes stride the in-dim coalesced; one `subgroupAdd` reduces
// the partial dots — no shared memory, no barriers. This is MLX's qmv structure.
//
// (naga 29's WGSL frontend rejects the `enable subgroups;` directive but DOES
// recognize the subgroup builtins in conv.rs, so we use them directly. The
// SUBGROUP device feature is requested in GpuContext::new.)

struct Dims { in_dim: u32, out_dim: u32, grid_x: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       quants4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let o = wid.y * d.grid_x + wid.x;
    if (o >= d.out_dim) { return; }
    let nblocks = d.in_dim / 32u;
    let row = o * nblocks; // vec4<u32> / scale index per output row

    // Each lane owns whole 32-weight blocks (b = lane, lane+32, …): one
    // vectorized vec4<u32> load per block, 32 dot terms, subgroupAdd reduces.
    var acc = 0.0;
    for (var b = lane; b < nblocks; b = b + 32u) {
        let scale = scales[row + b];
        let qv = quants4[row + b];
        let base = b * 32u;
        var s = 0.0;
        for (var w = 0u; w < 4u; w = w + 1u) {
            let word = qv[w];
            let xb = base + w * 8u;
            s = s + (f32((word) & 0xFu) - 8.0) * x[xb];
            s = s + (f32((word >> 4u) & 0xFu) - 8.0) * x[xb + 1u];
            s = s + (f32((word >> 8u) & 0xFu) - 8.0) * x[xb + 2u];
            s = s + (f32((word >> 12u) & 0xFu) - 8.0) * x[xb + 3u];
            s = s + (f32((word >> 16u) & 0xFu) - 8.0) * x[xb + 4u];
            s = s + (f32((word >> 20u) & 0xFu) - 8.0) * x[xb + 5u];
            s = s + (f32((word >> 24u) & 0xFu) - 8.0) * x[xb + 6u];
            s = s + (f32((word >> 28u) & 0xFu) - 8.0) * x[xb + 7u];
        }
        acc = acc + s * scale;
    }
    let sum = subgroupAdd(acc);
    if (lane == 0u) {
        y[o] = sum;
    }
}
