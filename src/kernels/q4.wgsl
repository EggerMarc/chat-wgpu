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

// Register-blocked GEMV: one subgroup (32 lanes) computes ROWS output rows at
// once. Each lane loads its 32-wide x block into REGISTERS once and reuses it
// across all ROWS weight rows — x (the dominant traffic, re-read per row in the
// old 1-row kernel) is fetched once, and ROWS× more work per thread amortizes
// the subgroupAdd + launch overhead. No threadgroup memory, so occupancy holds.
const ROWS = 4u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let o0 = (wid.y * d.grid_x + wid.x) * ROWS; // first output row of this subgroup
    let nblocks = d.in_dim / 32u;

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var b = lane; b < nblocks; b = b + 32u) {
        let base = b * 32u;
        // x block → 8 vec4 registers, loaded once, reused across ROWS rows.
        var xr: array<vec4<f32>, 8>;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let xb = base + k * 4u;
            xr[k] = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        }
        for (var r = 0u; r < ROWS; r = r + 1u) {
            let row = (o0 + r) * nblocks; // OOB rows clamp-read 0, never written
            let qv = quants4[row + b];
            let scale = scales[row + b];
            var s = 0.0;
            for (var w = 0u; w < 4u; w = w + 1u) {
                let word = qv[w];
                let lo = vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
                let hi = vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu)) - vec4<f32>(8.0);
                s = s + dot(lo, xr[w * 2u]) + dot(hi, xr[w * 2u + 1u]);
            }
            acc[r] = acc[r] + s * scale;
        }
    }
    for (var r = 0u; r < ROWS; r = r + 1u) {
        let sum = subgroupAdd(acc[r]);
        let o = o0 + r;
        if (lane == 0u && o < d.out_dim) {
            y[o] = sum;
        }
    }
}
