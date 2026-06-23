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

struct Dims {
    in_dim: u32,
    out_dim: u32,
    grid_x: u32,
    _b: u32};
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>; // <--- Vec4 bound
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> quants4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;

const ROWS = 4u;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let o0 = (wid.y * d.grid_x + wid.x) * ROWS;
    let nblocks = d.in_dim / 32u;

    var w_ptrs = array<u32, 4>();
    for (var r = 0u; r < ROWS; r = r + 1u) {
        w_ptrs[r] = (o0 + r) * nblocks + lane;
    }

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    for (var b = lane; b < nblocks; b = b + 32u) {
        let x_base = b * 8u;

        var xr: array<vec4<f32>, 8>;
        var x_sum = 0.0;

        for (var k = 0u; k < 8u; k = k + 1u) {
            let xv = x[x_base + k];
            xr[k] = xv;
            x_sum = x_sum + dot(xv, vec4<f32>(1.0));
        }

        for (var r = 0u; r < ROWS; r = r + 1u) {
            let p = w_ptrs[r];
            let qv = quants4[p];
            let scale = scales[p];
            w_ptrs[r] = p + 32u;

            var s_lo = 0.0;
            var s_hi = 0.0;

            for (var w = 0u; w < 4u; w = w + 1u) {
                let word = qv[w];
                let lo = vec4<f32>(vec4<u32>(word, word >> 4u, word >> 8u, word >> 12u) & vec4<u32>(0xFu));
                let hi = vec4<f32>(vec4<u32>(word >> 16u, word >> 20u, word >> 24u, word >> 28u) & vec4<u32>(0xFu));

                s_lo = s_lo + dot(lo, xr[w * 2u]);
                s_hi = s_hi + dot(hi, xr[w * 2u + 1u]);
            }

            let s_total = (s_lo + s_hi) - (8.0 * x_sum);
            acc[r] = acc[r] + s_total * scale;
        }
    }

    for (var r = 0u; r < ROWS; r = r + 1u) {
        let sum = subgroupAdd(acc[r]);
        let o = o0 + r;
        if lane == 0u && o < d.out_dim {
            y[o] = sum;
        }
    }
}
