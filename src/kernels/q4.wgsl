// q4-in-VRAM GEMV: y[out] = sum_in x[in] * W[out, in], W kept Q4_0-packed in
// VRAM and dequantized inline.
//
// One WORKGROUP per output row; its 64 threads split the in-loop and read
// *adjacent* memory (coalesced), then reduce in shared memory. Coalescing is the
// whole game for a memory-bound GEMV — thread-per-output read the weight row
// strided, which serializes into one memory transaction per element.
//
//   x:      [in]                  activation (f32)
//   scales: [out * in/32]         per-block f32 scale
//   quants: [out * in/8]          packed 4-bit quants (8 nibbles per u32)
//   y:      [out]

struct Dims { in_dim: u32, out_dim: u32, grid_x: u32, _b: u32 };
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       quants: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // 2-D workgroup grid (the dispatch dimension caps at 65535, but out can be
    // ~150k for the lm-head).
    let o = wid.y * d.grid_x + wid.x;
    if (o >= d.out_dim) { return; }
    let t = lid.x;
    let nblocks = d.in_dim / 32u;
    let qrow = o * (d.in_dim / 8u);
    let srow = o * nblocks;

    var acc = 0.0;
    var k = t;
    loop {
        if (k >= d.in_dim) { break; }
        // thread t handles input k; consecutive threads -> consecutive nibbles
        // (8 threads share a u32, 64 threads read 8 contiguous u32 = coalesced).
        let scale = scales[srow + k / 32u];
        let word = quants[qrow + k / 8u];
        let q = (word >> ((k % 8u) * 4u)) & 0xFu;
        acc = acc + scale * (f32(q) - 8.0) * x[k];
        k = k + WG;
    }

    partial[t] = acc;
    workgroupBarrier();
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (t == 0u) { y[o] = partial[0]; }
}
