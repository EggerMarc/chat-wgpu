// Normalization building blocks. Each gain rule is its OWN entry point (its own
// kernel / building block) — families pick the one they use; nothing branches on
// a runtime flag.
//
//   rmsnorm       gain = weight        (Llama / Qwen)
//   rmsnorm_unit  gain = 1 + weight    (Gemma)

struct Dims { rows: u32, dim: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

// One WORKGROUP (= one 32-lane subgroup on Apple/Metal) per row, dispatched as
// `(rows, 1, 1)`. The 32 lanes stride the row coalesced and a single
// `subgroupAdd` reduces the sum-of-squares — replaces the old 1-thread-per-row
// serial reduction (which, at decode's rows=1, ran the whole dim on one lane).
const WG = 32u;

fn row_inv(base: u32, lane: u32) -> f32 {
    var ss = 0.0;
    for (var i = lane; i < d.dim; i += WG) {
        let v = X[base + i];
        ss = ss + v * v;
    }
    ss = subgroupAdd(ss);
    return inverseSqrt(ss / f32(d.dim) + d.eps);
}

@compute @workgroup_size(WG)
fn rmsnorm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = wid.x;
    if (row >= d.rows) { return; }
    let base = row * d.dim;
    let inv = row_inv(base, lane);
    for (var i = lane; i < d.dim; i += WG) {
        Y[base + i] = X[base + i] * inv * W[i];
    }
}

@compute @workgroup_size(WG)
fn rmsnorm_unit(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = wid.x;
    if (row >= d.rows) { return; }
    let base = row * d.dim;
    let inv = row_inv(base, lane);
    for (var i = lane; i < d.dim; i += WG) {
        Y[base + i] = X[base + i] * inv * (1.0 + W[i]);
    }
}
