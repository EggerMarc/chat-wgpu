// RMSNorm: per row, y = x / sqrt(mean(x^2) + eps) * gain(weight).
// One thread per row, serial over `dim` (two passes). Correct and fine for the
// small row counts of decode; a workgroup-reduction version can replace it for
// large-batch prefill.
//
// Family-agnostic via `variant`:
//   0 = plain      (Llama / Qwen):  gain = weight
//   1 = unit-shift (Gemma):         gain = 1 + weight

struct Dims { rows: u32, dim: u32, eps: f32, variant: u32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= d.rows) { return; }
    let base = row * d.dim;

    var ss = 0.0;
    for (var i: u32 = 0u; i < d.dim; i = i + 1u) {
        let v = X[base + i];
        ss = ss + v * v;
    }
    let inv = inverseSqrt(ss / f32(d.dim) + d.eps);

    for (var i: u32 = 0u; i < d.dim; i = i + 1u) {
        let w = W[i];
        let gain = select(w, 1.0 + w, d.variant == 1u);
        Y[base + i] = X[base + i] * inv * gain;
    }
}
