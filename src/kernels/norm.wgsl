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

fn row_inv(base: u32) -> f32 {
    var ss = 0.0;
    for (var i: u32 = 0u; i < d.dim; i = i + 1u) {
        let v = X[base + i];
        ss = ss + v * v;
    }
    return inverseSqrt(ss / f32(d.dim) + d.eps);
}

@compute @workgroup_size(64)
fn rmsnorm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= d.rows) { return; }
    let base = row * d.dim;
    let inv = row_inv(base);
    for (var i: u32 = 0u; i < d.dim; i = i + 1u) {
        Y[base + i] = X[base + i] * inv * W[i];
    }
}

@compute @workgroup_size(64)
fn rmsnorm_unit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= d.rows) { return; }
    let base = row * d.dim;
    let inv = row_inv(base);
    for (var i: u32 = 0u; i < d.dim; i = i + 1u) {
        Y[base + i] = X[base + i] * inv * (1.0 + W[i]);
    }
}
