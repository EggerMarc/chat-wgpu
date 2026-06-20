// SwiGLU activation: out = silu(gate) * up, elementwise, where
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). `gate` and `up` are the two MLP
// projections of equal length; one thread per element.

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    let g = gate[i];
    let silu = g / (1.0 + exp(-g));
    out[i] = silu * up[i];
}
