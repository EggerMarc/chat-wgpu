// Unary activation building blocks: y = f(x), elementwise. Each is its own
// kernel — composable primitives. `swiglu`/`geglu` (the fused gate·up forms)
// live in glu.wgsl.

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

fn silu_f(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

fn gelu_f(x: f32) -> f32 {
    let c = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    Y[i] = silu_f(X[i]);
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    Y[i] = gelu_f(X[i]);
}

@compute @workgroup_size(256)
fn tanh_act(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    Y[i] = tanh(X[i]);
}
