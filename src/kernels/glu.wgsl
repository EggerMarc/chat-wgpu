// Gated-MLP building blocks: out = act(gate) * up, elementwise. Each gate
// activation is its own kernel. These are themselves a fusion (activation +
// elementwise multiply); a family that wants the gate/up matmuls folded in too
// uses the fused-MLP block (todo), which dispatches a register-blocked kernel.
//
//   swiglu  act = silu  (Llama / Qwen)
//   geglu   act = gelu  (Gemma)

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

fn silu_f(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

fn gelu_f(x: f32) -> f32 {
    let c = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    out[i] = silu_f(gate[i]) * up[i];
}

@compute @workgroup_size(256)
fn geglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    out[i] = gelu_f(gate[i]) * up[i];
}
