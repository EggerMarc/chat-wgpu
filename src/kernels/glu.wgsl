// Gated MLP activation: out = act(gate) * up, elementwise. `gate` and `up` are
// the two MLP projections of equal length; one thread per element.
//
// Family-agnostic via `variant`:
//   0 = SwiGLU (Llama / Qwen):  act = silu(x) = x * sigmoid(x)
//   1 = GeGLU  (Gemma):         act = gelu_tanh(x)

struct Dims { n: u32, variant: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       gate: array<f32>;
@group(0) @binding(1) var<storage, read>       up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// tanh approximation of GELU (the form Gemma / GPT-2 use).
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608028654; // sqrt(2/pi)
    return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= d.n) { return; }
    let g = gate[i];
    let act = select(silu(g), gelu_tanh(g), d.variant == 1u);
    out[i] = act * up[i];
}
