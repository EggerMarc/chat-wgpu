// RoPE (rotary position embedding), GPT-NeoX / "rotate-half" layout — the form
// Qwen and Llama use. Each row is one head's `head_dim` vector at sequence
// position `pos`; head_dim is split in half and rotated pairwise:
//
//   freq_i = pos * theta^(-2i/head_dim),  i in [0, half)
//   y[i]      = x[i]*cos(freq_i) - x[i+half]*sin(freq_i)
//   y[i+half] = x[i+half]*cos(freq_i) + x[i]*sin(freq_i)
//
// All rows share one position (a single decode token across its heads). For
// prefill the host calls this per position. cos/sin are computed inline here;
// the model can swap in precomputed tables later.

struct Dims { rows: u32, head_dim: u32, pos: u32, theta: f32 };
@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= d.rows) { return; }
    let base = row * d.head_dim;
    let half = d.head_dim / 2u;

    for (var i: u32 = 0u; i < half; i = i + 1u) {
        let exponent = -2.0 * f32(i) / f32(d.head_dim);
        let freq = f32(d.pos) * pow(d.theta, exponent);
        let c = cos(freq);
        let s = sin(freq);
        let lo = X[base + i];
        let hi = X[base + i + half];
        Y[base + i] = lo * c - hi * s;
        Y[base + i + half] = hi * c + lo * s;
    }
}
