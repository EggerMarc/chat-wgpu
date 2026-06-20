// Single-query (decode) attention building block: one query position attends to
// `seq` cached keys/values, with GQA. One thread per query head; two passes over
// the key dimension for a numerically-stable softmax. head_dim <= 256.
//
//   Q: [n_heads, head_dim]            (current token, per head)
//   K: [seq, n_kv_heads, head_dim]    (cache)
//   V: [seq, n_kv_heads, head_dim]
//   O: [n_heads, head_dim]
//
// A flash-style key-parallel version replaces this for throughput later; this is
// the correctness reference.

struct Dims { n_heads: u32, n_kv_heads: u32, seq: u32, head_dim: u32, scale: f32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       Q: array<f32>;
@group(0) @binding(1) var<storage, read>       K: array<f32>;
@group(0) @binding(2) var<storage, read>       V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform>             d: Dims;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= d.n_heads) { return; }
    let hd = d.head_dim;
    let kv = h / (d.n_heads / d.n_kv_heads); // GQA: query head -> kv head
    let qbase = h * hd;
    let kvstride = d.n_kv_heads * hd;

    // pass 1: max score
    var m = -3.4e38;
    for (var j: u32 = 0u; j < d.seq; j = j + 1u) {
        let kbase = j * kvstride + kv * hd;
        var dot = 0.0;
        for (var t: u32 = 0u; t < hd; t = t + 1u) { dot = dot + Q[qbase + t] * K[kbase + t]; }
        m = max(m, dot * d.scale);
    }

    // pass 2: softmax-weighted sum of V
    var denom = 0.0;
    var acc: array<f32, 256>;
    for (var t: u32 = 0u; t < hd; t = t + 1u) { acc[t] = 0.0; }
    for (var j: u32 = 0u; j < d.seq; j = j + 1u) {
        let kbase = j * kvstride + kv * hd;
        var dot = 0.0;
        for (var t: u32 = 0u; t < hd; t = t + 1u) { dot = dot + Q[qbase + t] * K[kbase + t]; }
        let w = exp(dot * d.scale - m);
        denom = denom + w;
        for (var t: u32 = 0u; t < hd; t = t + 1u) { acc[t] = acc[t] + w * V[kbase + t]; }
    }

    let inv = 1.0 / denom;
    for (var t: u32 = 0u; t < hd; t = t + 1u) { O[qbase + t] = acc[t] * inv; }
}
