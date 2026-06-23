struct Dims {
    n_heads: u32,
    n_kv_heads: u32,
    seq: u32,
    head_dim: u32,
    scale: f32,
    heads_per_kv: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> d: Dims;

const WG = 32u;
const NEG_INF = -3.402823e38;

var<workgroup> q_sh: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let h = wid.x;
    if h >= d.n_heads {
        return;
    }

    let hd = d.head_dim;
    let hd4 = hd & ~3u;
    let qbase = h * hd;
    let kv_head = h / d.heads_per_kv;
    let kvstride = d.n_kv_heads * hd;

    for (var t = lane; t < hd; t += WG) {
        q_sh[t] = Q[qbase + t];
    }
    workgroupBarrier();

    var m = NEG_INF;
    var denom = 0.0;
    var acc: array<f32, 256>;
    for (var t = 0u; t < hd; t++) {
        acc[t] = 0.0;
    }

    for (var j = lane; j < d.seq; j += WG) {
        let kbase = j * kvstride + kv_head * hd;

        // score = scale * (q · k_j)
        var s = 0.0;
        for (var t = 0u; t < hd4; t += 4u) {
            let qv = vec4<f32>(q_sh[t], q_sh[t + 1u], q_sh[t + 2u], q_sh[t + 3u]);
            let kv = vec4<f32>(K[kbase + t], K[kbase + t + 1u], K[kbase + t + 2u], K[kbase + t + 3u]);
            s += dot(qv, kv);
        }
        for (var t = hd4; t < hd; t++) {
            s += q_sh[t] * K[kbase + t];
        }
        s *= d.scale;

        // Online softmax update: rescale the running accumulators to the new max.
        let m_new = max(m, s);
        let corr = exp(m - m_new);
        let p = exp(s - m_new);
        denom = denom * corr + p;
        for (var t = 0u; t < hd4; t += 4u) {
            let a = vec4<f32>(acc[t], acc[t + 1u], acc[t + 2u], acc[t + 3u]);
            let vv = vec4<f32>(V[kbase + t], V[kbase + t + 1u], V[kbase + t + 2u], V[kbase + t + 3u]);
            let r = a * corr + vv * p;
            acc[t] = r.x;
            acc[t + 1u] = r.y;
            acc[t + 2u] = r.z;
            acc[t + 3u] = r.w;
        }
        for (var t = hd4; t < hd; t++) {
            acc[t] = acc[t] * corr + p * V[kbase + t];
        }
        m = m_new;
    }

    let m_all = subgroupMax(m);
    let corr = exp(m - m_all);
    let inv = 1.0 / subgroupAdd(denom * corr);

    for (var t = 0u; t < hd; t++) {
        let o = subgroupAdd(acc[t] * corr);
        if lane == 0u {
            O[qbase + t] = o * inv;
        }
    }
}
