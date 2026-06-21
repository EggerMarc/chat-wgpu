// MEGAKERNEL TEST: the entire MLP sublayer in ONE dispatch —
//   out = hidden + Wdown · ( silu(Wgate·rmsnorm(hidden)) * (Wup·rmsnorm(hidden)) )
// replacing 6 dependent passes (norm, gate, up, swiglu, down, residual) and
// their 5 inter-pass hazard barriers with one pass using cheap *intra*-workgroup
// barriers. One workgroup per token (decode), so occupancy is low — the test is
// whether killing the inter-pass barriers wins anyway.
//
//   hidden_in [dim], ffn_w [dim], Wg/Wu [dim,hidden] (B-layout), Wd [hidden,dim]
//   out [dim] = hidden_in + Wd·act

struct Dims { dim: u32, hidden: u32, eps: f32, _p: u32 };
@group(0) @binding(0) var<storage, read>       hidden_in: array<f32>;
@group(0) @binding(1) var<storage, read>       ffn_w: array<f32>;
@group(0) @binding(2) var<storage, read>       Wg: array<f32>;
@group(0) @binding(3) var<storage, read>       Wu: array<f32>;
@group(0) @binding(4) var<storage, read>       Wd: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform>             d: Dims;

const WG: u32 = 256u;
var<workgroup> sh_normed: array<f32, 2048>; // dim   <= 2048
var<workgroup> sh_act: array<f32, 4096>;    // hidden <= 4096
var<workgroup> sh_red: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;

    // 1. RMS over `dim` (sum of squares → workgroup reduction)
    var ss = 0.0;
    var i = t;
    while (i < d.dim) {
        let v = hidden_in[i];
        ss = ss + v * v;
        i = i + WG;
    }
    sh_red[t] = ss;
    workgroupBarrier();
    var stride = WG / 2u;
    while (stride > 0u) {
        if (t < stride) { sh_red[t] = sh_red[t] + sh_red[t + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(sh_red[0] / f32(d.dim) + d.eps);

    // 2. normed = hidden/rms * ffn_w
    i = t;
    while (i < d.dim) {
        sh_normed[i] = hidden_in[i] * inv * ffn_w[i];
        i = i + WG;
    }
    workgroupBarrier();

    // 3. act[h] = silu(gate·normed) * (up·normed)
    var h = t;
    while (h < d.hidden) {
        var g = 0.0;
        var u = 0.0;
        for (var k: u32 = 0u; k < d.dim; k = k + 1u) {
            let nv = sh_normed[k];
            let idx = k * d.hidden + h;
            g = g + nv * Wg[idx];
            u = u + nv * Wu[idx];
        }
        sh_act[h] = (g / (1.0 + exp(-g))) * u;
        h = h + WG;
    }
    workgroupBarrier();

    // 4. down + residual: out[o] = hidden[o] + Wd·act
    var o = t;
    while (o < d.dim) {
        var acc = 0.0;
        for (var k: u32 = 0u; k < d.hidden; k = k + 1u) {
            acc = acc + sh_act[k] * Wd[k * d.dim + o];
        }
        out[o] = hidden_in[o] + acc;
        o = o + WG;
    }
}
