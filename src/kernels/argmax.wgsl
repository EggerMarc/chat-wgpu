// On-device argmax over the logits — returns the index of the max as a u32,
// so decode reads back 4 bytes instead of the whole [vocab] vector. One
// workgroup (256 threads) grid-strides the vector into per-thread (value,index)
// bests, then a shared tree-reduction picks the winner. Ties resolve to the
// lowest index, matching a CPU `>`-scan argmax.

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform>             d: Dims;

const WG = 256u;
const NEG_INF = -3.402823e38;

var<workgroup> sv: array<f32, 256>;
var<workgroup> si: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;

    // Per-thread best over a strided slice. Strict `>` keeps the lowest index.
    var best_v = NEG_INF;
    var best_i = 0u;
    for (var i = t; i < d.n; i = i + WG) {
        let v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    sv[t] = best_v;
    si[t] = best_i;
    workgroupBarrier();

    // Tree reduction; on equal value prefer the lower index.
    var stride = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (t < stride) {
            let ov = sv[t + stride];
            let oi = si[t + stride];
            if (ov > sv[t] || (ov == sv[t] && oi < si[t])) {
                sv[t] = ov;
                si[t] = oi;
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (t == 0u) {
        out[0] = si[0];
    }
}
