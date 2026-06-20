// C[m,n] = A[m,k] · B[k,n], row-major, f32. One thread per output element.
// First kernel of the engine — the decode/prefill workhorse. A tiled +
// quantized version supersedes this for the hot path; this stays as the
// correctness reference and the f32 fallback.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= d.m || col >= d.n) { return; }
    var sum = 0.0;
    for (var kk: u32 = 0u; kk < d.k; kk = kk + 1u) {
        sum = sum + A[row * d.k + kk] * B[kk * d.n + col];
    }
    C[row * d.n + col] = sum;
}
