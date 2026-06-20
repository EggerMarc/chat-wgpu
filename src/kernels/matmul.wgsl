// C[m,n] = A[m,k] · B[k,n], row-major, f32.
//
// Two entry points. `gemv` is the decode hot path (m = 1): 1-D over the n
// outputs, full thread occupancy, no per-thread integer division. `main` is the
// 2-D tile for prefill / m > 1, taking row/col straight from the dispatch grid
// (also division-free). The host picks `gemv` when m == 1.

struct Dims { m: u32, k: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

// Shared-memory caching of A was tried and *regressed* (16 KB/workgroup tanks
// occupancy on the 151936-wide lm-head). A is small and L2-resident, so the
// plain global read wins here.
@compute @workgroup_size(256)
fn gemv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= d.n) { return; }
    var sum = 0.0;
    for (var kk: u32 = 0u; kk < d.k; kk = kk + 1u) {
        sum = sum + A[kk] * B[kk * d.n + col];
    }
    C[col] = sum;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= d.m || col >= d.n) { return; }
    let arow = row * d.k;
    var sum = 0.0;
    for (var kk: u32 = 0u; kk < d.k; kk = kk + 1u) {
        sum = sum + A[arow + kk] * B[kk * d.n + col];
    }
    C[row * d.n + col] = sum;
}
