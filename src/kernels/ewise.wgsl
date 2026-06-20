// Elementwise building blocks. `add` is the residual connection.

struct Dims { n: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             d: Dims;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if (i >= d.n) { return; }
    C[i] = A[i] + B[i];
}
