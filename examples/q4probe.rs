//! Read-bandwidth sweep: how does pure quants4 read throughput scale with ROWS
//! (output rows per subgroup = work/memory-requests per threadgroup)? No compute
//! beyond an XOR to defeat elision. Finds whether the ~28 GB/s read ceiling is an
//! occupancy / in-flight-requests problem (climbs with ROWS) or a hard wall.

use chat_wgpu::context::GpuContext;

const IN_F: usize = 1024;
const OUT_F: usize = 151936;
const ITERS: u32 = 300;

fn raw_wgsl(rows: u32) -> String {
    format!(
        r#"
struct Dims {{ row_vecs: u32, out_dim: u32, grid_x: u32, _b: u32 }};
@group(0) @binding(0) var<storage, read>       quants4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform>             d: Dims;
const ROWS = {rows}u;
@compute @workgroup_size(32)
fn raw(@builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {{
    let o0 = (wid.y * d.grid_x + wid.x) * ROWS;
    var acc: array<u32, {rows}>;
    for (var r = 0u; r < ROWS; r = r + 1u) {{ acc[r] = 0u; }}
    for (var r = 0u; r < ROWS; r = r + 1u) {{
        var p = (o0 + r) * d.row_vecs + lane;
        for (var b = lane; b < d.row_vecs; b = b + 32u) {{
            let q = quants4[p];
            acc[r] = acc[r] ^ q.x ^ q.y ^ q.z ^ q.w;
            p = p + 32u;
        }}
    }}
    for (var r = 0u; r < ROWS; r = r + 1u) {{
        let s = subgroupAdd(f32(acc[r] & 1u));
        let o = o0 + r;
        if (lane == 0u && o < d.out_dim) {{ y[o] = s; }}
    }}
}}
"#
    )
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = GpuContext::new().await.expect("gpu");
    println!("backend: {}  shape {IN_F}->{OUT_F}  (read {} MB)\n", ctx.backend, OUT_F * (IN_F / 8) * 4 / 1_000_000);

    let row_vecs = (IN_F / 8) as u32;
    let quants = vec![0x88888888u32; OUT_F * (IN_F / 8)];
    let qb = ctx.storage_u32(&quants);
    let qbytes = OUT_F * (IN_F / 8) * 4;

    println!("{:>5}  {:>9}  {:>8}", "ROWS", "us/pass", "GB/s");
    for rows in [1u32, 2, 4, 8, 16, 32] {
        let entry = "raw";
        let name: &'static str = match rows {
            1 => "raw1",
            2 => "raw2",
            4 => "raw4",
            8 => "raw8",
            16 => "raw16",
            _ => "raw32",
        };
        let pipe = ctx.pipeline(name, &raw_wgsl(rows), entry);
        let groups = (OUT_F as u32).div_ceil(rows);
        let grid_x = groups.min(65535);
        let grid_y = groups.div_ceil(grid_x);
        let dims = [row_vecs, OUT_F as u32, grid_x, 0u32];
        let db = ctx.uniform(bytemuck::cast_slice(&dims));

        let out = ctx.empty(OUT_F);
        ctx.run(&pipe, &[&qb, &out, &db], (grid_x, grid_y, 1));
        let _ = ctx.read(&out, 1).await;
        ctx.clear_cache();

        ctx.begin_profile();
        for _ in 0..ITERS {
            ctx.reset_frame();
            let o = ctx.empty(OUT_F);
            ctx.run(&pipe, &[&qb, &o, &db], (grid_x, grid_y, 1));
        }
        let _ = ctx.read(&out, 1).await;
        let _ = entry;
        let rep = ctx.report_profile().await;
        let ms = rep.iter().find(|(n, _, _)| *n == name).map(|(_, _, m)| *m).unwrap_or(0.0);
        ctx.clear_cache();
        let us = ms * 1000.0 / ITERS as f64;
        let gbs = qbytes as f64 / (us * 1e-6) / 1e9;
        println!("{rows:>5}  {us:>9.1}  {gbs:>8.1}");
    }
}
