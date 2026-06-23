//! Microbench q4_gemv in isolation — clean µs/pass + GB/s per shape, no model.
//! Separates the per-dispatch floor from real bandwidth so we know which lever
//! (cut passes vs faster kernel) the gemv actually needs.
//!
//!   cargo run --release --example q4bench

use chat_wgpu::context::GpuContext;
use chat_wgpu::kernels::q4;

fn main() {
    pollster::block_on(run());
}

// Qwen3-0.6B projection shapes (in_f, out_f, label).
const SHAPES: &[(usize, usize, &str)] = &[
    (1024, 2048, "wq"),
    (1024, 1024, "wk/wv/wo/down"),
    (1024, 3072, "gate/up"),
    (3072, 1024, "down(in=3072)"),
    (1024, 151936, "lm_head"),
];

const ITERS: u32 = 200;

async fn run() {
    let ctx = GpuContext::new().await.expect("gpu");
    println!("backend: {}\n", ctx.backend);
    println!("{:<18} {:>9} {:>10} {:>9}", "shape", "us/pass", "GB/s", "bytes");

    for &(in_f, out_f, label) in SHAPES {
        // Packed q4 weight: out_f * in_f/2 bytes quants + out_f*in_f/32 * 4 scale bytes.
        let nblocks = in_f / 32;
        let scales = vec![0.01f32; out_f * nblocks];
        let quants = vec![0x88888888u32; out_f * (in_f / 8)]; // nibble 8 -> 0 after -8
        let x = vec![1.0f32; in_f];

        let xb = ctx.storage(&x);
        let sb = ctx.storage(&scales);
        let qb = ctx.storage_u32(&quants);

        // Warm up (pipeline build + first bind group), GPU-idle after.
        let warm = q4::gemv(&ctx, &xb, &sb, &qb, in_f, out_f);
        let _ = ctx.read(&warm, 1).await;
        ctx.clear_cache();

        // Time ITERS dispatches in one submit; divide by ITERS.
        ctx.begin_profile();
        let mut last = ctx.storage(&[0.0f32]);
        for _ in 0..ITERS {
            last = q4::gemv(&ctx, &xb, &sb, &qb, in_f, out_f);
        }
        let _ = ctx.read(&last, 1).await;
        let report = ctx.report_profile().await;
        let ms = report
            .iter()
            .find(|(name, _, _)| *name == "q4_gemv")
            .map(|(_, _, ms)| *ms)
            .unwrap_or(0.0);
        ctx.clear_cache();

        let us = ms * 1000.0 / ITERS as f64;
        let bytes = out_f * in_f / 2 + out_f * nblocks * 4;
        let gbs = bytes as f64 / (us * 1e-6) / 1e9;
        println!("{label:<18} {us:>9.1} {gbs:>10.1} {:>9}", bytes);
    }
}
