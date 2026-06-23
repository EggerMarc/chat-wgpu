//! Sustained q4_gemv loop for a clean Metal System Trace capture. Runs ONE
//! shape (lm_head: 1024→151936, the cleanest single-dispatch bandwidth signal)
//! in a tight loop so the GPU track is wall-to-wall q4_gemv — open the trace and
//! read the memory-stall / occupancy / bandwidth counters on that pass.
//!
//!   cargo build --release --example q4trace
//!   xcrun xctrace record --template 'Metal System Trace' \
//!     --launch -- ./target/release/examples/q4trace
//!   open *.trace

use chat_wgpu::context::GpuContext;
use chat_wgpu::kernels::q4;

const IN_F: usize = 1024;
const OUT_F: usize = 151936; // lm_head / vocab
const ITERS: u32 = 1000; // ~4.5 ms/pass ⇒ ~4.5 s of saturated GPU

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let ctx = GpuContext::new().await.expect("gpu");
    eprintln!("[q4trace] backend: {}  shape: {IN_F}->{OUT_F}  iters: {ITERS}", ctx.backend);

    let nblocks = IN_F / 32;
    let scales = vec![0.01f32; OUT_F * nblocks];
    let quants = vec![0x88888888u32; OUT_F * (IN_F / 8)];
    let x = vec![1.0f32; IN_F];
    let xb = ctx.storage(&x);
    let sb = ctx.storage(&scales);
    let qb = ctx.storage_u32(&quants);

    // Warm: build pipeline + bind group, then loop reusing the same arena slot
    // (reset_frame rewinds the cursor) so no per-iter allocation — the GPU stays
    // fed with back-to-back q4_gemv dispatches, flushed in batches.
    let mut last = q4::gemv(&ctx, &xb, &sb, &qb, IN_F, OUT_F);
    let _ = ctx.read(&last, 1).await;
    ctx.clear_cache();

    for i in 0..ITERS {
        ctx.reset_frame();
        last = q4::gemv(&ctx, &xb, &sb, &qb, IN_F, OUT_F);
        if i % 64 == 63 {
            ctx.flush(); // submit the batch; keep the queue saturated
        }
    }
    let _ = ctx.read(&last, 1).await; // drain
    eprintln!("[q4trace] done");
}
