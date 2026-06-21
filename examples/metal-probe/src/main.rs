//! Quantifies the wgpu-vs-MLX gap directly on Metal. Our wgpu engine pays
//! ~0.31 ms per dependent op because WebGPU forces a fresh command *encoder* per
//! compute pass. MLX issues many dispatches into ONE encoder. This measures that
//! exact difference with a trivial kernel:
//!
//!   A) one command buffer, a NEW encoder per dispatch  (== what wgpu does)
//!   B) one command buffer, ONE encoder, N dispatches   (== what MLX does)
//!
//!   cargo run --release   (from this dir)

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void inc(device float* y [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    y[i] = y[i] + 1.0;
}
"#;

fn main() {
    let device = Device::system_default().expect("no metal device");
    println!("device: {}", device.name());
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &CompileOptions::new())
        .unwrap();
    let func = lib.get_function("inc", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .unwrap();
    let buf = device.new_buffer(1024 * 4, MTLResourceOptions::StorageModeShared);
    let grid = MTLSize::new(1024, 1, 1);
    let tg = MTLSize::new(64, 1, 1);
    let n = 5000usize;

    // warm up
    let _ = bench(&queue, &pipeline, &buf, grid, tg, 10, true);
    let _ = bench(&queue, &pipeline, &buf, grid, tg, 10, false);

    let a = bench(&queue, &pipeline, &buf, grid, tg, n, true);
    let b = bench(&queue, &pipeline, &buf, grid, tg, n, false);

    println!("\n{n} dispatches of a trivial kernel:");
    report("A) new encoder per dispatch  (wgpu)", a, n);
    report("B) one encoder, N dispatches (MLX) ", b, n);
    println!(
        "\nMLX-style single encoder is {:.0}x cheaper per dispatch.",
        (a / n as f64) / (b / n as f64)
    );
    println!(
        "Extrapolated to ~420 ops/token: encoder churn ≈ {:.0} ms (wgpu) vs {:.1} ms (one encoder).",
        a / n as f64 * 420.0,
        b / n as f64 * 420.0
    );
}

fn bench(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    buf: &metal::Buffer,
    grid: MTLSize,
    tg: MTLSize,
    n: usize,
    new_encoder_each: bool,
) -> f64 {
    let t = Instant::now();
    let cmd = queue.new_command_buffer();
    if new_encoder_each {
        for _ in 0..n {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(buf), 0);
            enc.dispatch_threads(grid, tg);
            enc.end_encoding();
        }
    } else {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(buf), 0);
        for _ in 0..n {
            enc.dispatch_threads(grid, tg);
        }
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    t.elapsed().as_secs_f64() * 1000.0
}

fn report(name: &str, ms: f64, n: usize) {
    println!("  {name}: {ms:8.2} ms total  ({:.4} ms/dispatch)", ms / n as f64);
}
