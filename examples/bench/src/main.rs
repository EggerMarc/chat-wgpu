//! Native runner — validates the wgpu path on the platform GPU (Metal here)
//! before shipping it to the browser. `cargo run --release`.

// The browser build has no `main`; the entry point is the wasm `bench` export.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    // Decode-shaped (M=1 GEMV) and prefill-shaped (M=256) matmuls at the model's
    // hidden size, plus a larger GEMV.
    let cases = [
        (1usize, 1024usize, 1024usize, 100usize), // decode GEMV @ hidden
        (256, 1024, 1024, 50),                    // prefill GEMM
        (1, 4096, 4096, 100),                     // bigger decode GEMV (64 MB B)
    ];
    for (m, k, n, iters) in cases {
        match pollster::block_on(wgpu_spike::bench(m, k, n, iters)) {
            Ok(r) => println!("{}", r.to_json()),
            Err(e) => eprintln!("error: {e}"),
        }
    }
}
