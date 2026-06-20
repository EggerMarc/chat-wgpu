# wgpu-spike — WebGPU feasibility test

candle 0.10.2 has **no WebGPU backend** (its `Device` is `Cpu | Cuda | Metal`),
so getting browser-GPU inference means a real backend build. Before committing
to that, this spike answers: **how much faster is WebGPU than the CPU-wasm path
we have today?**

It runs an f32 matmul (the op that dominates LLM decode) as a WGSL compute
shader and compares it to a naive CPU matmul of the same size — natively
(Metal/Vulkan/DX) and in the browser (WebGPU). `bench()` is shared between both.

## Native (validates the wgpu code)

```bash
cargo run --release
```

On an M-series (Metal), GPU vs naive CPU:

| shape | GPU | speedup |
| --- | --- | --- |
| M=1, 1024×1024 (decode GEMV) | 0.46 ms | 5.8× |
| M=256, 1024×1024 (prefill GEMM) | 7.1 ms (76 GFLOP/s) | 49× |
| M=1, 4096×4096 (bigger GEMV) | 5.0 ms | 18× |

All with `max_abs_err = 0` (the WGSL result matches CPU exactly).

## Browser (the real target)

```bash
RUSTFLAGS="--cfg=web_sys_unstable_apis" \
  wasm-pack build --target web --out-dir web/pkg --release
python3 -m http.server -d web 8080      # open http://localhost:8080, click Run
```

Needs a WebGPU-capable browser (recent Chrome/Edge, or Safari Technology
Preview). The page benchmarks WGSL matmul vs CPU-wasm — and CPU-wasm is much
slower than native CPU, so the in-browser speedup is larger still.

## What the spike found

- WebGPU is reachable from our wasm stack and the matmul is correct + fast.
- **Constraint for a real port:** WebGPU caps `maxStorageBufferBindingSize`
  (often 128 MB). A full fp32 lm-head (vocab×dim ≈ 622 MB) won't fit one
  binding — it must be quantized/fp16 and/or tiled. Same goes for large weights.

Verdict: a WebGPU backend (Burn-wgpu, or hand-rolled WGSL) would dramatically
beat the CPU-wasm engine. This is the foundation to build it on.
