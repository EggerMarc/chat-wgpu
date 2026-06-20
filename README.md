# chat-wgpu

A **clean-room WebGPU LLM inference engine** for the browser. Hand-rolled WGSL
compute kernels, GGUF weights, HF tokenizer — **no candle, no other ML
framework**. Runs natively (Metal / Vulkan / DX12 via wgpu) for development and
kernel verification, and in the browser (WebGPU via wasm) as the real target.

Sibling to [`chat-candle`](../chat-candle), which is the **native** chat-rs
provider (candle on CPU / Metal / CUDA). chat-candle owns server/desktop/CLI;
chat-wgpu owns the browser.

## Why hand-rolled

candle 0.10.2 has no WebGPU backend (its `Device` is `Cpu | Cuda | Metal`), and
its wasm path is CPU-only — correct but far too slow for interactive chat. A
`wgpu` spike (see `examples/bench`) measured the matmul that dominates decode at
**5–49× over CPU** on Metal; in the browser the gap over CPU-wasm is larger
still. So the browser path is its own engine, built on WebGPU from the metal up.

## Status

Foundation. The GPU context + f32 matmul kernel are in place and verified
against a CPU reference:

```bash
cargo run --bin verify --features verify     # → "all kernels verified ✅"
```

## Build-out (see ROADMAP.md)

Kernels are added in dependency order, each verified vs CPU before the next:

| kernel | status |
| --- | --- |
| matmul (f32) | ✅ verified |
| rmsnorm | ✅ verified |
| swiglu / silu | ✅ verified |
| rope | ✅ verified |
| softmax + attention | todo |
| **q4 dequant-matmul** (the hot path) | todo |

Kernels run on the GPU and are checked with **GPU known-answer tests** (no CPU
reimplementation — pure wgpu). Then: attention → q4 dequant-matmul → GGUF loader
(hand-parsed) → Qwen3 forward loop → `wasm-bindgen` `LocalChat` API. Target model
families are in ROADMAP.md.

## Layout

```
src/
  context.rs        GpuContext: device/queue, pipeline cache, buffer + dispatch + readback
  kernels/          WGSL kernels (*.wgsl) + host dispatch wrappers + CPU oracles
  main.rs           `verify` — diffs every kernel vs CPU
  wasm.rs           browser entry (LocalChat API — pending the forward loop)
examples/
  bench/            WebGPU matmul benchmark (GFLOP/s, GPU vs CPU), native + browser page
  frontends/        browser examples — vanilla (working); react/vue/svelte on the roadmap
```

## Reference

The kernel design follows the patterns in production browser-LLM WebGPU
implementations (register-blocked GEMM tiles for prefill, GEMV for decode, q4
dequant fused into the matmul, subgroup reductions). Ours are reimplemented
against the Qwen3 layout and verified independently against the CPU oracle.
