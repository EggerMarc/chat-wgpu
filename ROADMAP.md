# chat-wgpu roadmap

The browser WebGPU engine, built bottom-up. Every kernel runs on the GPU and is
checked with **GPU known-answer tests** in `src/main.rs`
(`cargo run --bin verify --features verify`) — structural inputs whose outputs
are closed-form (identity → passthrough, ones → sum, rope@pos0 → identity). No
CPU reimplementation of any kernel: this is a pure wgpu provider.

## 1. Kernels

In dependency order:

- [x] **matmul (f32)** — `C[m,n] = A[m,k]·B[k,n]`. One thread per output element.
      Verified (ones→k, identity→passthrough, bit-exact).
- [x] **rmsnorm** — row-wise `x / sqrt(mean(x²)+eps) * weight`. One thread/row.
      Verified (ones→1/√(1+eps)).
- [x] **swiglu** — `silu(gate) * up`, elementwise. Verified vs the definition.
- [x] **rope** — rotate-half on q/k at a position offset (Qwen/Llama style).
      Verified (pos0→identity, 1-rad rotation).
- [ ] **softmax + attention** — single-query decode (GEMV-shaped scores → softmax
      → ·V), with GQA. Prefill variant later.
- [ ] **q4 dequant-matmul** — the hot path. Weights stay q4 in VRAM; dequantize
      per-tile inside a register-blocked GEMM (prefill) / GEMV (decode). This is
      the kernel that makes browser inference fast; biggest single effort.

Optimization pass (after correctness): the norm/rope/matmul kernels are
one-thread-per-row/element for clarity — replace with workgroup-reduction +
tiled versions for throughput once the model runs end-to-end.

Constraint surfaced by the spike: WebGPU caps `maxStorageBufferBindingSize`
(often 128 MB). The embedding / lm-head (vocab×dim) and large weights must be
quantized and/or tiled across bindings — fp32 won't fit one buffer.

## 2. Model

A model is a **trait** (`Model`): it loads its weights and composes its
architecture in `forward`. The forward *is* the architecture — read it and you
see the model; intermediates are local buffers tapped via `Hook` (mid-layer
sampling for meta-ML). No config-flag struct.

- [x] **`Model` trait + `Qwen3`** — load + forward composing the kernel building
      blocks, with intermediate `Hook` taps. Runs on GPU (random weights).
- [x] **Weight loaders** (`Weights` trait): hand-written **GGUF** (binary parse,
      metadata, dequant F32/F16/Q8_0/Q4_0/Q4_1/Q6_K, transpose) and
      **Safetensors** (JSON header, F32/F16/BF16, `config.json` metadata, GGUF↔HF
      name translation). Unit-tested. No candle.
- [x] **KV cache** — preallocated per-layer GPU buffers; cache-aware attention.
- [x] **full forward + generation** — embedding gather → N layers → final norm →
      lm-head → greedy argmax → feedback. `model::generate`.
- [x] **🎉 real Qwen3-0.6B generates coherent text** end-to-end on Metal
      (`cargo run --release --example generate`). ~2 tok/s on the naive kernels.
- [ ] GGUF **Q4_K / Q5_K** dequant — to load the common `*-Q4_K_M` files (Q6_K
      already done).
- [ ] **Llama / Gemma** model impls — prove the components + kernels compose
      (Llama = Qwen3 minus QK-Norm; Gemma swaps norm/activation + post-norms).
- [ ] **sampling** — temp/top-k/top-p, on-device argmax (avoid the per-token
      logits readback).
- [ ] **perf** — the kernels are one-thread-per-row/element for correctness;
      tiled GEMM + flash attention + q4-in-VRAM dequant-matmul are the levers
      (the spike measured ~5–49× headroom).

## 3. Browser API

- [ ] **`wasm-bindgen` `LocalChat`** — `new(ggufBytes, tokenizerBytes)` +
      streaming `generate(prompt, opts, onToken)`, mirroring chat-candle's wasm
      API so frontends are portable.
- [ ] **weight loading** — fetch + Cache API / IndexedDB; the GGUF arrives as
      bytes from JS.

## 4. Frontend examples (`examples/frontends/`)

- [x] **vanilla** — build-free plain HTML + ES-module JS reference: auto-load on
      entrance, Web-Worker streaming chat. Defines the worker + message protocol
      the framework wrappers reuse. (UI + worker written; runs once the wasm
      `LocalChat` API lands.)
- [ ] **react** — `useLocalChat()` hook.
- [ ] **vue** — `useLocalChat()` composable.
- [ ] **svelte** — a `localChat` store.

All run the model in a Web Worker (the wasm `generate` is synchronous) and share
the vanilla example's message/streaming contract; only the framework glue
differs.

## 5. Model families

Family-agnostic, like chat-mlx: variant-parameterized kernels + a `FamilySpec`
per family in `src/families/`, resolved from GGUF `general.architecture`. The
specs exist now (`from_arch` verified); each becomes "done" when the forward
loop runs that family end-to-end.

- [x] **spec layer** — `FamilySpec` + qwen3 / qwen2 / llama / gemma, arch
      resolver with Llama fallback. Norm + activation variants verified on GPU.
- [ ] **Qwen3** — first end-to-end target. Per-head QK-Norm, no QKV bias, tied
      embeddings. Drives the initial bring-up.
- [ ] **Qwen2 / Qwen2.5** — QKV bias, tied embeddings, no QK-Norm.
- [ ] **Llama 3 / MiniCPM** — GQA, SwiGLU, RoPE; the baseline + fallback.
- [ ] **Mistral** — Llama-shaped; sliding-window attention is the one addition.
- [ ] **Gemma 2/3** — exercises the `UnitShift` RMSNorm + `GeGLU` variants
      (both verified); plus sliding window + logit soft-capping in the loop.
- [ ] **Phi-3** — Llama-shaped; fused QKV / gate-up unpacking at load.

Per-family work is confined to: norm variant, MLP activation, QKV bias on/off,
QK-Norm on/off, attention masking (sliding window), logit soft-cap — all on the
`FamilySpec`, dispatching the right kernel variant. New families that fit the
existing variants are a ~15-line spec file; genuinely new behavior (a new norm,
a new activation) adds one branch to a kernel + one enum case.
