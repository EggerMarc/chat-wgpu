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

- [ ] **GGUF loader** — hand-parse the GGUF container (header, metadata,
      tensor table) into typed weights; architecture-agnostic via
      `general.architecture` (like chat-candle's loader). No candle dependency.
- [ ] **Qwen3 forward loop** — embed → N decoder layers (rmsnorm, attention with
      KV cache, swiglu) → final norm → lm-head → sample. Verified end-to-end
      against chat-candle's CPU output for the same prompt/weights.
- [ ] **KV cache** — preallocated per-layer GPU buffers.
- [ ] **sampling** — greedy + temp/top-k/top-p (on-device argmax where possible).

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

Architecture is config-driven from GGUF metadata (`general.architecture`), same
as chat-candle — no per-family source files. The current kernel set (plain
RMSNorm, rotate-half RoPE, SwiGLU, GQA) already covers the Llama/Qwen line; the
others are small, isolated kernel additions.

- [ ] **Qwen3** — first target. Per-head QK-Norm (extra RMSNorm on q/k), no QKV
      bias, tied embeddings. Drives the initial bring-up.
- [ ] **Qwen2 / Qwen2.5** — QKV bias, tied embeddings, no QK-Norm.
- [ ] **Llama 3 / MiniCPM** — GQA, SwiGLU, RoPE; bias per `attention_bias`.
- [ ] **Mistral** — Llama-shaped; sliding-window attention is the one addition.
- [ ] **Gemma 2/3** — needs the `(1 + weight)` RMSNorm variant, `gelu`/`geglu`
      MLP (vs SwiGLU), and logit soft-capping. A few gated kernel variants.
- [ ] **Phi-3** — Llama-shaped with fused QKV / gate-up packing.

Per-family work is confined to: which norm variant, which MLP activation, QKV
bias on/off, QK-Norm on/off, and any attention masking (sliding window) — all
selected from metadata at load, dispatching the right kernel variant.
