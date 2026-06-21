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
- [ ] **perf** — Qwen3-0.6B Q4_0 on Metal: **~2.6 tok/s** (was 1.8). Measured
      findings:
      - **command-buffer batching** (one submit/token vs ~250): *no change* —
        submit overhead wasn't the bottleneck. Kept anyway (cleaner, needed for
        browser).
      - **dedicated M=1 GEMV** decode path (no wasted threads, no integer
        division): **1.4×**, the win so far.
      - flat 1-thread-per-output (idx/n, idx%n): *regressed* — GPU integer
        division is expensive.
      - shared-memory A cache: *regressed* — 16 KB/workgroup tanks occupancy on
        the 151936-wide lm-head.
      - We're ~50× below memory-bandwidth peak → **latency/occupancy bound** on
        many small serial GEMVs, not bandwidth bound.
      - **fused QKV** (3 GEMVs → 1 wide GEMV of width q+2·kv, weights concat at
        load, q/k/v sliced out): **1.15×** → 2.9 tok/s. Numerically exact
        (random-weight argmax unchanged).
      - fused gate+up *matmul only*: neutral, reverted (slices cancel it).
      - fused **gate+up+swiglu** kernel (2 matmuls + activation in 1 pass, no
        slices, reads `x` once): numerically exact but *also perf-neutral*.
        **Kept** — strictly cleaner forward (MLP is 2 ops not 4).
      **Diagnosis (two independent confirmations):** batching and MLP fusion
      both neutral ⇒ dispatch/barrier overhead is *not* the bottleneck. We read
      ~2.4 GB f32 weights/token at ~7 GB/s effective (2–5% of peak) ⇒
      **memory-latency-bound**: the narrow layer GEMVs (512–3072 wide) lack
      enough in-flight parallelism to hide memory latency.
      - **q4-in-VRAM** (opt-in `--quantize`): weights re-quantized to Q4_0 at
        load and kept packed in VRAM; a q4 GEMV dequantizes inline (8 weights
        per 32-bit load). **~5.25 tok/s vs ~3.3 f32 (1.6×)**, rock-stable, **4×
        less VRAM** (2.4 GB → 0.4 GB), loads in 1.3 s. Coherent; kernel verified
        exact (1e-7 vs CPU dequant). Uniform `Proj { Dense | Quant }` so the
        forward is the same in both modes. This is also the **browser enabler**
        — 0.4 GB fits under WebGPU's buffer caps where 2.4 GB did not.
      - coalesced q4 GEMV (one workgroup/output, shared-mem reduction): also
        *neutral* (~5 tok/s).
      **Root cause (the wall):** ~565 GPU compute passes per token (28 layers ×
      ~20 ops + lm-head), and wgpu spends a fresh Metal encoder + barrier per
      pass (~0.35 ms). At 200 ms/token that *is* the cost — we're
      **per-pass-overhead bound, not compute-bound**. Explains why batching,
      fusion, q4-kernel-quality, and coalescing were all neutral: none cut the
      pass count enough. 80 tok/s (12.5 ms/token) needs ~30 passes/token, i.e.
      **whole-decoder-layer megakernels** (norm+QKV+attention+o+norm+MLP+residual
      in one dispatch) — the MLX/llama.cpp/web-llm approach. Large effort; wgpu's
      per-pass floor may still cap us below MLX's hand-tuned Metal.
      Net session: 1.8 → **~5 tok/s (q4)**, 4× less VRAM, browser-viable.

      **Root cause of the MLX gap (definitive):** removing 4 trivial-compute ops
      /layer (q/k-norm + rope) gave +20% → the cost is *per-GPU-op overhead*, not
      compute. Ruled out allocation (uniform cache + buffer pool both neutral).
      It's the **compute pass**: wgpu creates a separate Metal command encoder
      per dependent op (~0.31 ms), and WebGPU has no intra-pass barrier — so
      ~420 ops/token ≈ 130 ms of encoder churn. MLX issues many dispatches into
      ONE Metal encoder with cheap intra-encoder barriers; wgpu structurally
      can't. (Kept the uniform cache + buffer pool — neutral now, useful once
      passes drop.)
      **CORRECTION via direct Metal probe (`examples/metal-probe`):** native
      Metal records a dispatch in **2.8 µs** (new encoder each) / **1.1 µs** (one
      encoder) — so ~420 ops/token is **~1 ms** of Metal encoder cost, NOT the
      130 ms we see. Metal encoders and the MLX single-encoder trick are *not*
      the gap. The ~0.31 ms/op is **wgpu's own CPU-side per-dispatch overhead**
      (bind-group creation + validation + HAL translation) — ~100× Metal's. The
      GPU isn't the bottleneck; **wgpu's command recording is.**
      **Implications:**
      - The browser path (WebGPU/wgpu) cannot escape this — it's inherent to the
        abstraction. chat-wgpu is structurally *portable, not fastest*.
      - Native Metal (chat-mlx) records ~100× faster → that's why it hits 80.
        Reaching MLX-level means native Metal, i.e. chat-mlx's domain.
      - **Bind-group caching: tested, no gain.** Cached bind groups (arena +
        op-index cache, the "correct" WebGPU reuse pattern) → still ~5.1 tok/s.
        So bind-group creation is NOT the per-op cost. Ruled out: allocation,
        bind groups, encoder creation (probe: 2.8 µs).
      - **Refined root cause:** our ops are a *dependent chain*, so wgpu inserts
        a **hazard barrier between every compute pass** (flush/fence). That
        barrier is the ~0.3 ms/op, ~100× native Metal's (the probe's cheap
        dispatches had no barriers between them). The +20% from dropping 4
        ops/layer = 4 fewer barriers; caching didn't touch barriers → no gain.
      - **The one lever: fewer passes.** Independent ops can share a pass (no
        barrier between them); dependent chains must be *fused* into one kernel.
        Path: (a) pack independent ops (q/k/v matmuls, gate+up, q/k-norm, rope)
        into shared passes for ~1.3–1.5×; (b) **whole-layer megakernels**
        (norm+QKV+attn+o+norm+MLP+residual in one WGSL kernel) → ~420 passes to
        ~30, the only route to MLX-level. Both portable (Metal/Vulkan/DX12).
      - **Megakernel test (decisive): whole-MLP-in-one-dispatch is SLOWER** —
        f32 3.3 → 1.8 tok/s. Decode is one token, and a fused layer's dependency
        chain (gate/up→down) needs `workgroupBarrier`, forcing ONE workgroup =
        ~1% GPU occupancy. Killing 5 inter-pass barriers is dwarfed by the
        occupancy collapse. **Whole-layer megakernels don't work for decode.**
      - **The real strategy** (what MLX/llama.cpp actually do): keep matmuls as
        separate *full-occupancy* dispatches, but fuse the CHEAP ops (rmsnorm,
        rope, swiglu, residual, qk-norm) INTO the adjacent matmul kernels — they
        ride the matmul's threads, no occupancy loss, fewer passes.
      - **Honest ceiling:** even with all cheap ops fused, ~5 *dependent* matmul/
        attention passes/layer remain × 28 layers = ~140 passes × wgpu's ~0.3 ms
        barrier ≈ 42 ms → **~20–25 tok/s is the wgpu ceiling** for this model.
        80 tok/s (12.5 ms) needs the per-pass barrier ~100× cheaper, which only
        native Metal has. **wgpu cannot reach MLX speed here; ~20–25 tok/s (≈5×
        current) is the realistic target** via cheap-op-into-matmul fusion.
      - Kept: q4-in-VRAM, uniform cache, arena, bind-group cache; `mlp_mega` left
        as a documented negative experiment.

      ## ⚠️ DIRECT PROFILE OVERTURNS ALL OF THE ABOVE
      Split CPU-record vs GPU-execute per token (q4): **record(CPU) = 0.3 ms,
      read(GPU+sync) = ~85 ms.** We are **100% GPU-compute-bound.** Every CPU
      theory above (bind groups, inter-pass barriers, encoders, submit batching,
      "0.3 ms/op recording") was WRONG — CPU recording of all ~420 ops is 0.3 ms
      total. The bind-group cache / arena / batching optimized a non-problem.
      **The real bottleneck: the kernels run at ~3.5 GB/s — ~1–2% of the M1's
      200–400 GB/s.** That is the entire 16× gap to MLX.
      **Fix = efficient kernels, nothing else.** The q4 GEMV is the hot path
      (every projection + lm-head). Ours: 64-thread workgroup/output, 6-barrier
      shared-mem reduction, scalar nibble unpack. Needs: vectorized packed-weight
      loads (u32/vec4), **subgroup reductions** (`subgroupAdd`, 1 instr vs 6
      barriers), multiple outputs per simdgroup, high occupancy — i.e. MLX's
      `qmv` structure. Portable WGSL (subgroups on Metal/Vulkan/DX12). The
      megakernel being slower was the occupancy signal pointing here.

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
