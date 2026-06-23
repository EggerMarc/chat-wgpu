# Profiling & benchmarking chat-wgpu

Decode is **GPU-bound** (CPU records a token in ~0.3 ms; the GPU then runs it in
~85–110 ms). So a CPU sampling profiler won't find the bottleneck — you need
*GPU* timing. Three tools, in order of usefulness here:

1. **Built-in per-kernel GPU profiler** (timestamp queries) — where GPU time goes, by kernel.
2. **Kernel microbenchmark** (`examples/bench.rs`) — isolate one kernel's µs / GB·s.
3. **Xcode Instruments → Metal System Trace** — a visual GPU timeline/flamegraph.
   (CPU flamegraph via `samply` is in §4, but expect it to show the CPU side is idle.)

Every GPU dispatch is now labeled with its kernel name (`q4_gemv`, `attention`,
`rmsnorm`, …), so the names show up in all three tools.

---

## 1. Per-kernel GPU breakdown (start here)

Set `WGPU_PROFILE=<decode-token-index>` to capture one decode token with GPU
timestamps and print an aggregated table:

```bash
WGPU_PROFILE=2 cargo gen \
  /tmp/qwen3-test/qwen3-0.6b-q4_0.gguf /tmp/qwen3-test/tokenizer.json \
  "The capital of France is" 8 --quantize
```

(`cargo gen` is an alias for `cargo run --release --example generate --`, so the
args follow directly — no extra `--`. See `.cargo/config.toml`.)

Output (real Qwen3-0.6B, M1):

```
┌─ GPU per-kernel breakdown (one decode token) ─────────────
│ kernel            passes   total ms   µs/pass      %
├───────────────────────────────────────────────────────────
│ attention             28     33.481    1195.8  38.7%
│ q4_gemv              197     28.332     143.8  32.8%
│ rmsnorm              113     21.097     186.7  24.4%
│ rope                  56      2.237      39.9   2.6%
│ ewise_add             57      1.145      20.1   1.3%
│ swiglu                28      0.167       6.0   0.2%
│ TOTAL                        86.459
└─ note: buffer copies (KV write, embed gather) are untimed ─
```

How to read it:
- **`µs/pass`** is the per-dispatch cost. `rmsnorm` does almost no work yet costs
  ~187 µs/pass — that's the *fixed per-compute-pass overhead* (≈ what `bench.rs`
  measures as the floor). 113 such passes ⇒ 21 ms of mostly-overhead.
- **`total ms`** is what actually adds up. `attention` is the biggest single
  cost; `q4_gemv` (every projection + lm-head) is 197 passes.
- The kernel **TOTAL (~86 ms)** is below the measured `read(GPU+sync)` (~107 ms).
  The gap is untimed buffer copies (KV/embed) + submit/sync + inter-pass gaps.
- Profiling adds a resolve+map to *that one token*, so its `read` number is
  inflated — use an **un-profiled** token for wall-clock, the table for the split.

Implementation: `src/profile.rs` (the `Profiler`), wired through
`GpuContext::{begin_profile, report_profile}` and the labeled `dispatch` in
`src/context.rs`. Needs `wgpu::Features::TIMESTAMP_QUERY` (Metal/Vulkan/DX12 have it).

---

## 2. Kernel microbenchmark — `examples/bench.rs`

Isolates a single kernel in a tight loop (no model) to get clean µs and GB/s,
and separates fixed overhead from real compute:

```bash
cargo gpubench        # alias for: cargo run --release --example bench
```

It reports the q4 GEMV (subgroup), the f32 matmul, the cached-bind-group cost,
and the per-dispatch floor vs work size. Edit the shapes at the top to probe a
specific kernel. Use this to A/B a kernel change without the whole model.

---

## 3. GPU timeline / flamegraph — Xcode Instruments

The real visual "what's the GPU doing" view. No code; passes show up by label.

```bash
cargo build --release --example generate    # build first
xcrun xctrace record --template 'Metal System Trace' \
  --launch -- ./target/release/examples/generate \
  /tmp/qwen3-test/qwen3-0.6b-q4_0.gguf /tmp/qwen3-test/tokenizer.json "Hi" 8 --quantize
open *.trace          # opens in Instruments
```

In Instruments: the **GPU track** shows each compute pass (named), its duration,
and gaps/bubbles between passes (the serialization cost). This is the most
direct way to *see* dispatch overhead and occupancy. You can also capture a
single GPU frame from Xcode (Debug → Capture GPU Frame) for per-kernel occupancy
and bandwidth counters.

---

## 4. CPU flamegraph — `samply` (expect it to be boring)

The CPU side is ~0.3 ms/token, so this mostly *confirms* the host isn't the
bottleneck. Still useful to verify recording cost and model-load time.

```bash
cargo install samply                        # one-time (already installed here)
cargo build --release --example generate
samply record ./target/release/examples/generate \
  /tmp/qwen3-test/qwen3-0.6b-q4_0.gguf /tmp/qwen3-test/tokenizer.json "Hi" 64 --quantize
# opens an interactive flamegraph in the browser
```

(`samply` is preferred over `cargo flamegraph` on macOS — no `sudo`/dtrace
hassle, better symbolication.)

---

## Current snapshot (M1, Qwen3-0.6B q4)

- ~85 ms GPU / decode token ⇒ ~6 tok/s. MLX on the same model+HW ≈ 80 tok/s.
- Cost splits ~3 ways: **attention (~39%)**, **q4_gemv matmuls (~33%)**,
  **rmsnorm overhead (~24%)**.
- `rmsnorm`/`rope`/`ewise_add` are ~187/40/20 µs *per pass* for near-zero work —
  the per-dispatch floor. ~560 dispatches/token, mostly a serial chain.

Two independent levers the numbers point at: make the **attention** kernel
faster (biggest single item), and **cut dispatch count** (fuse the cheap ops
into matmuls) to reclaim the rmsnorm/rope/add overhead.
