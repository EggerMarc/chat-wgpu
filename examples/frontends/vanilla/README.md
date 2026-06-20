# chat-wgpu — vanilla example

Browser chat: plain HTML + ES-module JS, no framework, no bundler. Qwen3-0.6B
auto-downloads on entrance and runs on **WebGPU** via chat-wgpu's wasm — the same
engine that runs natively.

```bash
bun run dev      # builds the wasm pkg, then serves → http://localhost:8080
```

`bun run dev` = `build:wasm` + a tiny Bun static server (`server.ts`). After the
first build, iterate on the JS with `bun run serve` (skips the wasm rebuild), or
rebuild the engine with `bun run build:wasm`. (No `bun install` — zero deps.)

Needs a WebGPU-capable browser (recent Chrome/Edge, or Safari Technology
Preview). First load downloads ~360 MB of Q4_0 weights (then cached).

## Heads-up: memory

Weights dequantize to **f32 in VRAM** (~2.4 GB for 0.6B), and the embedding /
lm-head exceed WebGPU's `maxStorageBufferBindingSize` on many devices. We request
the adapter's max limits, so this runs on a **strong desktop GPU** but will fail
on integrated/mobile. The lightweight path — keeping weights q4-packed in VRAM
(a dequant-matmul kernel) — is the perf milestone in
[../../../ROADMAP.md](../../../ROADMAP.md). It's also slow (~1–2 tok/s) on the
naive kernels; the demo caps generation at 48 tokens.

## Files

- `index.html` — markup + styles
- `app.js` — UI: spawns the worker, auto-loads, renders the streaming chat
- `worker.js` — wraps the wasm `LocalChat` off the main thread (model load +
  streaming generation), so the UI never freezes
- `server.ts` — tiny Bun static server (correct `application/wasm` MIME)
- `package.json` — `dev` / `build:wasm` / `serve` scripts
- `pkg/` — `wasm-pack` output (gitignored)
