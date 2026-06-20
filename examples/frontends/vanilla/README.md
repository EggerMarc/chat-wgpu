# chat-wgpu — vanilla example

Build-free browser chat: plain HTML + ES-module JS, no framework, no bundler.
The model auto-downloads on entrance and runs on WebGPU via chat-wgpu's wasm.

```bash
# build the wasm pkg into ./pkg (run from this dir)
RUSTFLAGS="--cfg=web_sys_unstable_apis" \
  wasm-pack build ../../.. --target web --out-dir examples/frontends/vanilla/pkg --release

# serve statically and open
python3 -m http.server 8080      # → http://localhost:8080
```

Needs a WebGPU-capable browser (recent Chrome/Edge, or Safari Technology
Preview). First load downloads ~400 MB of weights (then cached).

## Files

- `index.html` — markup + styles
- `app.js` — UI: spawns the worker, auto-loads, renders the streaming chat
- `worker.js` — wraps the wasm `LocalChat` off the main thread

> The `LocalChat` API lands with the engine forward loop (see
> [../../../ROADMAP.md](../../../ROADMAP.md)). This example is the reference
> integration, ready to run once it does.
