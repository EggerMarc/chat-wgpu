# chat-wgpu frontend examples

Putting a local WebGPU LLM one import away in the browser.

- **`vanilla/`** — the working reference. Build-free plain HTML + ES-module JS:
  auto-loads the model on entrance and streams chat over WebGPU. No framework,
  no bundler. Start here.

Framework wrappers (a small idiomatic hook/composable/store per stack, sharing
the vanilla example's Web-Worker + message protocol) are on the roadmap:

- **react** — `useLocalChat()` hook — *planned*
- **vue** — `useLocalChat()` composable — *planned*
- **svelte** — `localChat` store — *planned*

See [../../ROADMAP.md](../../ROADMAP.md). They all wrap the same worker contract
the vanilla example already defines; only the reactive glue differs.
