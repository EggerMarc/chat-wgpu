// Build-free Web Worker around the chat-wgpu wasm `LocalChat`. Plain ES module
// — no bundler. `../pkg` is the `wasm-pack build --target web` output, which
// browsers load directly.
//
// Runs the model off the main thread (the wasm `generate` is synchronous) and
// streams tokens back. Protocol (worker <-> UI):
//   in:  { type:"load", ggufUrl, tokenizerUrl } | { type:"generate", prompt, maxTokens?, temperature? }
//   out: { type:"status"|"progress"|"ready"|"token"|"done"|"error", ... }

import init, { LocalChat } from "../pkg/chat_wgpu.js";

const CACHE = "chat-wgpu-v1";
let chat = null;

const post = (m) => self.postMessage(m);

async function fetchBytes(url, label) {
  const cache = await caches.open(CACHE);
  let resp = await cache.match(url);
  if (!resp) {
    const net = await fetch(url);
    if (!net.ok) throw new Error(`${label}: HTTP ${net.status}`);
    await cache.put(url, net.clone());
    resp = net;
  }
  const total = Number(resp.headers.get("content-length")) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    post({ type: "progress", label, received, total });
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      post({ type: "status", status: "loading", detail: "init wasm + WebGPU" });
      await init();
      const tok = await fetchBytes(msg.tokenizerUrl, "tokenizer");
      const gguf = await fetchBytes(msg.ggufUrl, "weights");
      post({ type: "status", status: "loading", detail: "building model" });
      chat = await LocalChat.create(gguf, tok); // async: requests the GPU device
      post({ type: "ready" });
    } else if (msg.type === "generate") {
      if (!chat) throw new Error("model not loaded");
      post({ type: "status", status: "generating" });
      const t0 = performance.now();
      let n = 0;
      await chat.generate(
        msg.prompt,
        undefined,
        msg.maxTokens ?? 256,
        msg.temperature ?? 0.7,
        (piece) => {
          n++;
          post({ type: "token", piece });
        },
      );
      post({ type: "done", tokens: n, secs: (performance.now() - t0) / 1000 });
    }
  } catch (err) {
    post({ type: "error", error: String(err) });
  }
};
