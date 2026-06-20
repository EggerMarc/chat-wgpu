// Vanilla UI: spawns the worker, auto-loads the model on entrance, renders the
// streaming chat. No framework, no bundler.

const GGUF_URL =
  "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf";
const TOKENIZER_URL =
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const chatEl = $("chat");
const promptEl = $("prompt");
const sendEl = $("send");

const worker = new Worker(new URL("./worker.js", import.meta.url), {
  type: "module",
});

let assistant = null; // the in-progress assistant bubble

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = role === "user" ? "user" : "bot";
  el.textContent = text;
  chatEl.appendChild(el);
  return el;
}

function setReady(ready) {
  promptEl.disabled = !ready;
  sendEl.disabled = !ready;
  promptEl.placeholder = ready ? "Ask something…" : "loading model…";
}

worker.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case "status":
      statusEl.textContent = m.detail ? `${m.status} — ${m.detail}` : m.status;
      break;
    case "progress":
      statusEl.textContent =
        `downloading ${m.label}: ${(m.received / 1e6).toFixed(0)}` +
        (m.total ? ` / ${(m.total / 1e6).toFixed(0)} MB` : " MB");
      break;
    case "ready":
      statusEl.textContent = "ready";
      setReady(true);
      break;
    case "token":
      if (assistant) assistant.textContent += m.piece;
      break;
    case "done":
      statusEl.textContent = `ready · ${m.tokens} tok · ${(m.tokens / m.secs).toFixed(1)} tok/s`;
      setReady(true);
      break;
    case "error":
      statusEl.textContent = "error: " + m.error;
      setReady(true);
      break;
  }
};

function send() {
  const prompt = promptEl.value.trim();
  if (!prompt || promptEl.disabled) return;
  bubble("user", prompt);
  assistant = bubble("assistant", "");
  promptEl.value = "";
  setReady(false);
  statusEl.textContent = "generating…";
  worker.postMessage({ type: "generate", prompt });
}

sendEl.addEventListener("click", send);
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});

// Auto-load on entrance.
worker.postMessage({ type: "load", ggufUrl: GGUF_URL, tokenizerUrl: TOKENIZER_URL });
