// Tiny static dev server (Bun). Serves this directory — `Bun.file` sets the
// right content-type, including `application/wasm` so `instantiateStreaming`
// works. No bundler, no deps.

const PORT = Number(Bun.env.PORT ?? 8080);

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname === "/" ? "/index.html" : url.pathname;
    const file = Bun.file("." + path);
    if (await file.exists()) {
      return new Response(file, {
        // no-store keeps JS/HTML edits live during dev (weights are cached by
        // the worker's own Cache API, so this doesn't re-download the model)
        headers: { "cache-control": "no-store" },
      });
    }
    return new Response("not found", { status: 404 });
  },
});

console.log(`chat-wgpu vanilla → http://localhost:${server.port}`);
