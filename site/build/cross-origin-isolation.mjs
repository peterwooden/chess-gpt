/**
 * Cross-origin isolation makes SharedArrayBuffer available, which ONNX Runtime
 * Web needs for WASM multi-threading in the arena's model worker.
 *
 * Every layer that can serve a response has to send these. `next.config.ts`
 * covers app-router responses, but a dedicated worker spawned from an isolated
 * document must have COEP on its own script response too — and the worker
 * script is served by Vite in development and by the Cloudflare ASSETS binding
 * in production, neither of which passes through `headers()`.
 *
 * `require-corp` rather than `credentialless`: every cross-origin load the site
 * makes is a CORS-mode `fetch()` to Hugging Face, which `require-corp` permits.
 */
export const CROSS_ORIGIN_ISOLATION_HEADERS = Object.freeze({
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
});

/** Vite plugin: isolate dev-server responses, including transformed modules. */
export function crossOriginIsolation() {
  const applyHeaders = (server) => {
    server.middlewares.use((_request, response, next) => {
      for (const [name, value] of Object.entries(CROSS_ORIGIN_ISOLATION_HEADERS)) {
        response.setHeader(name, value);
      }
      next();
    });
  };
  return {
    name: "chess-gpt-cross-origin-isolation",
    configureServer: applyHeaders,
    configurePreviewServer: applyHeaders,
  };
}
