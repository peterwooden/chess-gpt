const MAXIMUM_WASM_THREADS = 4;

export async function provisionOnnxRuntime({
  ort,
  runtimeUrl,
  fetchRuntime = globalThis.fetch,
}) {
  if (!ort?.env?.wasm) throw new Error("The arena's ONNX Runtime is unavailable.");

  const response = await fetchRuntime(runtimeUrl, {
    cache: "force-cache",
    credentials: "omit",
  });
  if (!response.ok) {
    throw new Error(`The arena could not load ONNX Runtime WASM (${response.status}).`);
  }
  const wasmBinary = await response.arrayBuffer();
  if (wasmBinary.byteLength === 0) {
    throw new Error("The arena received an empty ONNX Runtime WASM binary.");
  }
  ort.env.wasm.wasmBinary = wasmBinary;

  // Multi-threading needs SharedArrayBuffer, which only exists when the page is
  // cross-origin isolated (COOP/COEP — see build/cross-origin-isolation.mjs).
  // Without isolation ort-web silently falls back to one thread; pin 1 so the
  // fallback is explicit rather than a silent halving of every search.
  //
  // Four is the measured knee on the reference machine: 23.8 ms/position at 1
  // thread, 8.7 ms at 4, and back up to 10.6 ms at 8. Machines with fewer cores
  // take their core count instead, so a contestant never oversubscribes.
  const isolated = globalThis.crossOriginIsolated === true;
  const cores = globalThis.navigator?.hardwareConcurrency;
  ort.env.wasm.numThreads = isolated
    ? Math.max(1, Math.min(MAXIMUM_WASM_THREADS, cores ?? MAXIMUM_WASM_THREADS))
    : 1;
  console.info(
    `[arena] ONNX Runtime WASM threads: ${ort.env.wasm.numThreads} (crossOriginIsolated=${isolated})`,
  );
}
