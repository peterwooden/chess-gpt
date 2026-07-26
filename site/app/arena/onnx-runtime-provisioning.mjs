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
}
