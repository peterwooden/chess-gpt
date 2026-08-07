import assert from "node:assert/strict";
import test from "node:test";

import { provisionOnnxRuntime } from "../app/arena/onnx-runtime-provisioning.mjs";

test("the runner supplies ONNX Runtime with its own WASM binary", async () => {
  const expected = Uint8Array.from([0x00, 0x61, 0x73, 0x6d]);
  const ort = { env: { wasm: {} } };
  const requests = [];

  await provisionOnnxRuntime({
    ort,
    runtimeUrl: "https://arena.example/assets/runtime.wasm",
    fetchRuntime: async (url) => {
      requests.push(url);
      return new Response(expected, { status: 200 });
    },
  });

  assert.deepEqual(requests, ["https://arena.example/assets/runtime.wasm"]);
  assert.deepEqual(new Uint8Array(ort.env.wasm.wasmBinary), expected);
  assert.equal(ort.env.wasm.numThreads, 1);
});

async function provisionIsolated(t, cores) {
  const ort = { env: { wasm: {} } };
  const originalNavigator = globalThis.navigator;
  globalThis.crossOriginIsolated = true;
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { hardwareConcurrency: cores },
  });
  t.after(() => {
    delete globalThis.crossOriginIsolated;
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: originalNavigator,
    });
  });

  await provisionOnnxRuntime({
    ort,
    runtimeUrl: "https://arena.example/assets/runtime.wasm",
    fetchRuntime: async () =>
      new Response(Uint8Array.from([0x00, 0x61, 0x73, 0x6d]), { status: 200 }),
  });
  return ort;
}

test("the runner enables four WASM threads under cross-origin isolation", async (t) => {
  const ort = await provisionIsolated(t, 10);
  assert.equal(ort.env.wasm.numThreads, 4);
});

test("the runner never asks for more threads than the machine has cores", async (t) => {
  const ort = await provisionIsolated(t, 2);
  assert.equal(ort.env.wasm.numThreads, 2);
});
