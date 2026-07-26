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
});
