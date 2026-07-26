export async function loadPackage({ artifacts, ort }) {
  const modelBytes = artifacts.get("model");
  if (!(modelBytes instanceof Uint8Array)) {
    throw new Error("The ONNX smoke package requires a Uint8Array artifact named model.");
  }
  if (!ort?.InferenceSession || !ort?.Tensor || !ort?.env?.wasm) {
    throw new Error("The runner did not provide a compatible ONNX Runtime Web instance.");
  }

  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
  });

  return {
    async newGame() {
      return {
        async chooseMove({ legalMoves }) {
          if (!Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires at least one legal SAN move.");
          }
          const input = new ort.Tensor("float32", Float32Array.of(0), [1]);
          const result = await session.run({ input });
          const index = Math.abs(Math.trunc(Number(result.output.data[0]))) % legalMoves.length;
          return legalMoves[index];
        },
        async dispose() {},
      };
    },
    async dispose() {
      await session.release();
    },
  };
}
