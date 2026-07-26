# ONNX browser-runtime smoke test

This fixture answers one infrastructure question: can a valid package initialize and run the ONNX Runtime Web instance supplied by the arena? It is not a trained chess model and is not tournament eligible.

## Minimal mechanism

The generated `model.onnx` is 140 bytes and contains one `Add` operation:

```text
scalar input + constant zero -> scalar output
```

The adapter creates a WASM inference session during `loadPackage`, runs that graph when asked for a move, and uses its zero output to select the first legal SAN move. This deliberately separates ONNX runtime initialization from model architecture, tokenization, chess logic, and sampling.

## Reproduce the package

```bash
uv run python -m chess_gpt.onnx_smoke \
  --output runs/onnx-runtime-smoke/browser
```

The public fixture is [`peterwooden/chess-gpt-onnx-smoke`](https://huggingface.co/peterwooden/chess-gpt-onnx-smoke/tree/d2a54a143350ec36ec52fc9a90c226ad48bf5b80). Load this immutable reference in the arena:

```text
peterwooden/chess-gpt-onnx-smoke@d2a54a143350ec36ec52fc9a90c226ad48bf5b80
```

## Original red result

On 27 July 2026, the public arena downloaded and verified the package, then failed while its pinned ONNX Runtime Web 1.27.0 tried to initialize:

```text
no available backend found. ERR: [wasm] RuntimeError: Aborted(Error: Package access to XMLHttpRequest is forbidden.). Build with -sASSERTIONS for more info.
```

The arena disabled `XMLHttpRequest` before calling the package entrypoint, while ONNX Runtime initialized its WASM execution engine lazily during `InferenceSession.create()`. The fixture therefore isolated runner-owned runtime provisioning from model complexity.

## Fixed result

The runner now loads its matching, content-addressed WASM binary, assigns the bytes to `ort.env.wasm.wasmBinary`, and only then restricts package capabilities. In a local production build, the same immutable Hugging Face package loaded and returned legal SAN `a3` without adding a runtime artifact to the submission.
