# Arena model requirements

To run a model in the browser arena, publish it in this format; the [full browser model contract](BROWSER_MODEL_CONTRACT.md) contains the manifest schema and details.

- **Use a public Hugging Face model repository:** the arena downloads files directly from `huggingface.co` without credentials, so private or gated repositories will not load.
- **Export a browser-ready ONNX model:** place `model.onnx`, `vocabulary.json`, and `manifest.json` under `browser/`; repository Python code is never downloaded or executed.
- **Tokenize whole SAN moves:** each input history item and each predicted move must occupy one vocabulary entry, including notation such as `Nf3`, `Rae1`, `O-O`, `e8=Q+`, and `Qh7#`; include a BOS token so the model can play White's first move.
- **Match the tensor interface:** accept `int64` `input_ids` shaped `[1, sequence]` plus an optional same-shaped `attention_mask`, and return logits whose final dimension equals the vocabulary size; the arena reads the final position, masks illegal moves, and chooses the highest-scoring legal SAN move.
- **Declare and hash every artifact:** `manifest.json` must use schema `chess-gpt-browser-v1`, runtime `onnx-next-san`, name the tensors and context length, and contain each artifact's relative path and full SHA-256 digest.
- **Keep each download below 150 MB:** this is the arena's per-artifact browser safety limit; tournament submissions must also satisfy the separate 100 MB learned-state rule.
- **Share a pinned reference:** after uploading, give players `owner/repository@<40-character-commit-sha>` so everyone loads the same immutable model; loading it successfully in the arena is the final compatibility check.
