# Browser model contract

The browser arena accepts a Hugging Face model reference, downloads its artifacts directly to the user's device, and asks it one question at a time:

> Given the complete move history in standard algebraic notation (SAN), choose one legal SAN move.

The v1 contract is intentionally narrow. It makes models from different experiments interchangeable without downloading or executing arbitrary repository code.

## Accepted references

- `owner/repository@revision` resolves to `browser/manifest.json` at that revision.
- A Hugging Face repository, `tree`, `blob`, or `resolve` URL is normalized to a downloadable artifact.
- A direct `model.json.gz` URL loads this repository's count-based SAN n-gram checkpoint format.

Use a full 40-character Git commit revision for reproducible experiments and tournament submissions. Branches such as `main` are accepted for exploration but are labelled mutable in the arena.

## Repository layout

```text
browser/
├── manifest.json
├── model.onnx
└── vocabulary.json
```

`browser/manifest.json` uses this schema:

```json
{
  "schema": "chess-gpt-browser-v1",
  "name": "experiment-0042",
  "runtime": "onnx-next-san",
  "context_length": 256,
  "model": {
    "path": "model.onnx",
    "sha256": "FULL_LOWERCASE_SHA256_HEX",
    "bytes": 73400320
  },
  "vocabulary": {
    "path": "vocabulary.json",
    "sha256": "FULL_LOWERCASE_SHA256_HEX",
    "bytes": 24576,
    "bos_token": "<BOS>",
    "unknown_token": "<UNK>"
  },
  "inputs": {
    "input_ids": "input_ids",
    "attention_mask": "attention_mask"
  },
  "output": {
    "logits": "logits"
  }
}
```

`bytes` is informative and optional. `sha256` is required and is verified after download, before the artifact is used. Artifact paths must be relative to the manifest and remain inside the same Hugging Face repository.

## Vocabulary and tensors

`vocabulary.json` is either a JSON array of SAN strings or an object with a `tokens` array. Special tokens named by the manifest must also appear in that array. Every SAN move that the model can emit occupies one vocabulary position.

The ONNX graph receives `input_ids` with shape `[1, sequence]` and data type `int64`. If declared, `attention_mask` has the same shape and type. The arena prepends the beginning-of-sequence token, truncates from the left to `context_length`, and expects `logits` whose final dimension equals the vocabulary size. It reads the final sequence position, masks out illegal SAN moves, and chooses the legal move with the highest logit.

The arena uses WebGPU when the browser exposes it and otherwise falls back to WebAssembly. The interface stays the same regardless of the transformer's internal depth, width, attention design, or training method.

## Safety and failure behaviour

- Only `huggingface.co` artifacts are fetched; credentials are never sent.
- The browser never executes code from a model repository.
- Each downloaded artifact has a 150 MB safety limit.
- Manifest-declared model and vocabulary hashes must match exactly.
- A missing vocabulary move, mismatched tensor name, wrong logits shape, invalid SAN output, or corrupt artifact stops the game with a visible error.
- If a valid model assigns no score to any legal SAN token, the arena uses a deterministic legal fallback so exploratory games can continue.

Architectures that cannot expose final logits over a single SAN vocabulary need a future, explicitly versioned contract. They should not work around v1 by shipping remote scripts.
