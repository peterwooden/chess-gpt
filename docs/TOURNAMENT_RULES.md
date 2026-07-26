# Tournament rules

- **Submission-size limit:** The complete manifest-referenced model package may be at most 100 MB in its canonical uncompressed form, including its manifest, entrypoint, model artifacts, vocabulary, configuration, and model-specific constants, but excluding the arena-provided runtime.
- **Shared dataset:** Everyone must train only on the [January–March 2026 Lichess standard rated games](https://database.lichess.org/) and use April 2026 only for validation, with the exact files and checksums fixed in [`data/dataset.toml`](../data/dataset.toml).
- **Training-compute limit:** Producing a submitted checkpoint may use at most $10^{18}$ training FLOPs (1 exaFLOP) across its entire training lineage, measured by the same agreed profiler regardless of local or cloud hardware.
- **Inference interface:** Every submission must implement the public Hugging Face package and SAN-move interface specified in the technical appendix below.
- **Tournament winner:** Each pair of models plays 100 games from 50 openings with colors reversed on the same runner; wins score 1 point, draws score ½ point, and losses score 0, so the model with the highest total score wins, while a first-place tie is settled by additional color-reversed opening pairs until one model leads after a complete pair, with all checkpoints, code, data, and configurations frozen before any openings are revealed.

## Technical appendix: model package v1

This appendix is normative: `MUST`, `MUST NOT`, and `MAY` describe the tournament interface.

### Public package

The submission MUST be a public Hugging Face model repository containing `browser/manifest.json`. The arena downloads that manifest and every file it declares once, verifies them, and runs the same package in casual games and tournament games.

`browser/manifest.json` MUST have this shape:

```json
{
  "schema": "chess-gpt-package-v1",
  "name": "example-model",
  "entrypoint": {
    "path": "entry.js",
    "bytes": 12345,
    "sha256": "FULL_LOWERCASE_SHA256_HEX"
  },
  "artifacts": {
    "model": {
      "path": "model.onnx",
      "bytes": 90000000,
      "sha256": "FULL_LOWERCASE_SHA256_HEX"
    },
    "vocabulary": {
      "path": "vocabulary.json",
      "bytes": 24576,
      "sha256": "FULL_LOWERCASE_SHA256_HEX"
    }
  },
  "config": {}
}
```

The entrypoint and artifact paths MUST be unique relative paths beneath `browser/`, MUST NOT contain `..`, and MUST resolve inside the same Hugging Face repository. `bytes` MUST equal the uncompressed file size and `sha256` MUST be the file's lowercase SHA-256 digest. The manifest, entrypoint, and all declared artifacts together MUST total no more than `100,000,000` bytes; the manifest counts by its downloaded byte length and does not hash itself. Artifact names and `config` contents are submission-defined, and no undeclared model-specific file or dependency may be loaded.

The author MAY maintain any source layout, but the submitted entrypoint MUST be one self-contained JavaScript module. The runner supplies its pinned ONNX Runtime Web instance as `ort`; that runtime is not part of the submission-size total.

### JavaScript interface

The entrypoint MUST export `loadPackage` with this interface:

```ts
export async function loadPackage(context: {
  artifacts: ReadonlyMap<string, Uint8Array>;
  config: unknown;
  ort: unknown;
}): Promise<{
  newGame(context: { random(): number }): Promise<{
    chooseMove(input: {
      history: readonly string[];
      legalMoves: readonly string[];
    }): Promise<string>;
    dispose(): Promise<void>;
  }>;
  dispose(): Promise<void>;
}>;
```

The runner calls `loadPackage` once per loaded package, `newGame` once per game, and `chooseMove` once whenever that package is to move. `history` contains the complete game history as canonical SAN strings and `legalMoves` contains every legal canonical SAN output for the current position. `chooseMove` MUST return exactly one string that is an exact member of `legalMoves`; resignation, draw offers, annotations, and additional text are not moves.

Each game MUST receive fresh logical adapter state, although an immutable model session MAY remain cached between games. A submission MUST use the supplied deterministic `random()` for stochastic choices and MUST NOT read opponent identity, rating, persistent storage, or state from another game. Tournament inference settings MUST be fixed in the submitted package.

### Runner and failures

The entrypoint runs in a dedicated Web Worker so the runner can isolate lifecycle and terminate failures; participants are trusted, so this worker is not treated as a hostile-code security boundary. Contestant code is forbidden from initiating network access after the declared package files load, although the arena-provided runtime may load its own same-origin assets. Any load error, exception, worker crash, non-string result, invalid SAN, or illegal move is an immediate game loss with no retry or fallback move.

Version 1 has no per-move inference-time limit. A later rules revision MAY introduce one before a future tournament.
