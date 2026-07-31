# Tournament rules

- **Submission-size limit:** The complete manifest-referenced model package may be at most 100 MB in its canonical uncompressed form, including its manifest, entrypoint, model artifacts, vocabulary, configuration, and model-specific constants, but excluding the arena-provided runtime.
- **Shared dataset:** Everyone must train only on the [January–March 2026 Lichess standard rated games](https://database.lichess.org/) and use April 2026 only for validation, with the exact files and checksums fixed in [`data/dataset.toml`](../data/dataset.toml).
- **Training-compute limit:** Producing a submitted checkpoint may use at most $10^{18}$ training FLOPs (1 exaFLOP) across its entire training lineage, measured by the same agreed profiler regardless of local or cloud hardware.
- **Evaluation:** Agreed 2026-07-30: evaluation is unrestricted — competitors may evaluate their models with any tools, engines, or data (for example Stockfish). Training restrictions above still apply to producing the submitted checkpoint.
- **Inference interface:** Every submission must implement the public Hugging Face package and SAN-move interface specified in the technical appendix below.
- **Entries and nomination:** Agreed 2026-07-31. A competitor may publish and register as many model versions as they want. Each tournament is entered separately, and the tournament that decides the title is entered by exactly one nominated model version per competitor. Other tournaments may be run over any set of registered versions for interest, but cannot decide the title.
- **Match protocol:** Agreed 2026-07-31, superseding the earlier opening-book protocol. Every game starts from the standard chess starting position. No opening book, no externally supplied position, and no runner-injected move is used, so opening play is part of what the tournament measures. Variation between games of the same pairing therefore comes only from stochasticity a submission chooses to implement, using the supplied deterministic `random()`.
- **Tournament winner:** Each tournament is a round robin. Every pair of entered models plays a configured number of games with colors alternating, on one pinned runner. Wins score 1 point, draws score ½ point, and losses score 0, and the model with the highest total score wins. Models tied on points **share the title**; no additional games are played to break a tie. All checkpoints, code, data, and configurations are frozen when the tournament leaves registration.
- **Game length:** Agreed 2026-07-31. A tournament configures a maximum ply count. A game reaching it is a draw with termination `max_plies`. All other terminations are the ordinary rules of chess, including stalemate, threefold repetition, the fifty-move rule, and insufficient material.
- **Per-move time limit:** Agreed 2026-07-31, replacing "version 1 has no per-move inference-time limit". Each tournament configures a per-move wall-clock limit, which is supplied to the submission so it may budget its own search. Because results now depend on the clock, every game of a tournament MUST run on one pinned machine, and that machine MUST run the games sequentially and remain otherwise idle.

## Technical appendix

### Agreed training FLOP profiler v1

Ratified by all competitors on 2026-07-29, the shared profiler counts dense training operations independently of hardware:

- one multiply-add is two FLOPs;
- profile all dense matrix multiplications executed by one forward pass, including every evaluated mixture-of-experts branch;
- count training as three times those forward FLOPs: one forward pass plus twice the forward cost for backward computation;
- multiply by the actual number of examples processed, not the planned dataset size; and
- add the recorded FLOPs of every parent checkpoint in the submitted checkpoint's lineage.

For the repository's 65-token board Transformer, [`profiled_training_flops`](../src/chess_gpt/snapshot_training.py) is the executable reference formula. Embedding lookup, normalization, activation, optimizer, and other non-matrix operations are excluded consistently for every competitor. A run record MUST include the profiler version, actual processed examples, actual run FLOPs, prior-lineage FLOPs, and the parent artifact digest whenever prior-lineage FLOPs are nonzero.

### Model package v1

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

The author MAY maintain any source layout, but the submitted entrypoint MUST be one self-contained JavaScript module. The runner supplies ONNX Runtime Web 1.27.0 as `ort`, including its matching WASM binary; neither counts toward the submission-size limit.

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
      moveTimeLimitMs: number;
    }): Promise<string>;
    dispose(): Promise<void>;
  }>;
  dispose(): Promise<void>;
}>;
```

The runner calls `loadPackage` once per loaded package, `newGame` once per game, and `chooseMove` once whenever that package is to move. `history` contains the complete game history as canonical SAN strings and `legalMoves` contains every legal canonical SAN output for the current position. `chooseMove` MUST return exactly one string that is an exact member of `legalMoves`; resignation, draw offers, annotations, and additional text are not moves.

`moveTimeLimitMs` was added on 2026-07-31 and is the wall-clock budget in milliseconds for this move. A submission MAY use it to decide how much search to perform, and MAY ignore it. Because it is an additional input property, packages published before this revision remain valid and unchanged. The budget is advisory to the submission but enforced by the runner, so a submission that overruns forfeits the game as described below.

Each game MUST receive fresh logical adapter state, although an immutable model session MAY remain cached between games. A submission MUST use the supplied deterministic `random()` for stochastic choices and MUST NOT read opponent identity, rating, persistent storage, or state from another game. Apart from adapting search to `moveTimeLimitMs`, tournament inference settings MUST be fixed in the submitted package.

### Runner and failures

The entrypoint runs in a dedicated Web Worker so the runner can isolate lifecycle and terminate failures; participants are trusted, so this worker is not treated as a hostile-code security boundary. Network access is forbidden after the declared package files load. Any load error, exception, worker crash, non-string result, invalid SAN, or illegal move is an immediate game loss with no retry or fallback move.

A submission that has not returned a move after 1.25 times the configured `moveTimeLimitMs` has its worker terminated and loses the game. The grace factor exists so that a package aiming at its budget is not forfeited for a few milliseconds of overshoot; it is not additional thinking time and MUST NOT be relied on. Termination is the only enforcement that survives a package that blocks its worker synchronously.

Runner-side faults are distinct from submission faults. A closed tab, reloaded page, sleeping machine, or failed result write costs no competitor a point: the affected game is simply replayed. A scheduled game that fails repeatedly is recorded as a forfeit after a fixed number of attempts, so a package that reliably crashes the runner cannot stall a tournament indefinitely.

Every game of a tournament MUST be played on the machine pinned when the tournament started, and only one runner may be active at a time. If that machine becomes unavailable, an administrator MAY authorise a different one; the change is recorded permanently against the tournament and displayed alongside its standings, because a tournament split across machines is not a clean result under a wall-clock limit.

### Entry integrity

A registered entry stores the Hugging Face reference resolved to a full 40-character commit SHA, together with the SHA-256 digest of its manifest. The runner loads that exact pinned reference and MUST NOT re-resolve a branch or tag, so publishing new weights after registration closes cannot affect a tournament. The runner verifies the manifest digest against the registered value before play and refuses an entry that does not match.

### Open question: hand-written inference knowledge

Not yet ruled on. The training rules forbid outside data, pretrained weights, and engine labels when producing a checkpoint, but say nothing about chess knowledge hand-written into `entry.js` at inference time. A submission could legally ship a search whose leaf evaluation is a hand-tuned material and piece-square-table function, contributing strength that was never learned from the frozen corpus. Competitors should agree before the tournament whether inference-time evaluation must derive from the submitted learned artifacts, or whether hand-written chess knowledge in the entrypoint is permitted.
