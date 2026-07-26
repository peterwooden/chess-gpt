# Baseline 0001: a functional chess model

On 25 July 2026, experiment `0001-basic-san-ngram` trained the repository's first functional chess model. Given a legal history in Standard Algebraic Notation (SAN), its command-line interface prints exactly one legal SAN move. This experiment predates the agreed Lichess monthly corpus and is a teaching baseline, not a tournament-eligible submission.

## What it learned

This is a deliberately inspectable count-based language model, not a neural network. From 8,990 training games, it counted which move most often followed the previous two moves; when that exact context was unseen, it backed off to one move, then a side-to-move frequency table. At inference, it discards illegal candidates before choosing.

That makes the core language-model intuition visible: use the preceding tokens as context to estimate the next-token distribution. The next experiment can replace those lookup tables with learned embeddings and a neural network while keeping the same data split and evaluation contract.

## Versioned inputs

- Dataset: [`shazmate/lichess-chess-tokens`](https://huggingface.co/datasets/shazmate/lichess-chess-tokens/tree/cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94) at revision `cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94`
- File: `tokenised/shard-00040.parquet`
- File SHA-256: `51554f41ef244a19c926f862d0c414137f295329922c925438f13bf90d5cfb3b`
- First 10,000 valid games, split deterministically by hashing `seed:site`
- Seed: `20260725`
- Producing code: [`19140c2`](https://github.com/peterwooden/chess-gpt/commit/19140c22a59bf23e1bd7df753845d187a3a3e02d)
- Machine: Apple M1 Pro MacBook Pro, 16 GB RAM, macOS 26.5.1; CPU execution

## Held-out result

| Measure | Result |
| --- | ---: |
| Training games / plies | 8,990 / 602,945 |
| Validation games / plies | 1,010 / 68,489 |
| Model top-1 next-move accuracy | 22.41% |
| Alphabetically-first legal-move fallback | 4.76% |
| Predictions from a learned context | 92.91% |
| Legal output rate | 100% |
| Canonical learned state | 6.69 MB |
| Compressed checkpoint | 1.41 MB |
| Wall-clock training time | 56.96 seconds |

The model beat its predeclared fallback control by 17.65 percentage points and passed every acceptance criterion. Accuracy asks whether it predicted the human move actually played; it does not mean that every different legal move was bad.

The local generated checkpoint is `runs/0001-basic-san-ngram/model.json.gz`, with SHA-256 `c143646b163c8c91c39bc654bf9a57d0a142bf010ec7b9c90971248616103f1c`. It is ignored by Git because generated artifacts should be recreated from the tracked experiment rather than committed as opaque binaries.

## Reproduce it

After placing the pinned shard at the path below:

```bash
uv sync --group dev
uv run chess-gpt-baseline train \
  --data data/downloads/lichess-chess-tokens-cb90f1b/tokenised/shard-00040.parquet \
  --output runs/0001-basic-san-ngram \
  --max-games 10000 \
  --validation-percent 10 \
  --seed 20260725 \
  --order 2 \
  --top-moves-per-context 16
```

Ask it for Black's next move after `e4 e5 Nf3`:

```bash
uv run chess-gpt-baseline move \
  --checkpoint runs/0001-basic-san-ngram/model.json.gz \
  --moves e4 e5 Nf3
```

It returns one line:

```text
Nc6
```

An 80-ply self-play smoke test also completed with no illegal moves. This proves the interface is functional, not that the baseline plays strong chess: it has no board understanding, search, or long-range plan, and its self-play eventually becomes repetitive. Those weaknesses give us measurable targets for the first neural model.

## Play it in the browser

The verified checkpoint is exported through the unified package contract in the public [`peterwooden/chess-gpt-demo-ngram`](https://huggingface.co/peterwooden/chess-gpt-demo-ngram/tree/bea221167728c33f0a5df54051cd27717cae6586) model repository. Open the [Chess GPT browser arena](https://chess-gpt-lab.peter-r-wooden.chatgpt.site/arena), enter this pinned package reference for Player 1, and load it:

```text
peterwooden/chess-gpt-demo-ngram@bea221167728c33f0a5df54051cd27717cae6586
```

The package contains a self-contained JavaScript adapter and the canonical 6,687,169-byte model state. The arena verified all 6,689,698 package bytes, started a fresh game in a dedicated Worker, and reproduced the command-line model's `Nf3` and `Nc3` choices for the histories tested. Its manifest SHA-256 is `fc1160f487f315036fc320a7e7ced5a2e8cf9872345318a2959f8f727b949702`.

Recreate the package from the measured checkpoint:

```bash
uv run chess-gpt-package \
  --checkpoint runs/0001-basic-san-ngram/model.json.gz \
  --entrypoint adapters/san-ngram/entry.js \
  --output runs/0001-basic-san-ngram/browser
```

The older compressed `model.json.gz` remains in the Hub repository as a historical artifact, but the unified arena accepts only `chess-gpt-package-v1`.

### What the package test taught us

- The manifest works as a root digest: hashing it commits to the entrypoint and model hashes it contains, so the arena can display one short package identity after verifying the whole graph.
- The 100 MB rule must be checked on canonical files, not transfer size: this model's 1.41 MB gzip becomes a 6.69 MB package when its learned state is represented uncompressed.
- A Worker gives the runner clean lifecycle and termination, but it is not a hostile-code sandbox; import parsing and blocked Worker capabilities enforce the trusted-participant contract without pretending to provide a complete security boundary.
- The pure-JavaScript adapter proves package v1 end to end, but ONNX remains unproven: ONNX Runtime Web may lazily load a WASM asset after package loading, which conflicts with the literal zero-network rule. We have left the agreed rule unchanged; before accepting an ONNX submission, the three participants should either agree that runner-owned same-origin runtime assets are exempt or require the runner to preload/bundle them before contestant code begins.
