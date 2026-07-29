# Chess GPT

A teaching-first, reproducible laboratory for building a small chess language model and entering a three-friend tournament.

The priorities are: win fairly, learn ML engineering deeply, and turn creative ideas into controlled experiments. Start with [MISSION.md](MISSION.md), follow the checkbox [curriculum](CURRICULUM.md), then read the [founding intent](PROJECT_INTENT.md) and finalized [tournament rules](docs/TOURNAMENT_RULES.md).

## Current stage

The repository now has a reproducible, functional floor: a count-based SAN language model trained on 10,000 games. It achieved 22.41% held-out next-move accuracy, returned legal moves on every validation position, and completed an 80-ply self-play smoke test. Read the [full baseline record](docs/BASELINE_0001.md) or [play its package-v1 export](https://huggingface.co/peterwooden/chess-gpt-demo-ngram/tree/bea221167728c33f0a5df54051cd27717cae6586) through the browser arena.

The complete learning roadmap and adaptive placement diagnostic are live in the Sites app. The learner's latest diagnostic attempt scored 7/8 on the direct track and the follow-up prediction demonstrated the remaining train/validation/test concept. The resulting [adaptive Chapter 1 plan](docs/CHAPTER_1_PLAN.md) now begins with an interactive mission on honest game-level data splits; completion remains unrecorded until the learner returns its code and explanation.

The same site now includes a client-only [browser arena](site/app/arena/page.tsx). It downloads and verifies the unified package from Hugging Face, runs its entrypoint in a dedicated Worker, and supports human-versus-model or model-versus-model play while showing SAN moves and inference timing. Verified files from full 40-character commit-SHA references use a best-effort browser cache and are re-verified before execution; branches, tags, and short hashes always bypass that cache. The exact submission and inference contract is the [tournament rules technical appendix](docs/TOURNAMENT_RULES.md#technical-appendix-model-package-v1).

## Local setup

```bash
uv sync --group dev
uv run chess-gpt-doctor
uv run pytest
uv run ruff check .
uv run pyright
```

`chess-gpt-doctor` performs one tiny forward/backward pass and reports whether PyTorch selected the Apple MPS backend. PyTorch has a CPU fallback so correctness does not depend on acceleration.

## First playable baseline

Experiment `0001-basic-san-ngram` is the deliberately simple, pre-tournament floor: it learns which SAN moves tend to follow the previous two moves, backs off to broader frequencies when needed, and filters every prediction through the legal moves in the current position. It predates the agreed Lichess monthly corpus and is therefore not tournament-eligible. Its [versioned specification](experiments/0001-basic-san-ngram.toml) contains the measured result; generated checkpoints and metrics stay under the ignored `runs/` directory.

After obtaining the pinned dataset shard named in the experiment, train and ask for one move:

```bash
uv run chess-gpt-baseline train --data data/downloads/lichess-chess-tokens-cb90f1b/tokenised/shard-00040.parquet --output runs/0001-basic-san-ngram --max-games 10000 --validation-percent 10 --seed 20260725 --order 2 --top-moves-per-context 16
uv run chess-gpt-baseline move --checkpoint runs/0001-basic-san-ngram/model.json.gz --moves e4 e5 Nf3
```

## Board-snapshot tournament candidates

Experiments [`0002`](experiments/0002-board-snapshot-policy.toml) and [`0003`](experiments/0003-phase-moe-policy.toml) replace SAN-history input with the complete current board position. The second candidate adds deterministic opening, middlegame, and endgame experts. Both predict move identities internally and return exact legal SAN through a self-contained browser adapter.

The reusable, resumable tournament-data workflow is documented in [`data/README.md`](data/README.md). Generated archives, prepared Parquet, checkpoints, ONNX models, and packages remain outside Git under `data/downloads/` and `runs/`.

For the three-hour laptop MoE run, start the live logarithmic loss chart in one terminal and training in another:

```bash
uv run chess-gpt-snapshot-monitor --run runs/0003-phase-moe-policy
uv run chess-gpt-snapshot-train \
  --train data/downloads/tournament-2026/prepared/board-snapshot-v1/2026-01.parquet \
  --validation data/downloads/tournament-2026/prepared/board-snapshot-v1/2026-04.parquet \
  --output runs/0003-phase-moe-policy \
  --experiment-id 0003-phase-moe-policy \
  --architecture phase_moe --d-model 336 --layers 6 --heads 8 \
  --ff-multiplier 4 --batch-size 128 --device mps \
  --max-hours 3 --max-updates 26650 --log-every-updates 1
```

The chart refreshes every two seconds. Its **End training** button finishes the current optimizer update, evaluates the saved model, and writes a valid checkpoint and metrics rather than abandoning the run.

## Repository map

```text
CURRICULUM.md       versioned course map and progress checklist
data/               provenance manifests, never bulk data
docs/               tournament and engineering decisions
experiments/        versioned hypotheses and run specifications
learning-records/   demonstrated knowledge, not activity logs
runs/               ignored generated metrics and checkpoints
site/               published learning lab: lessons, glossary, browser model arena
src/chess_gpt/      inspectable project code
tests/              fast correctness checks
```

## Weekly rhythm

One session should produce one small conceptual win and one trustworthy artifact:

1. predict;
2. build the smallest version;
3. measure and inspect failures;
4. explain the result;
5. record what changed our understanding.

The next teaching step is Chapter 1 Mission 1: predict how position-level leakage biases validation, choose the trustworthy split implementation, then explain the result from memory.
