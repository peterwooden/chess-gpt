# Chess GPT

A teaching-first, reproducible laboratory for building a small chess language model and entering a three-friend tournament.

The priorities are: win fairly, learn ML engineering deeply, and turn creative ideas into controlled experiments. Start with [MISSION.md](MISSION.md), follow the checkbox [curriculum](CURRICULUM.md), then read the [founding intent](PROJECT_INTENT.md) and [tournament rules draft](docs/TOURNAMENT_RULES_DRAFT.md).

## Current stage

The repository now has a reproducible, functional floor: a count-based SAN language model trained on 10,000 games. It achieved 22.41% held-out next-move accuracy, returned legal moves on every validation position, and completed an 80-ply self-play smoke test. Read the [full baseline record](docs/BASELINE_0001.md) or [play the pinned checkpoint](https://huggingface.co/peterwooden/chess-gpt-demo-ngram/tree/fecf413cfe0e5dab427c4cec7a78aafa4410aa65) through the browser arena.

The complete learning roadmap and adaptive placement diagnostic are live in the Sites app. The learner's latest diagnostic attempt scored 7/8 on the direct track and the follow-up prediction demonstrated the remaining train/validation/test concept. The resulting [adaptive Chapter 1 plan](docs/CHAPTER_1_PLAN.md) now begins with an interactive mission on honest game-level data splits; completion remains unrecorded until the learner returns its code and explanation.

The same site now includes a client-only [browser arena](site/app/arena/page.tsx). It can download a compatible model from Hugging Face, run human-versus-model play, or run two models against each other while showing SAN moves and inference timing. The narrow, hash-verified interchange format is documented in the [browser model contract](docs/BROWSER_MODEL_CONTRACT.md).

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

Experiment `0001-basic-san-ngram` is the deliberately simple floor: it learns which SAN moves tend to follow the previous two moves, backs off to broader frequencies when needed, and filters every prediction through the legal moves in the current position. Its [versioned specification](experiments/0001-basic-san-ngram.toml) contains the measured result; generated checkpoints and metrics stay under the ignored `runs/` directory.

After obtaining the pinned dataset shard named in the experiment, train and ask for one move:

```bash
uv run chess-gpt-baseline train --data data/downloads/lichess-chess-tokens-cb90f1b/tokenised/shard-00040.parquet --output runs/0001-basic-san-ngram --max-games 10000 --validation-percent 10 --seed 20260725 --order 2 --top-moves-per-context 16
uv run chess-gpt-baseline move --checkpoint runs/0001-basic-san-ngram/model.json.gz --moves e4 e5 Nf3
```

## Repository map

```text
CURRICULUM.md       versioned course map and progress checklist
assets/             shared lesson components
data/               provenance manifests, never bulk data
docs/               tournament and engineering decisions
experiments/        versioned hypotheses and run specifications
learning-records/   demonstrated knowledge, not activity logs
lessons/            short interactive teaching units
reference/          durable glossaries and cheat sheets
runs/               ignored generated metrics and checkpoints
site/               published learning lab and browser model arena
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
