# Chess GPT

A teaching-first, reproducible laboratory for building a small chess language model and entering a three-friend tournament.

The priorities are: win fairly, learn ML engineering deeply, and turn creative ideas into controlled experiments. Start with [MISSION.md](MISSION.md), then read the [founding intent](PROJECT_INTENT.md) and [tournament rules draft](docs/TOURNAMENT_RULES_DRAFT.md).

## Current stage

The repository now has a reproducible, functional floor: a count-based SAN language model trained on 10,000 games. It achieved 22.41% held-out next-move accuracy, returned legal moves on every validation position, and completed an 80-ply self-play smoke test; read the [full baseline record](docs/BASELINE_0001.md).

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
assets/             shared lesson components
data/               provenance manifests, never bulk data
docs/               tournament and engineering decisions
experiments/        versioned hypotheses and run specifications
learning-records/   demonstrated knowledge, not activity logs
lessons/            short interactive teaching units
reference/          durable glossaries and cheat sheets
runs/               ignored generated metrics and checkpoints
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

The first lesson is [One move, three representations](lessons/0001-one-move-three-representations.html).
