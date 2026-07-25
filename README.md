# Chess GPT

A teaching-first, reproducible laboratory for building a small chess language model and entering a three-friend tournament.

The priorities are: win fairly, learn ML engineering deeply, and turn creative ideas into controlled experiments. Start with [MISSION.md](MISSION.md), then read the [founding intent](PROJECT_INTENT.md) and [tournament rules draft](docs/TOURNAMENT_RULES_DRAFT.md).

## Current stage

We are establishing the rules, data provenance, local environment, and first-principles curriculum. The public dataset is recorded as a **candidate**, not silently treated as frozen, because its published artifacts and GPCT vocabulary generation do not yet agree.

## Local setup

```bash
uv sync --group dev
uv run chess-gpt-doctor
uv run pytest
uv run ruff check .
uv run pyright
```

`chess-gpt-doctor` performs one tiny forward/backward pass and reports whether PyTorch selected the Apple MPS backend. PyTorch has a CPU fallback so correctness does not depend on acceleration.

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
