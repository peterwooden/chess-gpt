# Working agreement for this repository

Before making project decisions, read `MISSION.md`, `PROJECT_INTENT.md`, `NOTES.md`, and `docs/TOURNAMENT_RULES.md`.

## Teaching

- This is both a competitive ML project and a teaching workspace. Explain the mechanism the learner is about to use, then give them a small prediction or retrieval task.
- Keep lessons narrow enough for roughly one hour of work per week. Record demonstrated understanding in `learning-records/`; do not confuse exposure with learning.
- Prefer small, inspectable implementations before framework abstractions. Production improvements must remain explainable by the learner.

## Experimental integrity

- Never use the test split for routine model selection.
- Never compare runs unless their data revision, split, evaluation semantics, and budget match.
- Every material run needs a versioned experiment specification and must record code revision, data revision, environment lock, seed, metrics, artifacts, and actual token/compute budget.
- Validation loss alone is not playing strength. Maintain chess-level metrics and paired match evaluation.

## Data and tournament fairness

- Use only the frozen monthly files and split assignments in `data/dataset.toml` for tournament experiments.
- Do not add outside games, pretrained weights, engine labels, or synthetic data unless the agreed tournament rules permit them; search and auxiliary inference systems must fit inside the same submitted package and interface.
- Count the manifest, entrypoint, model artifacts, vocabulary, configuration, and model-specific constants in the 100 MB submission cap as specified by the rules.

## Engineering

- Use the checked-in `uv.lock`; do not install project packages ad hoc.
- Keep large datasets, checkpoints, and generated run artifacts out of Git.
- Tests, lint, type checks, and a tiny end-to-end smoke run must pass before scaling an experiment.
- After requested repository edits pass their relevant checks, commit and push them to `main` unless the user explicitly says not to.
