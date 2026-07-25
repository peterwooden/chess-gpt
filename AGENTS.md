# Working agreement for this repository

Before making project decisions, read `MISSION.md`, `PROJECT_INTENT.md`, `NOTES.md`, and `docs/TOURNAMENT_RULES_DRAFT.md`.

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

- Treat the Hugging Face dataset revision in `data/dataset-candidate.toml` as a candidate until all three competitors freeze it in writing.
- Do not add outside games, pretrained weights, engine labels, synthetic data, search, or auxiliary inference systems unless the agreed tournament rules permit them.
- Count all model-specific learned state, including weights, embeddings, biases, learned buffers, quantization scales, codebooks, and constants, when checking the 100 MB submission cap.

## Engineering

- Use the checked-in `uv.lock`; do not install project packages ad hoc.
- Keep large datasets, checkpoints, and generated run artifacts out of Git.
- Tests, lint, type checks, and a tiny end-to-end smoke run must pass before scaling an experiment.
- After requested repository edits pass their relevant checks, commit and push them to `main` unless the user explicitly says not to.
