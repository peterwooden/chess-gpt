# Data

Large datasets do not live in Git. [`dataset.toml`](dataset.toml) freezes the four agreed Lichess monthly files, their train/validation roles, and their official SHA-256 digests.

A download command must verify every required filename and digest before exposing the data to training code. Small, intentionally sampled fixtures for tests may be committed under `tests/fixtures/` with their provenance.
