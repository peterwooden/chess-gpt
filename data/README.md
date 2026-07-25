# Data

Large datasets do not live in Git. `dataset-candidate.toml` records the public source and observed revision without pretending it is already tournament-frozen.

Later, a download command will verify every required filename and digest before exposing the data to training code. Small, intentionally sampled fixtures for tests may be committed under `tests/fixtures/` with their provenance.
