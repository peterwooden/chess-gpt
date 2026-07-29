# Data

Large datasets do not live in Git. [`dataset.toml`](dataset.toml) freezes the four agreed Lichess monthly files, their train/validation roles, and their official SHA-256 digests.

A download command must verify every required filename and digest before exposing the data to training code. Small, intentionally sampled fixtures for tests may be committed under `tests/fixtures/` with their provenance.

## Reusable tournament cache

The board-snapshot pipeline stores verified source archives and compressed prepared shards under the Git-ignored `data/downloads/tournament-2026/`. The four source archives total about 116.3 GB, while this machine currently cannot hold all of them at once. Fetch, prepare, verify, and discard each raw month sequentially:

```bash
uv run chess-gpt-snapshot-data fetch-prepare \
  --month 2026-01 --max-games 750000 --discard-raw
uv run chess-gpt-snapshot-data fetch-prepare \
  --month 2026-02 --max-games 750000 --discard-raw
uv run chess-gpt-snapshot-data fetch-prepare \
  --month 2026-03 --max-games 750000 --discard-raw
uv run chess-gpt-snapshot-data fetch-prepare \
  --month 2026-04 --max-games 100000 --discard-raw
```

Downloads resume into `.part` files. A completed archive is exposed only after its SHA-256 matches [`dataset.toml`](dataset.toml). Prepared Parquet rows contain the 64-square position, rule state, deterministic phase, and next-move class; model training never receives SAN history. Do not use April shards for training.
