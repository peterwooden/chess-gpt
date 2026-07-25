# Tournament rules draft

- Models may contain at most 50,000,000 unique trainable parameters, including embeddings and output heads.
- Everyone must use [`shazmate/lichess-chess-tokens`](https://huggingface.co/datasets/shazmate/lichess-chess-tokens/tree/cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94) at revision `cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94` as the shared dataset.
- Producing a submitted checkpoint may use at most $10^{18}$ training FLOPs across its entire training lineage, measured by the same agreed profiler regardless of local or cloud hardware.
- The inference interface receives the game history as SAN moves and returns exactly one SAN move, with no search or auxiliary chess engine.
- Each pair of models plays 100 games from 50 openings with colors reversed on the same runner; wins score 1 point, draws score ½ point, and losses score 0, so the model with the highest total score wins, while a first-place tie is settled by additional color-reversed opening pairs until one model leads after a complete pair, with all checkpoints, code, data, and configurations frozen before any openings are revealed.
