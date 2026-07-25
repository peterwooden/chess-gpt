# Tournament rules draft

- **Learned-state limit:** The submitted inference model may contain at most 100 MB of learned state in its canonical uncompressed representation, including all weights, embeddings, biases, learned buffers, quantization scales, codebooks, and model-specific constants.
- **Shared dataset:** Everyone must use [`shazmate/lichess-chess-tokens`](https://huggingface.co/datasets/shazmate/lichess-chess-tokens/tree/cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94) at revision `cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94` as the shared dataset.
- **Training-compute limit:** Producing a submitted checkpoint may use at most $10^{18}$ training FLOPs (1 exaFLOP) across its entire training lineage, measured by the same agreed profiler regardless of local or cloud hardware.
- **Inference interface:** The model receives the game history as SAN moves and returns exactly one SAN move.
- **Tournament winner:** Each pair of models plays 100 games from 50 openings with colors reversed on the same runner; wins score 1 point, draws score ½ point, and losses score 0, so the model with the highest total score wins, while a first-place tie is settled by additional color-reversed opening pairs until one model leads after a complete pair, with all checkpoints, code, data, and configurations frozen before any openings are revealed.
