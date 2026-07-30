# Strong-winner candidates 0004 and 0005

This release tests one data-selection idea against both published board-snapshot architectures: keep only decisive games whose winner is rated at least 1600, then train only on positions where that winner is to move. The loser's rating is unrestricted. No architecture, optimizer, seed, processed-position budget, or FLOP budget changed within either old-versus-new comparison.

## Prediction and design

Before training, the learner predicted each new model would win 80 of 100 games against its corresponding predecessor, with 40% confidence. The key assumption was that sub-1600 winners do not add useful signal at the lower end; the likely failure mode was very poor play against poor players.

The frozen January preparation scanned 210,285 games to accept 120,000, filtered 90,285, found zero invalid games, and produced 4,149,869 winner-side positions. The frozen April validation preparation scanned 9,332 games to accept 5,000, filtered 4,332, found zero invalid games, and produced 174,404 positions. April was never used for optimization. Neither a final test split nor the hidden tournament openings were inspected.

Both new models used the same January and April prepared shards. Model 0004 processed exactly the same 3,356,140 positions and 11,029,491,768,153,600 FLOPs as model 0002. Model 0005 processed exactly the same 3,358,828 positions and 11,074,728,541,459,968 FLOPs as model 0003.

## Results

| Measure | 0004 · snapshot | 0005 · phase MoE |
| --- | ---: | ---: |
| Parameters | 10,586,256 | 12,397,296 |
| Training positions | 3,356,140 | 3,358,828 |
| Training FLOPs | 11,029,491,768,153,600 | 11,074,728,541,459,968 |
| Final minibatch loss | 2.70889 | 2.75638 |
| Filtered-April validation loss | **2.79660** | 2.88798 |
| Filtered-April raw top-1 | **28.486%** | 27.030% |
| Filtered-April legal top-1 | **30.405%** | 28.944% |
| Legal move rate | 100% | 100% |
| Canonical package bytes | 42,585,891 | 49,841,122 |
| Wins against predecessor | 22 | 10 |
| Losses against predecessor | 3 | 4 |
| Draws against predecessor | 75 | 86 |
| Score against predecessor | **59.5/100** | **53/100** |

The paired matches use the first 50 distinct valid ply-12 positions from the same frozen, unfiltered April shard used for the earlier tournament-shaped match, with colors reversed for 100 games. Model 0004 beat model 0002 by 22 wins to 3 and scored 59.5%. Model 0005 beat model 0003 by 10 wins to 4 and scored 53%. This supports the predicted direction for both matched comparisons, but not the predicted 80% all-game win rate: the observed win rates were 22% and 10%. The high draw rates make win rate and score rate materially different.

The predicted weak-opponent failure mode remains untested. The opponents were the prior tournament candidates, not deliberately weak policies or low-rated humans. A separate preregistered opponent-strength evaluation would be needed to test that claim.

## Interpretation

The plain snapshot result is the stronger signal: it gained 19 net wins and 9.5 match points over its old counterpart under exact matched compute. The phase-MoE result is positive but smaller: six net wins and three match points. These are limited 100-game validation matches, not precise Elo estimates or the unrevealed official tournament.

Filtered next-move metrics cannot be directly compared with the old models' unfiltered-April metrics because the target populations differ. Within the new models, 0004 has better imitation metrics than 0005 on the identical filtered shard. Match play remains the relevant strength evidence.

## Compliance

- Only checksum-pinned January training and April validation Lichess standard-rated games were used.
- No pretrained weights, engine labels, outside games, or synthetic data were added.
- The winner rating and game result come from the source PGN headers; only the winner's own moves become targets.
- Each run starts from a fresh random initialization and has zero parent-checkpoint lineage FLOPs.
- The ratified dense-operation profiler charges all evaluated MoE branches.
- Both browser packages are below the 100,000,000-byte cap and pass local ONNX Runtime Web 1.27.0 loading plus 40 legal SAN self-play plies.
- Exact data, checkpoint, package, evaluator, and match digests are recorded in the experiment TOML files and published evidence bundles.

## Public models

- **Model 0004:** [`peterwooden/chess-gpt-board-snapshot-strong-winner-0004`](https://huggingface.co/peterwooden/chess-gpt-board-snapshot-strong-winner-0004), immutable verified browser-package revision [`d29db50441c36a109f714b9aafd231fa8e37008c`](https://huggingface.co/peterwooden/chess-gpt-board-snapshot-strong-winner-0004/tree/d29db50441c36a109f714b9aafd231fa8e37008c), complete evidence revision `0069b9427d101dfa9f18ab946b6870ec88523e68`.
- **Model 0005:** [`peterwooden/chess-gpt-phase-moe-strong-winner-0005`](https://huggingface.co/peterwooden/chess-gpt-phase-moe-strong-winner-0005), immutable verified browser-package revision [`6bae33a48f207aa1519bbca620094f77ace61dfb`](https://huggingface.co/peterwooden/chess-gpt-phase-moe-strong-winner-0005/tree/6bae33a48f207aa1519bbca620094f77ace61dfb), complete evidence revision `e34c647cf4cd8cb27ce6dd64af90d6ff14d90e88`.
