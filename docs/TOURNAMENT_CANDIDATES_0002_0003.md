# Tournament candidates 0002 and 0003

This release publishes two browser-native chess policy candidates trained from the
same frozen Lichess slice under the same laptop budget. Model 1 consumes a complete
board snapshot. Model 2 keeps the same encoder and adds deterministic
opening/middlegame/endgame experts.

## Shared experimental contract

- Training data: the first 100,000 games from the checksum-pinned January 2026
  Lichess standard-rated archive.
- Validation data: the first 2,000 games from the separately pinned April 2026
  archive; April positions were never used for optimization.
- Representation: 64 piece-square tokens plus side to move, castling rights, legal
  en-passant square, and halfmove clock.
- Phase distribution: January contains 1,960,245 opening (29.61%), 3,328,159
  middlegame (50.28%), and 1,331,400 endgame positions (20.11%). April contains
  39,019 opening (29.67%), 66,419 middlegame (50.51%), and 26,053 endgame
  positions (19.81%).
- Target: one of 4,272 stable move identities. The browser adapter replays SAN,
  masks to the runner-supplied legal SAN moves, and returns the highest-logit legal
  move exactly.
- Optimizer budget: AdamW, seed `20260729`, batch size 128, at most 26,650 updates
  and three hours per model.
- Compute accounting: ratified dense-operation profiler v1, including all three
  evaluated MoE branches and any checkpoint lineage.
- Runtime contract: `chess-gpt-package-v1`, ONNX Runtime Web 1.27.0 supplied by the
  arena, deterministic argmax inference, and a fresh logical adapter per game.

## Results

| Measure | Model 1 · snapshot | Model 2 · phase MoE |
| --- | ---: | ---: |
| Parameters | 10,586,256 | 12,397,296 |
| Updates | 26,255 | 26,276 |
| Training positions | 3,356,140 | 3,358,828 |
| Training FLOPs | 11,029,491,768,153,600 | 11,074,728,541,459,968 |
| Final minibatch loss | 2.87425 | 2.57173 |
| April validation loss | **2.83490** | 2.85425 |
| April raw top-1 | **26.858%** | 26.749% |
| April legal top-1 | **29.387%** | 29.253% |
| Legal move rate | 100% | 100% |
| Canonical package bytes | 42,585,883 | 49,841,108 |
| Local paired score | 50/100 | 50/100 |

Both runs stopped at the three-hour training limit and cost $0. Model 2 processed
2,688 more positions and used about 0.41% more charged FLOPs, so the published runs
are close but not exactly matched on those causal controls. Model 1 had slightly
better April imitation metrics. Neither validation metrics nor the lower final
Model 2 minibatch loss establish playing strength.

The tournament-shaped local comparison used 50 frozen April validation openings
with colors reversed. Across 100 games, each model won six, 88 were drawn, and each
model scored exactly 50 points. Terminations were 12 checkmates, one stalemate, and
87 threefold repetitions. Thus the observed Model 2 score rate was 50%, not the
predicted 70%. This does not support the prediction for these published candidates;
because processed positions and FLOPs were not exactly matched, it is not the
specified causal architecture test either.

## Public models

- **Model 1:**
  [`peterwooden/chess-gpt-board-snapshot-0002`](https://huggingface.co/peterwooden/chess-gpt-board-snapshot-0002),
  verified browser-package revision
  [`ecdf3c42046c01abdd351d1327b77d18388c4306`](https://huggingface.co/peterwooden/chess-gpt-board-snapshot-0002/tree/ecdf3c42046c01abdd351d1327b77d18388c4306).
- **Model 2:**
  [`peterwooden/chess-gpt-phase-moe-0003`](https://huggingface.co/peterwooden/chess-gpt-phase-moe-0003),
  verified browser-package revision
  [`a49f3ee42bc747258b5191f1dd7fca11a6b4bb25`](https://huggingface.co/peterwooden/chess-gpt-phase-moe-0003/tree/a49f3ee42bc747258b5191f1dd7fca11a6b4bb25).

The final evidence heads are `c625220345702176aff2dbaa930924a80c86e29b`
for Model 1 and `757864da4a7e0bb5736af6cbf2cfff13a54c4a09` for
Model 2. Each contains the model card, experiment TOML, measured metrics, full
per-update loss log, and identical paired-match JSON. Those evidence files are
outside the manifest-referenced tournament byte count.

## Architectures

Model 1 (`0002-board-snapshot-policy`) is a six-layer, eight-head Transformer with
width 336 and one post-encoder feed-forward expert. It has 10,586,256 trainable
parameters and a profiled cost of 3,286,362,240 training FLOPs per processed
position.

Model 2 (`0003-phase-moe-policy`) uses the same encoder dimensions and replaces the
single expert with three phase-specific experts selected from visible material and
move count. The profiler charges all evaluated branches. It has 12,397,296 trainable
parameters and a profiled cost of 3,297,200,256 training FLOPs per processed
position.

Before material training, the learner predicted Model 2 would win 70% of a matched
head-to-head comparison, with 70% confidence. The stated assumption was an
information-efficient shared board encoder; the predicted failure mode was
inconsistent phase routing at tournament inference.

## What the decoder does

Both networks return 4,272 unnormalized move logits, not SAN strings. During
training, cross-entropy consumes those logits directly. During tournament play,
the package reconstructs the position from the runner's SAN history, indexes only
the supplied legal SAN choices, and returns the legal move with the largest logit.
Softmax probabilities would preserve that ordering, so computing them would add no
information for deterministic argmax play. The adapter—not the neural network—does
the final conversion back to exact SAN.

## Interpretation guardrails

The fixed safety budget is three wall-clock hours per model. Because the MoE has
more parameters and evaluates three expert branches, equal time does not guarantee
equal processed positions or equal charged FLOPs. The paired games compare the two
published candidates as they actually exist; they do **not** by themselves test the
learner's causal 70% prediction, which explicitly assumes matched data, optimizer,
processed positions, and FLOPs.

Likewise, held-out next-move agreement is an imitation metric rather than a direct
measure of chess strength. Any difference in April validation loss or accuracy is
reported descriptively and is not substituted for the paired match result. With
only 100 paired games, the match remains a limited strength signal with substantial
sampling uncertainty, not a precise Elo estimate.

## Compliance and package verification

- The January and April source SHA-256 digests are pinned in
  [`data/dataset.toml`](../data/dataset.toml); their prepared Parquet digests and
  row counts are captured in each run's `metrics.json`.
- April is validation-only. No hidden or final test split is opened, inspected, or
  used for model selection.
- Each browser manifest pins its entrypoint, ONNX model, and vocabulary by exact
  byte count and SHA-256. Only those canonical files count toward the 100 MB cap.
- Publication verification downloads an immutable Hugging Face commit into a clean
  directory, compares every byte with the local package, loads the real supplied
  ONNX Runtime Web implementation, and requires 40 consecutive legal SAN plies.
- The paired match uses 50 distinct frozen April positions at ply 12 and reverses
  colors, for 100 games, matching the tournament's opening and color structure. It
  is a local validation match, not a preview of the unrevealed official tournament
  openings. The starts are position snapshots, so repetition history before ply 12
  is unavailable. The evaluator records each result and termination along with code,
  environment, dataset, and checkpoint digests.
- Both runs use local Apple M4 hardware, so monetary training cost is $0.

## Reproduce

The exact commands, data digests, environment-lock digest, code revision, seed,
actual positions, actual FLOPs, hardware, losses, checkpoint hashes, and validation
metrics are recorded in the two experiment TOML files and generated run metrics.
Large source archives, prepared Parquet, checkpoints, and ONNX packages remain
outside Git as documented in [`data/README.md`](../data/README.md).

Validation next-move accuracy measures agreement with human moves, not chess
strength. The release reports paired play separately and did not inspect or tune on
any final test split or hidden tournament opening.
