# Overnight atomic experiment map — 2026-07-30

Control (all deltas are vs this): d=128 L=6 transformer, jan-10k unfiltered, game split, 3 epochs, batch 1024 / lr 1.2e-3 + fastest flags that pass tonight's recipe suite, greedy decoding, ladder-rated. One atomic change per experiment. Elo: ladder rating for all; round-robin among top 8 for head-to-head refinement. `*` = needs new code.

## Data
1. + winner-only filter, elo floor 1600
2. + winner-only filter, elo floor 2000 (reuse elite2000 shard)
3. + decisive games only, both sides kept *
4. + no bullet games (rapid/classical only) *
5. + 20k games (2× data)
6. + 40k games (4× data)
7. + opening dedup: cap identical positions *
8. + both-players floor 1800 *
9. + draws-only training (negative control) *

## Representation
10. + last-2-moves tokens *
11. + last-8-moves tokens *
12. + repetition-count in state token *
13. + material-balance in state token (mild structural prior) *
14. − state token (board only, ablation) *

## Architecture
15. − transformer (MLP, matched params) *
16. + per-square readout *
17. + phase-MoE, 3 deterministic experts *
18. + rank/file/diagonal attention bias (geometric prior) *
19. + piece-value-initialized embeddings (structural-prior trap) *
20. d=64 L=2 (scale curve, low anchor)
21. d=96 L=4 (scale curve, mid)
22. d=160 L=8 (scale curve, high — wall-time permitting)
23. heads=8 (head_dim 16)
24. heads=2 (head_dim 64)
25. ff multiplier 2 *
26. + dropout 0.1 *

## Objective
27. + value head, weight 0.25
28. + value head, weight 0.5 (= Run B, reuse)
29. + value head, weight 1.0
30. + aux head: material balance *
31. + aux head: plies-remaining bucket *
32. + label smoothing 0.1 *
33. value-only, no policy head (play = 1-ply value search) *

## Optimization
34. 6 epochs (2× compute)
35. 9 epochs + cosine LR decay *
36. lr 3e-3
37. batch 4096 / lr 4.8e-3
38. + LR warmup 5% *

## Post-training
39. pretrain all-elo → fine-tune winner>2000, 1 epoch *
40. pretrain all → fine-tune decisive-only *
41. + self-play fine-tune: greedy self-play games, winner-side moves *
42. + expert-iteration lite: self-play with 1-ply value search targets *
43. distill best big → d=64 L=2 *

## Inference (re-rate best checkpoint, no retraining)
44. temperature 0.7 sampling *
45. top-3 sampling *
46. 1-ply value lookahead (needs 27–29) *
47. + contempt: shun repetition when value head says winning *

## Best shot
48. Professor's pick: compose every measured winner from 1–47, training still under ~15 min.
