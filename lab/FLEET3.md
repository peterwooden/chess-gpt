# Fleet 3 — broad technique sweep on the f1 baseline (2026-07-31, very late)

Baseline (all arms inherit): f1 recipe = MLP 2×1152, history-8, per-square, value 1.0, dropout 0.1,
warmup 2% + cosine, batch 1024 / lr 1.2e-3, bf16, torch.compile. Arms are 2-epoch matched-budget
A100 jobs scored by val loss/top-1 vs a fresh triplicate control. Deliberate bad ideas marked `!`.
Nothing repeats a Fleet-1/2 arm.

## Controls
c1-c3. f1 baseline ×3 (new noise floor)

## Data
d1. 960k games × 1 epoch (data-vs-repetition duel, shard prepping)
d2. skip opening plies <6 (book covers them anyway)
d3. ! opening-only training (ply <20)
d4. ! endgame-only training (ply ≥40)
d5. 50% position subsample × 2× epochs (coverage-vs-repetition control)
d6. ! 5% random-label noise on policy targets

## Representation
r1. history 0 (ablation at scale) / r2. history 4
r3. ! shuffled history order (does sequence order matter?)
r4. zero halfmove clock (ablation) / r5. zero castling rights (ablation)
r6. rank+file factored square embeddings (8+8 replaces 64)
r7. input dropout: mask 5% of squares
r8. d_model 192 token embeddings
r9. ! fixed random permutation of square embeddings (geometry destroyed — instrument floor)

## Architecture
a1. hidden 1536 / a2. hidden 768 (scale curve on the new recipe)
a3. gated MLP (GLU-style)
a4. two-tower: separate policy and value trunks (tests the trunk-sharing hypothesis)
a5. LayerNorm on the flattened input
a6. input→token residual (skip the whole trunk)
a7. low-rank token projection (bottleneck 64)
a8. untied per-square readout (64 separate heads)

## Objective
j1. value as win-probability BCE (draws = 0.5)
j2. value as scalar regression (MSE)
j3. soft value labels (0.9 smoothing)
j4. focal loss on policy
j5. ! entropy bonus on policy (encourage indecision)
j6. aux: predict opponent's reply (next-move head)
j7. aux: masked-square reconstruction (BERT-style, 4 squares)
j8. ! value weight 5.0
j9. self-distill from #57 (harness-bug retry)

## Optimization
o1. warmup 5% / o2. warmup 0.5% (dose-response)
o3. cosine floor 10% (don't decay to zero)
o4. peak lr 1.8e-3 (higher peak, now warmup-protected)
o5. two cosine cycles
o6. embeddings at 10× lower LR
o7. adam eps 1e-4
o8. ! SGD + momentum + cosine
o9. muon-v2: Muon on trunk only, AdamW heads untouched (fix for Fleet-2's value damage)
o10. batch 512 (smaller, never tried)
o11. ! gradient gaussian noise
o12. SWA: average weights over final 20%

## Finalists
f1-f2. top gate-compatible arms at full budget → beam-gate vs champion f1
