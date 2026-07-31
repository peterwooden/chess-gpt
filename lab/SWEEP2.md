# Fleet 2 — learning-efficiency sweep + opening play (2026-07-31)

Cloud arms: A100 jobs, 2 epochs over the 480k-game shard cache (~54M positions, matched budget),
scored by validation loss/top-1. Control in triplicate = noise floor; arms inside the band are noise.
Top finalists rerun at full budget and beam-gated vs #57. Opening arms run locally (inference layer),
scored by match points vs #57 plus an opening-conformity metric against a corpus book. `†` = deferred
(needs prep work the harness can't do tonight).

## Controls
c1-c3. champion recipe ×3 (noise floor)

## Optimizers
o1. Muon (2D weights) + AdamW rest
o2. Lion
o3. schedule-free AdamW
o4. wd 0.1 / o5. wd 0
o6. betas (0.9, 0.98)
o7. lr 2.4e-3 / o8. lr 6e-4
o9. batch 4096 + lr 4.8e-3 (multi-epoch retest)
o10. grad clip 1.0

## Schedules
s1. cosine (anomaly rematch) / s2. warmup 2% + cosine
s3. warmup-stable-decay (20% decay tail)
s4. step ×0.1 at 80%
s5. weight EMA 0.999 (constant LR)
s6. EMA + cosine

## MLP trunk efficiency
m1. residual blocks / m2. + LayerNorm between blocks
m3. GELU / m4. ReLU²
m5. 3 layers × hidden 896 (≈matched params)
m6. 1 layer × hidden 2048 (≈matched params)
m7. dropout 0 / m8. dropout 0.2

## Objective
j1. value weight 0.5 / j2. value weight 2.0
j3. ply-weighted value loss (late-game emphasis)
j4. value masked to decisive games
j5. label smoothing 0.05
j6. aux plies head / j7. aux material head (scale retests)
j8. self-distill from #57 (KL T=2)
j9. search-labeled value †

## Data recipes
d1. elite mix 25% / d2. elite mix 50%
d3. endgame oversample ×2 (ply ≥ 60)
d4. curriculum short-games-first / d5. long-games-first
d6. 960k × 1 epoch vs 480k × 2 † (shard not prepared)
d7. dedup at scale † (prep-side)

## Velocity (wall-time scored, not quality)
v1. torch.compile on CUDA
v2. batch 2048 + sqrt-scaled lr

## Opening play (local, inference layer, vs #57 + conformity metric)
p1. corpus opening book (top popular move, first 12 plies, frequency-gated)
p2. policy-trust blend: root score += β·policy prob, β decaying by ply
p3. flat-value fallback: if root values within ε, play policy argmax
p4. book + search hybrid (p1 until out of book, then normal)

## Finalists
f1-f2. top recipes at full 6-epoch budget → beam-gate vs #57 → adopt or archive
