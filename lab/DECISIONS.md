# Decision log

Each entry: the question, the options considered, what we chose, and why — written at the time of the decision, not after.

## Decision 0 — bench slice size (2026-07-30)

**Question:** how many games in the lab's standard data slice?

**Derivation, not a guess:** two constraints. Wall-clock: ~80 positions/game means 10k games ≈ 650–800k positions ≈ a few thousand optimizer steps per epoch at batch 256 — minutes on the M4 for a sub-1M-param model. Resolution: 10% held out gives ~65–80k validation positions; standard error of an accuracy near 30% is √(p(1−p)/n) ≈ ±0.16%, so half-point differences between experiments are readable. **Chose 10,000 games.** It is a dial, not a door: 5k trades resolution for speed, 20k the reverse.

## Decision 1 — train/validation split policy (2026-07-30)

**Question:** how to separate training from validation so the validation number can be trusted?

**Options considered:** random position split, game-level split, player-level split, April-shard validation.

**Learner's pick before the reveal:** random position split ("maximizes data usage, identical distributions").

**Chose: game-level split.** Random position splits leak: consecutive positions from one game are near-duplicates, so the model is partially shown validation answers during training, inflating validation metrics — and inflating them more for bigger, more memorization-capable models, which biases model selection. Player-level was judged overkill at bench scale; the April shard is reserved untouched as the outer ruler for re-verifying bench winners at medium scale.

**Preregistered prediction (learner):** the position-split arm will *appear* ~10 percentage points better in validation top-1 than the game-split arm. Confidence: not stated (noted; required next time).

**Test:** Experiment 1 — identical TinyPolicy (393,840 params), identical 10k-game slice, identical seed, 3 epochs, both split policies. The measured gap is the size of the lie.

**Result (2026-07-30): no gap.** Game split 10.51% validation top-1, position split 10.50%. Both predictions failed: the learner predicted +10 points for the position arm, the teacher predicted a visible landmine. Diagnosis from the same table: train ≈ validation in both arms (10.8% vs 10.5%), so the model was underfitting — leakage is a memorization phenomenon, and an underfit model has memorized nothing worth leaking. The game-level policy stands (the danger grows with scale and capacity, exactly when a broken ruler hurts most), but the honest record is that at bench scale the two rulers agreed.

## Experiment 2 — spring the trap deliberately (2026-07-30)

**Question:** does the position-split lie appear once the model can memorize?

**Learner's first proposal:** grow the dataset to 1M games — rejected on mechanism (more data lowers the capacity-to-data ratio and *suppresses* memorization; also exceeds the prepared shard). The corrected lever is the inverse: shrink data, extend training.

**Design:** same TinyPolicy, same seed, 1,000 games (~65k positions), 30 epochs, both split policies. Control between arms unchanged.

**Preregistered predictions:** learner — position arm appears ~10 points better, 50% confidence (carried over). Teacher — train/val gap opens in both arms; position arm inflates by 3–8 points, 70% confidence.

**Result (2026-07-30): the trap half-sprang.** Overfitting engaged (train 12.7% vs validation 10.4% top-1, loss gap 0.6 nats, both arms), yet the split gap stayed zero: game 10.36%, position 10.21% — inside noise. Both predictions failed again, the teacher's at 70% stated confidence.

**Refined mechanism:** leakage inflates validation in proportion to target correlation across the leaked examples. Next-move targets decorrelate within a couple of plies, so near-duplicate positions leak almost nothing; game *outcome* is identical across a whole game, so a value head under a position split should leak heavily. The folklore "position splits are dangerous" survives, but the reason is sharper: it depends on the target, not just the input similarity. Game-level splitting remains bench policy as cheap insurance, now held for evidence-shaped reasons.

## Decisions 2–6 — pipeline co-design pass (2026-07-30)

Ratified together after the learner called out unilateral building (three times; recorded as standing feedback): **targets** from×to move classes; **ruler** random-legal floor plus strength-capped Stockfish rungs; **decoding** greedy argmax with seeded 6-ply random openings for determinism-safe variety; **representation** board snapshot (successor candidate: snapshot + last-K moves for repetition awareness); **architecture** transformer encoder, summary-token readout, frozen sizes bench-S (d=64, 2 layers, ~394k) and bench-M (d=128, 4 layers, ~1.4M) with dropout 0 while runs underfit; **upstream data policy** decisive games with winner strictly above 2000 Elo, winner's moves only (~20% game acceptance, ~36 positions/game). Parked experiments: CNN vs transformer at matched compute; per-square readout; geometric attention bias.

## Experiment 4 — elite-data filter at matched budget (2026-07-30)

One variable: training-data policy. Elite slice (18,500 games, winner >2000, winner moves) vs unfiltered baseline (10,000 games, both sides), both ~660k positions, same bench-S, seed, epochs, split. Validation metrics are population-incomparable across the arms; matches are the verdict: 100 games head-to-head (6-ply seeded openings, colors reversed) and 100 vs random-legal (baseline reference: 52.5/100).

**Preregistered predictions:** learner — elite above baseline, magnitude unstated, 60% (initially mis-attributed the effect to data *and architecture*; corrected — architecture is matched by construction). Teacher — head-to-head 57–62/100, vs random 55–60 with the draw wall standing, 60%. Result: pending.

## Decision 7 — lab rating ladder (2026-07-30)

Learner proposed Elo-anchored adaptive search against calibrated Stockfish; refined together after discovering Stockfish 18's UCI_Elo floor is 1320, far above bench models. Final instrument: internal lab scale anchored at random-legal = 0, sub-basement rungs via Stockfish-skill-0/random mixtures (25/50/75%) and node caps (1/4/16/64), adjacent-pair self-calibration at 100 games converting scores to Elo-style gaps, then per-model probe-ascent (20 games/rung, stop under 30%) plus 100-game informative rungs, maximum-likelihood rating with 95% profile-likelihood interval. Lab ratings are internal and monotone — not Lichess or FIDE comparable. Rationale: decisive engine play bypasses the draw wall that saturated head-to-head matches (88/100 draws in Experiment 4).

## Experiments 5A/5B — depth and the value objective (preregistered 2026-07-30, night run)

**Run A (capacity):** d=128, 6 layers (~1.8M params, trunk finally 2:1 over head), unfiltered january-10k, batch 256 / lr 3e-4 / 3 epochs / same seed — one variable (architecture) vs the rated baseline (val 10.5%, lab rating 0). *Learner:* validation 13%, lab rating 100, confidence 40%. *Teacher:* validation ~12%, rating ~+60, confidence 45%.

**Run B (objective):** identical to Run A plus value head at weight 0.5. One variable (objective) vs Run A. *Learner:* policy validation goes DOWN (magnitude/confidence unstated). *Teacher:* policy validation within ±0.3 points of Run A (trunk sharing neutral-to-positive), value top-1 50–55%, lab rating ≈ Run A's since greedy inference ignores the value head; 55%.

**Also queued:** batch-size calibration (256/1024/4096 with linearly scaled LR, bench-S, 1 epoch) to set the future bench default; training data now lives device-resident (smoke: 2.9s vs 4.5s at 200 games).

**Results (2026-07-31):** Run A: validation top-1 **17.46%** (from 10.5% — both forecasts far too timid: learner said 13%, teacher 12%), lab rating **58** [12.5, 102] (learner said 100 — inside the CI but above the point estimate; teacher's 60 nearly exact). Depth/width was the largest single improvement the bench has produced. Run B: policy validation **17.49%** — the value head cost the policy nothing (learner predicted a drop — miss; teacher's ±0.3 band — hit), value top-1 55.4%, rating 35 [−15.5, 85.5] ≈ Run A within noise, as predicted (greedy inference never consults the value head). Speed calibration at bench-S: wall time flat across batch sizes (256: 93s; 1024: 100s; 4096: 100s) — device residency had already removed the launch overhead — but batch 1024 + lr 1.2e-3 reached 8.26% in one epoch vs 6.81% for the old recipe: the effective-LR finding. First run of the fixed recipe suite died on a zsh word-splitting quirk; rerun in flight.

## Decision 8 — fleet recipe (2026-07-31, overnight)

Recipe suite at d=128/L=6, 1 epoch, batch 1024 / lr 1.2e-3: control 491.6s / 10.53%; +fused AdamW 485.6s / 10.63%; +torch.compile 539.5s / 10.35% (the predicted MPS reduction-kernel regression — rejected); +compile+bf16 479.7s / 10.66%. Adopted fleet flags: `--fused-adam --precision bf16`, no compile. Overnight fleet launched per lab/OVERNIGHT.md: 43 atomic trainings (fleet_train.sh, sequential GPU) with a pipelined ladder-rating queue (fleet_rate.sh, CPU) plus 4 inference-decode re-ratings; control is 1 epoch on the enriched 40k-shard's first 10k games. Best-shot #48 composed after results.

## Fleet results — headline (2026-07-31, dawn)

All 43 trainings and 47 ratings completed without failure. **The inference layer dwarfed every training intervention:** the value-0.5 checkpoint rates −25 decoded greedily and **630** [586, 677] with 1-ply value search + contempt — a +650-point swing from test-time compute alone, versus +179 for the best training-side change (4× data). Pure value-only + search: 488. Training-side ranking: 40k data 179, 3 epochs 171, elite fine-tune 119, decisive fine-tune 76, MLP 61 (trained in 14s — 33× faster than the transformer at this scale), 20k data 61, history-2 60, per-square readout 53 (with 30% fewer params). Most other greedy-decoded atomics are inside a ±50 mush — honestly indistinguishable at ~250 rating games. Self-play arms produced too few decisive games to matter (draw-wall limits greedy self-play data). Val-top-1 comparisons across filtered shards remain population-incomparable; ladder only. Best-shot #48 (40k + history-8 + per-square + value head, search+contempt decode) and top-8 round-robin running.

## Fleet finale (2026-07-31, morning)

Best-shot #48 (40k games × history-8 × per-square readout × value 0.5, one epoch, 35 min): validation top-1 **27.79%** at 1.27M parameters — within a point of tournament model 0004's 28.5% at 10.6M — and lab rating **596.5** [552, 641.5] under search+contempt, statistically tied with 28-contempt (630.5): at 1-ply, a much better policy barely moves match strength, because the value head decides. Top-8 round-robin (20 games/pair): 28-contempt went 105–5 in decisive games (20-0 vs 05-20k); greedy-decoded peers drew each other massively (multiple 0-20-0 lines); joint Elo ordering matches the ladder. Results artifact published (claude.ai/code/artifact/573a2cda-603a-46af-b661-a2e7daf8de55). Follow-ups queued for daylight: multi-seed re-runs of winners at bench-M, cosine-underperformance re-check, search-decode self-play regeneration, 2-ply/MCTS prototype, browser-package legality review for search at inference.

## Experiment 3 — value-head leakage (parked, 2026-07-30)

Add a win/draw/loss value head; measure outcome-accuracy under both splits. Teacher's sealed prediction: position split inflates outcome accuracy by +8 to +20 points, 75%. Lab prepare now emits a `result` column, so this runs whenever unparked. Parked at learner's request to prioritize the end-to-end pipeline.
