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

## Decision 9 — per-square readout is bench standard; Experiment 49 (2026-07-31)

Per-square readout ratified as the default going forward (fleet evidence: competitive-to-better with 30% fewer parameters; used in best-shot #48). To combine it with the MLP the trunk was redesigned token-in → token-out (one wide Linear reshaped to 65 tokens), so all readout heads and the browser adapter work unchanged; old arch=mlp checkpoints (15-mlp) no longer reload under the new module names. **Experiment 49 — MLP composite:** MLP trunk (hidden 768, 2 layers, 14.2M params) × 40k games × history-8 × per-square × value 0.5, sized by a 1-epoch timing probe (149.7s/epoch, 19.17% top-1) to 11 epochs ≈ 27 min — the "30 minutes of MLP" bet: 14× more epochs per wall-minute than the transformer composite. Rating under search+contempt to compare against #48's 596.5. Result: pending.

## Experiment 50 — MLP performance push (2026-07-31)

Diagnosis-driven squeeze on #49's 6-point overfit gap: data ×3 (120k games ≈ 7M positions, 4 epochs at matched wall time) + dropout 0.1 (first bench model with a gap for it to close). Declared as a bundled performance push, not an attribution experiment. Levers deliberately NOT pulled: capacity (amplifies overfit before data re-levels), cosine (fleet anomaly unexplained), mirror augmentation (castling breaks the symmetry). Teacher's sealed prediction: val top-1 29–31%, gap < 2 points, beats #48's 27.8%. **#49 result for the record:** wall 1212s, train 32.03 / val 26.03 (gap 6.0), value 54.8, rating 506 [461, 552] — below #48's 596 with intervals touching at exactly 552; at matched wall-clock and 40k games the transformer converts compute to generalization where the MLP converts it to recall. The crossover is real and sits near this scale.

## Experiments 51–52 — better judge, deeper search (2026-07-31)

**#51 judge:** value weight 1.0 on 120k shard — policy 29.24% (no cost vs #50's 29.36), value top-1 55.89% (+0.22, looked like nothing). **#52 beam search:** policy-pruned minimax (root value screen top-8, policy beam 6, depth 4, batched on MPS, ~20–45s/game) in lab/match.py, plus --opponent-search and calibrated --stockfish-elo opponents.

**Battery (20 games each):** 4-ply vs 1-ply same weights: **18.5/20** (teacher predicted 14+, hit — depth ≈ +430 Elo-equivalent over 1-ply). 4-ply vs Stockfish UCI_Elo 1320: **0.5/20** (teacher predicted 3–7, too optimistic — first calibrated-scale reading; absolute strength still far below 1320, one survived draw). New judge vs old judge at 1-ply: **13/20** (teacher predicted noise, too pessimistic — the +0.22% aggregate value accuracy hid a real decision-quality gain; 20-game CI keeps this suggestive). Lesson pair: aggregate metrics can hide decision-relevant improvement, and depth compounds harder than any training lever measured so far.

## Experiment 53 — exhibition submission (2026-07-31, evening)

Deadline build for the exhibition match. Model: MLP composite (hidden 768, d128 embeddings, history-8, per-square readout, value weight 1.0, dropout 0.1, 14.2M params) chosen over the transformer for WASM searchability (~5× cheaper per evaluated position); trained 6 epochs on a fresh 240k-game January shard (15.9M positions, 69 minutes, per-epoch deadline checkpoints). Result: **32.39% validation top-1** (lab record, +3 over #50), value 56.4%, gap 0.9. Adapter: depth-4 beam search (root value screen 6, policy beam 5, contempt 0.15) ported to JavaScript, validated on ONNX Runtime Web at ~0.7s/move, 57 MB package. Sanity match vs #51 at identical beam-4: **17.5/20**. Compliance: pinned January data only, fresh lineage, trivial FLOPs, package under cap. Published: `peterwooden/chess-gpt-beam-exhibition-53` @ `355e21607668248070873400f610f1e1c04810f7`. Feb/Mar archive downloads running for the post-exhibition scaling phase.

## Experiment 54 — lab graduate vs pre-lab champion (2026-07-31)

50 games, both as submitted: #53 (beam-4 entry) vs tournament candidate 0004 (greedy legal argmax). Result: **44/50 — 39 wins, 10 draws, 1 loss (88%)**, ≈ +340 Elo. Teacher's sealed 45–48 at 75% was a near-miss low (0004's draw-scraping was slightly better than credited). Perspective: the entire pre-lab strong-winner program (full compute budget, matched-FLOP design) moved 0004 to 59.5/100 over its predecessor; two days of lab plus one evening of training moved 88/100 over 0004 — with the inference layer carrying most of it. The single loss is queued for autopsy.

## Experiments 55–56 — the hour that taught instead of improved (2026-07-31, late)

Plan: warm-start #53 on 240k fresh January games (3 epochs) then elite fine-tune. Both stages failed the beam-vs-beam sanity gate against #53: continuation #55 scored 7.5/20 despite *better* validation top-1 (33.1 vs 32.4) and equal value accuracy; polished #56 scored 3/20. Three lessons, all measured: (1) the first #56 attempt poisoned the value head via winner-only label leakage (value_top1 = 100% — "side to move wins"); caught by reading metrics, rerun policy-only; (2) policy-only fine-tuning still shifts the shared trunk under a frozen value head — a conclusion measured under greedy decode (#39's +106) does not transfer to search decode; (3) warm-starting a converged model with fresh optimizer state at full constant LR improves aggregate imitation metrics while degrading match strength — proper continuation needs decayed LR + warmup. Teacher's 13–15/20 prediction was the worst miss of the course. #53 remains champion and the published exhibition ref stands. Corrected-continuation recipe queued for the scaling phase.

## Experiment 57 — first cloud model (2026-07-31, night)

Maiden HF Jobs run: shards pushed to a private dataset repo (670 MB), standalone state-dict-compatible trainer (lab/cloud_train.py) launched on an A100 (~$0.65, 15.2 min). Hidden-1152 MLP composite (21.8M params) on all 480k prepared games: **34.41% validation top-1** (record), value 56.7%, gap 0.5. Gate vs #53 at beam-4 both sides: **13–7, zero draws** — passed. Packaged with the time-managed quiescence adapter (87.1 MB of the 100 MB cap), validator + tactical probes green. Published: `peterwooden/chess-gpt-cloud-beam-57` @ `023280b67f08e2ad9996f115babe757e6d62533f`. Gotcha logged: CUDA checkpoints need map_location on Apple-silicon load (fixed in lab/match.py). Velocity note: A100 trained 171M position-passes in the time the M4 does ~20M — the scaling phase is now a booking, not a plan.

## Fleet 2 — recipe sweep, opening squadron, and a new champion (2026-07-31, night)

42 parallel A100 arms at matched 2-epoch budgets (~$25 total), triplicate control noise floor [31.09, 31.86] top-1. **Headline: the LR schedule was the sleeping giant — warmup 2% + cosine gained +6.9 top-1 at identical compute**, more than tripling the data ever did; the entire schedule family won; Fleet 1's cosine anomaly resolved as an overfitting-regime artifact. Secondary wins: schedule-free AdamW, GELU, wide-shallow trunk, lower constant LR. Failures: Lion/b4096/ReLU²/lr2.4e-3 diverged; grad-clip 1.0 cost 3.3 points; short-first curriculum destroyed the value head (5.9% — target-distribution drift); elite-50% harmful; torch.compile +10% on CUDA (adopted). Opening squadron: champion plays book-top-3 only **26%** of the time while its own policy head plays 99% — the noisy early-game value head outvotes a well-read policy; book/policy/flat cures all strength-neutral; **book adopted** (instant moves bank clock for the time-managed search). **Finalists at full budget:** f1 (warmup+cosine only) 41.76% top-1, **gate 14.5/20 over #57 — new champion**; f2 (+GELU+wide) 42.64% — best imitation ever measured — **failed its gate 8/20**: second composition-trap catch. Standing rule reinforced: adopt single changes, gate each. Artifact: claude.ai/code/artifact/48c8a817-94ec-4403-8c7f-28f215cef18c. Deferred: self-distill arm (harness error), 960k duel, dedup at scale, search-labeled value.

## Fleet 3 — broad technique sweep on the f1 baseline (2026-08-01, small hours)

47 arms, zero failures, tightest control triplicate yet (38.69–38.83). Winners: gated MLP +2.0 (and faster), Muon-v2 +0.9 with the value-head damage fixed by protecting heads, hidden-1536 +0.8 (fp16 export now gates capacity), input LayerNorm / batch-512 / warmup-5% small-real. Nuggets: **960k-once ≈ 480k-twice** (second epochs free at this scale — scaling-pilot de-risked); square-shuffle bad-idea landed in the noise band (the MLP has no spatial wiring — coordinates are learned labels); castling-rights and halfmove-clock ablations cost nothing; history order matters (−5.5 shuffled) but length saturates at 4; 5% label noise costs 0.4. Graveyard: SGD, phase-diet data, grad noise, low-rank, rank-file embeddings, skip-openings, 2-epoch distill. **Finalists both gated at exactly 10.0/20 vs f1 despite record imitation (f3a-gated 42.94%) — policy accuracy is measured-saturated as a strength lever; every judge sits ~57.8% and the judge decides. Frontier: search-labeled value training, deeper search, fp16 scale. f1 remains champion.** Artifact updated (Fleets 2 & 3). Credits ≈ $35 remain.

## Experiment 58 — search-teaches-judge, cheap controlled test (2026-08-01)

Learner challenged the search-label mechanism ("outcomes already carry the signal — isn't noise just averaging?") and demanded a cheap controlled test. Design: 200k identical positions, champion f1, value-head-only fine-tune (trunk frozen), only label source differing — A: own 1-ply search backups; B: game outcomes (control); mean label disagreement 0.435. Gates vs untouched f1: **A 10.5/20, B 8.0/20** — A even with champion (teacher predicted 13–15: miss), A−B = +2.5 (noise-scale, direction only). Verdict: the cheap variant is null. Post-mortem: 1-ply labels are largely redundant with play-time search's own first ply, and a frozen-trunk linear head cannot learn features the trunk lacks. Deep-label + full-training variant remains untested and now bears the burden of proof — fold into the scaling run if at all. Cost: $0, ~1 hour. Student skepticism 1, teacher mechanism 0.

## Experiment 59 — the capstone (2026-08-02/03)

Fully co-designed submission candidate; every choice the learner's: transformer 8L·d256·8h·FFN1×·learned attention bias (3.66M params), **perspective flip** (canonicalized side-to-move; verified against python-chess mirror oracle on 4,000 positions), **elo data ladder** (stage 1: 2M games both≥1600, 143.5M positions; stage 2: 30k games both≥2600 at LR/10 warm), **no book** — policy-only first 6 plies plus a rules-only mate scan. Stage 1: val 53.68% / value **59.04%** (first break past the 57.8 judge plateau), 104 min A100. **Gates: stage 1 beat champion f1 20–0; stage 2 also 20–0; head-to-head even (9–11) → stage 1 ships** (shorter lineage, better judge). Lineage compliance: 129.1M positions × ~1.53 GFLOPs ≈ **1.98×10¹⁷ profiler-FLOPs ≈ 19.8% of budget**, fresh init, pinned Jan data. Package: 14.85 MB, flip-aware bookless adapter, all probes green incl. black-flip mate and search paths; opening mate-guard added after a probe caught the policy-window declining a mate-in-1. Published: `peterwooden/chess-gpt-capstone-59` @ `9e312f0316a1062d40ebf0f18764898ac5a7496a`. Incidents: /tmp sweeper ate the uv runtime (rebuilt); packager taught sweep-checkpoint format.

## Experiment 60 — the CNN hypothesis, dose-response ladder (2026-08-03)

Learner's hypothesis: spatial invariance + parameter sharing → CNNs learn chess well. Five matched-budget arms (flipped ≥1600 data, ~52M passes): classic 5×5-stem/3×3-resnet/global-pool 50.21; +rook rays 50.69; +full compass (shear diagonals, oracle-verified) & free knight leap **50.77**; +true knight mask 50.73 (null — free 5×5 had learned the right taps); transformer control **51.71**. Verdicts: geometry priors climbed monotonically (+0.56, mostly rook rays — sliding-piece blindness confirmed as the CNN's handicap); transformer wins per position; **per tournament-FLOP it is a statistical tie** (CNN ~0.9 vs 1.5 GFLOPs/pos; scaling-adjusted TF ≈ 50.6 at CNN's spend), and CNN wall-cost is ~40% lower. Teacher's 70% "transformer wins" survives only per-position. Knight-masked kernel = clean prior-vs-learning null. x5b rerun launched with checkpoint for the kernel autopsy + replication.

## Experiment 60b — kernel autopsy (2026-08-03)

x5b replicated x5 exactly (50.75/1.5496 vs 50.77/1.5497). Weight census of the trained 5×5 stem: **24 knight-detector filters (best: 89% of mass on the 8 knight taps), 72 rook-cross filters, 60 bishop-diagonal filters** — piece-movement geometry self-organized from free kernels in layer one. The mid-network leap group learned no knight structure (max 0.45): by that depth channels are abstract, geometry extraction belongs to the stem — which mechanistically explains the knight-mask null. Lesson sealed: provide the support, let the structure grow; and first-layer conv kernels are a readable window into what the network knows.

## Experiment 61 — the modern CNN at budget, and the silicon lesson (2026-08-04)

ConvNeXt-style chess CNN (7×7 depthwise, inverted bottleneck, GELU, SE, mean+max pooling, depthwise compass rays with learnable gates; 14.8M params) launched at capstone-matched budget (1.98×10¹⁷, same data/epoch). **Silicon lesson:** depthwise convs ran memory-bound on A100 at ~half projected throughput; HF's job timeout was not enforced; the run was cancelled at 7h43m (~$19) at ~75% of the epoch via tripwire. **CORRECTED (2026-08-04): the run DIVERGED — 312 weight tensors NaN/Inf** (verified on CPU; likely bf16 × untested GroupNorm/max-pool/tanh-gate stack at full LR, no clipping). The 0.5/20 gate measured a NaN-poisoned policy and is **retracted**; the architecture question at budget scale is **untested**, and the earlier "outclassed at scale / regime-flip #5" claim was wrong. Standing conclusions: (1) paper-FLOPs ≠ silicon-FLOPs, architecture-specific — pilot curves in wall-dollars alongside FLOPs; (2) long jobs get self-enforced watchdogs; (3) new architectures at new scales get stabilizers (clip, lower LR) and stability smokes; (4) partial checkpoints must be per-mark, not overwritten — the pre-divergence snapshots were lost. Transformer remains primary for the pilot on the evidence that exists; a stabilized CNN rematch costs ~$15–19 if ever wanted. Credits ≈ $15 remain.

## Experiment 3 — value-head leakage (parked, 2026-07-30)

Add a win/draw/loss value head; measure outcome-accuracy under both splits. Teacher's sealed prediction: position split inflates outcome accuracy by +8 to +20 points, 75%. Lab prepare now emits a `result` column, so this runs whenever unparked. Parked at learner's request to prioritize the end-to-end pipeline.
