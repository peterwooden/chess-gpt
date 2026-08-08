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

## Experiment 61 epilogue — the full root cause, and an economic terminus (2026-08-04)

Five-probe elimination tree, ~$25 total: block structure ✗ (LayerScale+pre-norm added anyway, correct), fused-AdamW×channels-last ✗, channels-last ✗, naive clipping **partially guilty** (unguarded `clip_grad_norm_` propagates one bad gradient into all parameters — guarded skip-step now standard), and the root cause: **torch.compile miscompiles the modern block's backward on CUDA, producing non-finite gradients every step** (proof: gradients verified clean on CPU in fp32 and bf16; every failed run compiled, every healthy smoke didn't; the guarded run trained nothing because it skipped every poisoned step — finite weights, exactly-uniform loss ln(4272)). Fix is `compile: false`; execution is economics: credits exhausted (~$100 total), M4 fallback measured at 95 pos/s — days per slice. **The modern CNN closes untrained: root cause documented, fix known, one ~$12–15 clean run from its answer if credits ever return.** The transformer's primacy rests on the small-scale ladder and capstone-59's record, not on this chapter. Meta-lesson for the tournament lineage: every stabilizer and speed flag is itself an experiment — smoke them on the target hardware, guarded, before they touch a budget run.

## Experiment 62 — CNN closed on economics (2026-08-04)

Compile-off smoke trained cleanly (39.9% at 5.2M positions, ~$1) — the fix works. But measured throughput: CNN 2,037 pos/s vs transformer 20,700 pos/s at identical paper-FLOPs — 10x slower on silicon, compile unfixable (selective-compile smoke: still poisoned, and only 5% faster anyway). The 20% run was killed at ~2h ($5): its outcome could not change any decision — the CNN loses 10x per dollar even if it ties per FLOP. Final verdict: transformer is the tournament architecture on economics, independent of quality. CNN program closed with: dose-response ladder, kernel autopsy, five debugging lessons, and this throughput datum.

## Experiment 63 — fresh-agent CNN + bilinear head, two-stage (2026-08-05)

A context-free subagent co-designed cnn2 (36 dense planes, 5×5 stem, pre-activation zero-init blocks, dense 1×15/15×1 ray convs, BEB global broadcast, bilinear from·to policy head, spatial value head). Stage 1 (5.2M positions, three arms): cnn2 44.28 / tf+bilinear 45.10 / tf-control 40.13 — the **bilinear head alone is worth ~+5 top-1 at small scale** (agent predicted +0.5–1.5; per-square readout was starving every model), and cnn2 beats the plain transformer trunk-for-trunk. Stage 2 (129M positions, 20% budget each): **cap64-tf-bilinear 54.12 / 1.406 loss** (capstone-59: 53.68), value 59.04, 105 min; **cap63-cnn2 53.21 / 1.444**, value 58.82, 78 min (25% cheaper wall — the fresh design fixed the modern-CNN 10× slowdown). Bilinear's +5 at 5.2M shrank to +0.44 at 129M: a data-efficiency win that mostly converges away. Gates vs capstone-59 (20 games, beam both sides): **cap63 5.5–14.5 (fails)**, **cap64 11–9 (narrow pass, within noise — not the 20-0 of prior adoptions)**. Metric authority (loss → top-1 → gate): cap64 leads on all three, none decisively. CNN verdict: real architecture, wrong side of 0.5 points; closed with honor. Export note: cnn2 plane builder rewritten concat-style for ONNX (bit-exact parity verified). Published: `peterwooden/chess-gpt-cap64-tf-bilinear` @ `cc840e2c15685fb6ca2b107fb204e74d5f149ea6`, `peterwooden/chess-gpt-cap63-cnn2` @ `1f11da86da81a56e25e871d35040bf679889c62a`. Adoption call deferred to learner: cap64 as new champion, or hold capstone-59 pending a larger gate.

## Experiment 65 — QKV weight-tying symmetry, double-controlled (2026-08-06)

**Question:** attention gives Q, K and V their own projection of the same input. How much of that separation is load-bearing? Tie each pair (and all three) and measure what it costs.

**Arms (5):** control `none` (Q, K, V independent), `qk` (Q = K, symmetric scores), `kv` (K = V, address is payload), `qv` (Q = V), `qkv` (all one projection). Everything else held at cap64's recipe: transformer 8L·d256·8h·FFN1×, learned attention bias, bilinear head, flip, elo1600, batch 1024, cosine + 5% warmup, seed 20260730. The control uses split per-role Linears rather than the legacy fused `qkv` weight, so all five arms share one code path; it lands at 3,713,011 parameters — bit-identical in count to cap64, which confirms the control is architecturally the same model.

**Two budgets, because tying cuts compute as well as weights** (each distinct projection is computed once, so a tie is cheaper per step, and a fixed step count would silently hand the control more FLOPs):

| arm | params | Δ params | fwd FLOPs/pos | vs control | equal-FLOPs steps | equal-time |
|---|---|---|---|---|---|---|
| none (control) | 3,713,011 | — | 512,591,360 | 100.0% | 36,000 | 1800 s |
| qk / kv / qv | 3,188,723 | −524,288 | 436,045,312 | 85.1% | 42,320 | 1800 s |
| qkv | 2,664,435 | −1,048,576 | 359,499,264 | 70.1% | 51,331 | 1800 s |

Ten runs: an equal-FLOPs family (step counts fixed a priori from the analytic FLOP model, ~5.7e16 each) and an equal-wall-clock family (new `time_budget_s` knob: the trainer measures its own steady-state rate over steps 20→70, after compile, then sets `total_steps` so the cosine schedule completes exactly at the budget). Running the control in both families is the consistency check — if its two cells disagree, the families are not comparable.

**Preregistered predictions (teacher):** (1) control ≥ every tie on val loss at equal FLOPs, 65% — but by a small margin, under 0.02 loss / 0.4 top-1; (2) `qk` is the least damaging tie, 60% — chess relations are largely symmetric (if A stands on B's ray, B stands on A's ray), so a symmetric score matrix loses less than it looks; (3) `kv` and `qv` cost more than `qk`, because merging address with payload constrains what a head can transport; (4) `qkv` is worst per step but claws most of it back at equal FLOPs via 1.43× the steps — the interesting cell is whether it closes the gap entirely; (5) the equal-time family ranks ties slightly worse than the equal-FLOPs family, because the FLOP saving will not fully convert to wall-clock (attention scores, optimizer and data movement are unchanged). Learner's predictions not yet recorded.

**Result (2026-08-06).** Consistency check passes first: the control's throughput is 25.78 steps/s in the FLOPs family and 25.86 steps/s in the time family (0.3% apart), so the two families are on the same footing.

| arm | equal FLOPs (66.1 PFLOPs) | | equal wall-clock (1800 s) | | |
|---|---|---|---|---|---|
| | steps / loss / top-1 | Δ loss | steps / loss / top-1 | Δ loss | PFLOPs spent |
| none (control) | 42,000 / 1.4660 / 52.62% | — | 46,547 / 1.4590 / 52.79% | — | 73.3 |
| Q=K | 49,373 / 1.4668 / 52.59% | +0.0008 | 48,325 / 1.4684 / 52.55% | +0.0094 | 64.7 |
| K=V | 49,373 / **1.4574** / **52.83%** | **−0.0085** | 48,650 / 1.4579 / 52.80% | −0.0011 | 65.2 |
| Q=V | 49,373 / 1.4670 / 52.59% | +0.0010 | 48,470 / 1.4702 / 52.50% | +0.0112 | 64.9 |
| Q=K=V | 59,886 / 1.4672 / 52.56% | +0.0012 | 51,419 / 1.4791 / 52.27% | +0.0201 | 56.8 |

**The two budgets answer different questions, and the control's own two cells let us ask a third.** Those cells define a compute-scaling slope of −0.0015 loss per 1,000 steps; scoring every arm against that curve at its own step count gives the *per-step* penalty of a tie, and each arm's two independent runs agree closely — an internal replication we did not pay for:

| tie | per-step penalty (FLOPs cell / time cell) | verdict |
|---|---|---|
| K=V | +0.0028 / +0.0021 | nearly free |
| Q=K | +0.0122 / +0.0121 | ~5x the K=V damage |
| Q=V | +0.0124 / +0.0142 | same as Q=K |
| Q=K=V | +0.0287 / +0.0276 | ≈ Q=K + Q=V, roughly additive |

So three rankings, all true: **per step** the control wins and every tie costs something; **per FLOP** K=V wins outright and the rest are a wash (each tie's FLOP saving buys back roughly what it lost); **per second** the control wins, because only ~22–25% of the theoretical FLOP saving converts to throughput (Q=K=V spends 0.70x the FLOPs but runs just 1.105x faster — attention scores, norms, embeddings, optimizer and memory traffic don't shrink when projections do). The tournament caps FLOPs, not seconds, so the middle column is the one that governs submissions; the right-hand column governs lab iteration speed.

**Why K=V is nearly free, mechanistically:** the output projection sits after the mixing, so the value path's effective map is W_out·W_V. Tying V to K makes it W_out·W_K — and W_out is still free, so the composition can express any linear map the untied pair could. Tying loses almost nothing on the value path; it only forces one projection to serve both matching and payload. Q=K is different in kind: it forces the score matrix to be x_i^T W^T W x_j, i.e. **symmetric**, and no downstream linear map can undo that. The measured costs match the argument — and Q=K=V's penalty is about the sum of Q=K's and Q=V's.

**Prediction scorecard (teacher): 2 of 5.** (1) FAILED — the control did *not* beat every tie at equal FLOPs; K=V beat it. (2) FAILED — Q=K was not the least damaging tie; K=V was, by 5x. (3) FAILED for K=V (predicted worse than Q=K, was much better), correct for Q=V. (4) CORRECT — Q=K=V is worst per step and closes the gap at equal FLOPs (+0.0012). (5) CORRECT, and strongly — the equal-time family ranks ties worse, with only a quarter of the theoretical saving converting. The reasoning behind (1)–(3) treated "address vs payload" as the load-bearing distinction; the actual structure is that **anything the output projection can absorb is cheap, and anything it cannot is expensive.** Symmetry of the score matrix, which sounded like the drastic constraint, costs 0.012 — real but small, because the learned positional bias B already carries the asymmetry that chess geometry needs.

**Follow-up before adopting:** K=V is a free 14% parameter cut and a small per-FLOP win, but every cell is a single seed. The per-step penalties replicate across two horizons, which supports the *ordering*; it does not bound seed variance. Cheapest confirmation is 2 seed replicates of control and K=V at the equal-FLOPs budget (~1 GPU-hour) before folding K=V into the next full-budget lineage run.

## Experiment 66 — K=V vs control at 10% budget (2026-08-06)

**Question:** does K=V's per-FLOP win at 6.6e16 survive a 1.5x scale-up to 1e17 (10% of the tournament budget per arm, cap64's recipe, same seed, single epoch — no data reuse)? Step counts fixed a priori from the analytic FLOP model so both arms land on 1.0000e17 exactly.

| arm | params | steps | train_s | val_loss | top-1 | value |
|---|---|---|---|---|---|---|
| control (untied) | 3,713,011 | 63,505 | 2459 | 1.4408 | 53.24% | 58.85% |
| **K=V** | 3,188,723 | 74,653 | 2755 | **1.4335** | **53.42%** | **58.89%** |

**K=V leads here by −0.0073 loss / +0.18 top-1 on 14% fewer parameters — but see Experiment 65b below, which retracts the inference drawn from this.** Both this pair and the Exp 65 pair used seed 20260730; at seed 20260807 the sign flips, so the honest reading is K=V ≈ control within seed noise. The claim originally written here ("4 of 4 cells agree on the sign") double-counted one seed. Checkpoints saved (`results3/tie66-none.pt`, `results3/tie66-kv.pt`) for a match gate. Caveat unchanged: single seed throughout; the control here uses the split-projection code path (same architecture as cap64, different init stream than the fused layer).

## Experiment 67 — training-data slicing screen, 9 arms (preregistered 2026-08-07)

**Question:** which training-data slices help, hurt, or do nothing at fixed compute? The champion's both>1600 floor was adopted inside capstone-59's many-variable bundle and has never been isolated; the other filters have never been tested at the modern recipe.

**Design.** Nine arms, one data variable each, everything else frozen at cap64's recipe (transformer 8L·d256·8h·FFN1×, attention bias, bilinear head, flip, untied QKV, seed 20260730, batch 1024, cosine + warmup). All arms 42,000 steps = 43.0M positions = 6.61e16 FLOPs — identical architecture means identical FLOPs, so data is the only variable. Every arm's pool is a fresh 750k-game January shard (~45–52M positions), so training is a single pass with no reuse and game-diversity is matched — the control deliberately does NOT reuse the 2M-game elo1600 set, which would hand it a 3× larger game pool. **Comparability fix:** per-arm 10% holdouts are population-incomparable (Exp 4's lesson), so a new `val_shard` option in cloud_sweep evaluates every arm on one frozen shard: April, both>1600, 25k games, 1,792,588 positions (`shards/slice67-val-april.parquet`). April is the designated validation month and is never trained on.

| arm | filter | prep note |
|---|---|---|
| control | both>1600 | champion policy at matched pool size |
| unfiltered | none | completes the ladder downward |
| elo1800 / elo2000 / elo2200 | dose-response | header-band census: >1800 = 35.7% of all January games, >2000 = 18.1%, >2200 = 7.3% — every rung has a deep pool; 2200 subset pre-skimmed by awk header filter |
| nobullet | >1600 + base time ≥180s | flag existed, never run |
| decisive | >1600 + drop draws | value-target population shifts (no draw labels) |
| noforfeit | >1600 + drop Termination="Time forfeit" games | new prepare.py flag; whole-game drop protects value labels at the cost of valid data (a flag in a lost position is a correct label; a flag while winning is noise — indistinguishable without eval). Finer variant (keep positions for policy, mask value loss) deferred unless this arm shows signal |
| dedup64 | >1600 + identical positions capped at 64 occurrences per shard | cap=64 chosen as moderate dose; val-loss interpretation caveat — dedup underweights openings relative to the natural val distribution |

Known caveats, accepted for a screen: single seed; arms sample different depths of the (roughly chronology-ordered) archive; nobullet and noforfeit overlap mechanistically (most forfeits are bullet); April acceptance for both>1600 measured at 53%. Cost: ~9 × 30 min A100 ≈ $10–12.

**Protocol change (2026-08-07):** the learner retired the shared-prediction ritual — its purpose was coached theory-building, which they consider complete; they now predict privately without recording here. Teacher predictions remain sealed before results, as before. This supersedes the NOTES.md line requiring recorded learner predictions.

**Teacher predictions (sealed before any result):**
1. Ladder shape: val loss bottoms at **elo1800**; elo2000 ≈ control; unfiltered clearly worst of the ladder (+0.02–0.04 loss vs control, 70%); elo2200 between control and 1800's improvement but not the minimum. Confidence 1800 is the exact minimum: 45%.
2. nobullet: small improvement, −0.004 to −0.010 loss, 55% — extra thinking time buys move quality, partially offset by the val set itself containing bullet-style moves.
3. decisive: policy slightly worse (+0.003–0.010); **value_top1 markedly worse (−3 to −8 points)** because a 3-class value head trained without draw labels can never predict the draw class that April contains. 70%.
4. noforfeit: policy ≈ control (±0.003); value_top1 better by +0.2–0.8. 55% on the value direction (the learner's own mechanism).
5. dedup64: val loss worse (+0.01–0.03) for distribution reasons, not strength reasons — flagged as the arm where val loss most misleads. 65%.
6. Headline: **no slicing arm beats control by more than 0.010 val loss** — at fixed FLOPs, slicing is mostly distribution-matching, not quality-mining. 55%.

**Learner predictions:** held privately by the learner's choice (see protocol change above).

**Results (2026-08-07).** All nine arms exactly 42,000 steps, ~27 min each, ~$10 total. Frozen April both>1600 val shard throughout. Prep censuses worth keeping: both>1600 acceptance 54.8% (January), draws only ~4.6% of that pool, time-forfeit games ~35% of it, bullet ~44% of it; >2200 games run longer (78.5 pos/game vs 71.4 at control).

| arm | val_loss | Δ | top-1 | value_top1 |
|---|---|---|---|---|
| decisive | **1.4797** | **−0.0032** | **52.22%** | 58.17% |
| control (>1600) | 1.4829 | — | 52.10% | 58.47% |
| elo1800 | 1.4848 | +0.0019 | 52.10% | 58.48% |
| noforfeit | 1.4875 | +0.0046 | 52.16% | 58.47% |
| dedup64 | 1.4914 | +0.0085 | 51.71% | 58.45% |
| elo2000 | 1.4930 | +0.0101 | 51.92% | 58.47% |
| nobullet | 1.4949 | +0.0120 | 52.10% | 58.54% |
| unfiltered | 1.4950 | +0.0121 | 51.83% | 58.40% |
| elo2200 | 1.5089 | +0.0260 | 51.55% | 58.44% |

**Findings.** (1) The Elo ladder is monotone *downward* from 1600 — every elitism step hurts, and elo2200 is the worst arm in the screen, worse than no filter at all. On this evaluation the story is distribution-matching, not quality-mining: the champion's 1600 floor sits at or near the optimum, and "train on better players" is dominated by distribution shift beyond it. (2) The only arm to beat control is **decisive** — removing the ~4.6% of games that are draws buys −0.0032 loss / +0.12 top-1 on an all-games val set, at a value-head cost of only −0.30 (a head that cannot predict the draw class loses little because it rarely argmaxed draw anyway). A ~5% dose producing a visible effect makes draw moves unusually low-value training signal per game. (3) noforfeit is null where it was supposed to act: value_top1 58.47 vs control's 58.47, policy slightly worse — dropping 35% of the pool for outcome-label hygiene bought nothing measurable at this scale/metric. (4) nobullet: loss +0.0120 with top-1 exactly flat — it mispredicts (or is miscalibrated on) the bullet-styled moves that fill ~44% of the val population while matching control on argmax; its value head is the best of the nine (58.54), a whisper in the saturated-judge band. (5) All value heads sit in a 0.37-point band — judge saturation again, as in Fleet 3.

**Teacher scorecard: 2 of 6.** Ladder-minimum-at-1800 MISS (minimum is control); unfiltered-worst-of-ladder MISS (elo2200 is, by 2×); nobullet-improves MISS (+0.0120 the wrong way); decisive DOUBLE MISS (predicted policy slightly worse and value −3 to −8, got best-in-screen policy and value −0.3 — the draw-rate census that invalidated the value prediction was measured *after* sealing); noforfeit value +0.2–0.8 MISS (exact null); dedup64-worse HIT (+0.0085, direction and near-band); no-arm-beats-control-by->0.010 headline HIT (best win is −0.0032).

**Caveats and follow-ups.** Single seed everywhere, and Experiment 65b (below) makes that binding: the control alone moved 0.0042 loss between seeds, so **decisive's −0.0032 is inside single-seed noise and is a direction to confirm, not a win** — finding (2) above is provisional until ≥3 seeds or a paired design says otherwise. The same standard clears the ladder's large effects (unfiltered +0.012, elo2000 +0.010, nobullet +0.012, elo2200 +0.026) and leaves dedup64 (+0.0085) and everything smaller unproven. All conclusions are val-loss-on-elo1600-reference; checkpoints are saved (`results3/slice67-*.pt`), so two cheap diagnostics remain open: re-evaluate the ladder arms on an elo2200-band val shard to confirm the distribution-shift story directly, and gate decisive vs control at beam search. The decisive+noforfeit composition is untested.

## Experiment 67b — round-robin strength ranking of the slicing arms (2026-08-07)

All 36 pairs of the nine slice67 checkpoints, beam-4 both sides (depth 4, beam 6, root 8, contempt 0.15), color-reversed seeded openings, run as parallel t4-small cloud jobs (~$15; 20-game jobs hit the 30m timeout with the upload unreached, so pairs were re-fanned as two 10-game blocks — plus 27 straggler jobs from the first wave that finished under the wire and count as independent openings). **1,260 games, 93% decisive** — beam search fully bypasses the draw wall. Joint MLE Elo, mean-anchored at 0:

| arm | Elo | points | (val-loss rank) |
|---|---|---|---|
| nobullet | +68.9 | 172/280 | 7th |
| noforfeit | +44.9 | 149.5/260 | 4th |
| control | +24.9 | 153/280 | 2nd |
| dedup64 | +20.9 | 152.5/280 | 5th |
| elo2200 | +20.9 | 148/280 | 9th |
| unfiltered | +12.9 | 144/280 | 8th |
| elo2000 | −47.1 | 128.5/300 | 6th |
| elo1800 | −55.1 | 114/280 | 3rd |
| decisive | −91.1 | 98.5/280 | 1st |

**The match ranking near-inverts the val-loss ranking.** The screen's best imitation arm (decisive) is last at play by 116 Elo below control; the second-worst imitation arm (nobullet) is first. This is the lab's Fleet-2/3 lesson in its sharpest form yet: val loss on a fixed reference measures imitation of a population, and under a search decode the value head and move quality govern strength instead.

Mechanisms, as the evidence supports them: **decisive**'s draw-blind value head (the only degraded judge in the screen, 58.17) is consulted at every beam backup, and a judge that cannot represent draws misprices exactly the positions contempt exists for — the val-loss "win" was an artifact of the ruler. **nobullet** had the best judge of the nine (58.54, a whisper in the saturated band that match play amplified) and its games carry real thinking time; its poor val loss was mispricing bullet-styled moves it was never trained to imitate. **noforfeit** — the learner's mechanism, an exact null on value_top1 — places second at play: clean outcome labels appear to matter to the judge's decision quality in ways aggregate value accuracy does not measure (echoing Exp 51's judge lesson). The Elo ladder is non-monotone at play: 1800/2000 clearly below control, 2200 ≈ control.

**Statistics, honestly:** ~280 games per model gives roughly ±40 Elo at 95% per pairwise-ish comparison. Solid: decisive, elo1800, elo2000 are below control. Suggestive but not established: nobullet (+44 over control ≈ 2 SE) and noforfeit (+20) above control. Adoption call: **no data-policy change to the champion lineage yet** — nobullet and noforfeit earn a properly powered gate (and their composition is untested); decisive is rejected for play despite winning the screen; the Elo floor stays at 1600.

## Experiment 65b — seed replication, and a retraction (2026-08-07)

**Why it ran:** Experiments 65 and 66 saved no checkpoints (`save_ckpt` defaults false; only tie66 had it set), so the arms could not be matched head-to-head. Rather than repeat them identically, the equal-FLOPs family was re-run at **seed 20260807** instead of 20260730 — same recipes, same a-priori step counts, same 66.1 PFLOPs — which buys checkpoints *and* the seed replication that had been the standing caveat.

| arm | seed 20260730 loss (Δ) | seed 20260807 loss (Δ) | 2-seed mean Δ |
|---|---|---|---|
| none (control) | 1.4660 | 1.4618 | — |
| K=V | 1.4574 (**−0.0085**) | 1.4639 (**+0.0021**) | −0.0032 |
| Q=K | 1.4668 (+0.0008) | 1.4658 (+0.0040) | +0.0024 |
| Q=V | 1.4670 (+0.0010) | 1.4677 (+0.0059) | +0.0035 |
| Q=K=V | 1.4672 (+0.0012) | 1.4714 (+0.0096) | +0.0054 |

**Retraction.** Experiment 66 recorded "K=V wins again … 4 of 4 cells agree on the sign." That claim was wrong in its central inference. All four of those cells shared seed 20260730 — Exp 65's K=V and Exp 66's K=V are different runs at different scales, but they draw from the same initialization stream, so they were never four independent confirmations. They were one seed, observed four times. At a second seed the sign flips: K=V is 0.0021 *worse* than its control. The correct summary is **K=V ≈ control, inside seed noise** — not a win.

**What does survive, and it is the more useful half.** The damage ordering is monotone and identical at both seeds: **K=V < Q=K < Q=V < Q=K=V**, and Q=K=V is worst in both. That matches the per-step analysis in Exp 65 and the mechanism: what the output projection can absorb is cheap (K=V re-parameterizes the value path as W_out·W_K, and W_out is free), what it cannot is expensive (Q=K forces a symmetric score matrix). The ordering is the durable finding; the absolute offsets ride on the seed.

**Methodological lesson, now the binding constraint on this lab.** The control alone moved 0.0042 loss / 0.11 top-1 between seeds — the same size as the effects being chased. Every single-seed comparison in the 0.002–0.010 loss band in this log, including cap64-vs-capstone-59 (+0.44 top-1 over a single pair of runs), is under-powered by this measure. Effects of that size need ≥3 seeds per arm, or a paired design, before they are called. Cheap and worth doing: seeds are ~$3 and 50 minutes each here.

**Adoption call, revised:** do **not** adopt K=V for the per-FLOP win, because there isn't one. Adopt it, if at all, for the parameter cut — 14% fewer weights and ~25% of a wall-clock saving at statistically indistinguishable loss, which is a package-size and iteration-speed argument under a 100MB cap, not an accuracy argument. Q=K=V stays rejected on both seeds.

## Lab practice — what the QKV week taught the instrument (2026-08-07)

Consolidated from Experiments 65 / 65b / 66. These are lessons about the *measuring apparatus*, not about attention; they bind on everything measured here from now on.

**1. The noise floor is 0.0042 loss / 0.11 top-1, and we had never measured it.** Two runs identical but for the seed differ by that much. Nearly every architecture verdict in this log sits in that band — bilinear's +0.44 top-1 at 129M, cap64's +0.44 over capstone-59, K=V's −0.0085. None of them were wrong on purpose; they were single-sample estimates of quantities smaller than the sampling noise. **Standing rule: nothing in the 0.002–0.010 loss band gets called from one run per arm.** Three seeds, or a paired design, or don't claim it. A seed costs ~$3 and 50 minutes here, which is far cheaper than a wrong architectural commitment carried into a full-budget run.

**2. Replications that share a seed are one observation, not many.** Exp 65 and Exp 66 agreed that K=V wins — across two budgets, two scales, four cells. All four drew from initialization stream 20260730, and the effect evaporated at 20260807. Different runs are not independent evidence if they share the thing driving the variance. The diagnostic question is not "did it reproduce?" but "what did the reproductions hold fixed?"

**3. Rank order survives noise; differences do not.** The tie damage ordering (K=V < Q=K < Q=V < Q=K=V) was identical at both seeds while every absolute margin moved. Prefer experiment designs that ask "which of these is least bad" over "is this better than baseline" when the expected effect is near the noise floor.

**4. Analytic FLOPs overstate wall-clock savings by ~4x at this model size.** Only 22–25% of a projected FLOP reduction converted to throughput; Q=K=V uses 0.70x the FLOPs and runs 1.105x faster. What doesn't shrink when projections shrink — attention scores, norms, embeddings, optimizer, memory traffic — dominates. Never schedule from a FLOP-derived time estimate without the discount.

**5. We are not FLOP-limited, so per-step efficiency is the wrong lever.** cap64 spent 1.985e17 of the 1e18 cap — **80% of the budget is unspent.** K=V's 17.6% step bonus at the cap is worth ~0.010 loss; spending the remaining budget is worth ~0.099. Architecture micro-optimization has been competing for attention against a lever 10x larger that requires no cleverness at all. The full-budget lineage run should outrank further ablations.

**6. Match gates cannot resolve what we have been asking them.** 40 games carries a standard error of ±3.2 games; the arms being compared differ by ~0.005 validation loss. Gates are for "is this model behaviorally broken" and for final champion selection, not for ranking micro-architecture variants. Budget them accordingly — the 5-match batch that closed this week cost ~12 core-hours and could not, even in principle, have separated the arms.

**Three tooling defects found the hard way, all now fixed:**
- `save_ckpt` defaults false, so Experiment 65's ten runs left no models and could not be matched at all. Any run that might need behavioral evaluation must set it; 15MB is nothing.
- Concurrent beam matches on one Metal device do not share it — they serialize, and five processes each fell to ~1% of a core (a ~30x per-process slowdown, not the 5x fair-share). `lab.match` gained `--device` so batches can run on CPU, which is 2x slower solo but scales across cores. Match throughput scales with *processes*, not threads: the beam search is bound by single-threaded python-chess move generation, so extra torch threads idle.
- `lab.match` printed its tally only after the final game, so an interrupted series yielded nothing rather than partial results. It gained `--progress`, which emits the running tally after every game.

## Experiment 3 — value-head leakage (parked, 2026-07-30)

Add a win/draw/loss value head; measure outcome-accuracy under both splits. Teacher's sealed prediction: position split inflates outcome accuracy by +8 to +20 points, 75%. Lab prepare now emits a `result` column, so this runs whenever unparked. Parked at learner's request to prioritize the end-to-end pipeline.

## Hardware trial for the full-budget run (2026-08-07)

Eight ~5-min steady-state jobs (~$2.50): {a100-large, h200, rtx-pro-6000, l40sx1} × {default compile, reduce-overhead}, all at the final run config (d384·12L, K=V tied, 9,708,819 params, batch 1024, bf16, slice67-nobullet shard, fwd 1,403,904,256 FLOPs/pos by the profiler on the instantiated model).

| flavor | steps/s | $/exaFLOP | wall/exaFLOP | training sane? |
|---|---|---|---|---|
| rtx-pro-6000 | 20.14 | **$8.50** | 3.1 h | yes — best val loss, on the steps→loss trend |
| h200 | 28.99 | $10.80 | 2.2 h | **no** — val 3.33 at 4,574 steps vs rtx 1.94 at 3,188 |
| a100-large | 13.79 | $11.30 | 4.5 h | yes |
| l40sx1 | 9.99 | $11.30 | 6.3 h | yes |

Two catches, both by the smoke-with-metrics protocol rather than by crashes: (1) `torch.compile(mode="reduce-overhead")` trained at full speed with no errors while producing garbage models on h200 (val 10.66) and rtx-pro-6000 (val 27.23), and gained ≤1.6% on the flavors where it worked — rejected permanently. (2) h200 is silently degraded even at default compile — more steps, much worse loss, only flavor off the trend — disqualified on quality despite being fastest. The Exp 61 meta-lesson (every speed flag and every new silicon is itself an experiment; smoke with metrics, not just for NaNs) has now paid for itself twice.

**Decision: rtx-pro-6000, default compile.** Budget run fixed at 224,929 steps = 230.3M positions = 0.97e18 profiler-FLOPs, ~3.3 h wall, ≈ $9. MFU note: 87 TFLOP/s achieved at 9.7M params (vs 60 on a100) — the 2.6× scale-up roughly doubled hardware utilization over cap64-size runs, as predicted.

## Run 68 — the full-budget model (2026-08-08)

Fresh lineage per lab/PROPOSAL_FULL_BUDGET.md: transformer 12L·d384·8h·FFN1×, K=V tied, attention bias, bilinear head, flip, 9,708,819 params; Jan+Feb+Mar both>1600+nobullet game-record corpus (293.5M positions available), single pass, 224,929 steps × batch 1024 = 230,327,296 positions; AdamW lr 1.2e-3, cosine + 5% warmup, dropout 0.1, value CE weight 1.0; seed 20260730; rtx-pro-6000, default compile. **Profiler FLOPs 9.70072e17 (97.0% of cap), lineage fresh, zero non-finite skips, 3.4 h, ≈$9.40.**

**Final (full April-nobullet ruler, 2.19M positions): val_loss 1.3169, top-1 56.15%, value_top1 60.47%** — the first judge past the 57.8–59.0 saturation band that has held since Fleet 3. Mid-run probe deltas flattened near 56% (1.3845) then the cosine tail recovered −0.063 more; the saturation-vs-schedule ambiguity resolved in favor of the schedule.

Data pipeline for this run was rebuilt mid-flight at the learner's direction: Rust extractor (pgn-reader/shakmaty) → u16-move game-record parquets (~150MB/month, 38s/month vs 91min for materialized prep) → vectorized numpy decode on-job (6.2M pos/s). Verification: selection parity exact vs the old pipeline (Jan/Feb), 6.4M sampled rows + full-shard aggregates bit-identical, cross-implementation April byte-equality. Two preflight smokes on the launch hardware gated the run. Checkpoints every 2% with full resume state; none were needed.

Checkpoint: results3/fullbudget-68.pt (+ 49 partial marks). Next: continuity-ruler eval, package, publish, arena round per the amended plan.

## Run 68 epilogue — eval, package, publish (2026-08-08)

Continuity ruler (slice67-val-april, 1.79M positions): **1.4060 / 54.37% / value 59.12%** vs screen control 1.4829 / 52.10 / 58.47 — best model on the frozen ruler by clear margins. Flag: the nobullet→all-TC distribution penalty is +0.089 loss at this scale vs +0.012 at slice67 scale — specialization deepens with compute. ONNX parity: argmax 100% policy and value, residual at GEMM-accumulation scale (3.8e-4 max logit), K=V export verified; same under ort-web 1.27 single-thread. Arena is single-threaded again (Sites cannot serve COEP on worker assets — see TOURNAMENT_RULES 2026-08-08 note); adapter retuned for ~31.5 ms/eval: root 4 / beam 3 / depth 4 / quiescence 2, budget fraction 0.85 — 198–254 evals, 6.3–8.2 s/move at a 10s clock. Package 39,059,685 bytes, validator + all 6 probes green. **Published: `peterwooden/chess-gpt-fullbudget-68` @ `0e3f0b824ef8b4cb8c6ef39591d86b833a7ea6eb`** with profiler runrecord (9.70072e17, lineage 0). Next: informal parallel arena comparisons vs each competitor's latest 1–2 entries, 4 games each at 10s/move.
