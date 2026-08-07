# Proposal — the full-budget tournament run (draft for sign-off, 2026-08-07)

One document, every decision, each tied to the experiment that earned it. Items marked **[OPEN]** are the consequential calls that need learner sign-off; everything else is proposed as settled unless challenged.

## 0. Budget arithmetic

- Cap: 1e18 profiler-FLOPs per **submitted checkpoint's lineage** (profiler v1: 3× forward matmul FLOPs × actual examples, plus all parent checkpoints).
- Exploration is therefore free in FLOPs: pilots, seed replicates, and abandoned candidates cost only dollars. Only the nominated model's own lineage counts.
- cap64/capstone-59 spent 1.985e17 (19.8%). A fresh-init run gets the whole 1e18 — warm-starting the champion would import its 20% debt for a technique (warm continuation) that failed both times it was tried (Exp 55/56).
- Measured FLOP constant: forward ≈ **138 FLOPs per parameter per position** (cap64: 512.59M fwd / 3.713M params); training ≈ 414 × params × positions.
- **Spend target: 0.97e18** (margin for crash-lost partial steps, which conservatively count against lineage, and for accounting drift). 3% margin ≈ 40 A100-minutes of slack. **[OPEN — or push to 0.99 with confident bookkeeping]**

| params | positions @0.97e18 | games (both>1600, ~72.8 pos/g) | steps @batch 1024 |
|---|---|---|---|
| 3.7M (cap64) | 631M | 8.7M | 616k |
| ~8M (d384·8L) | 293M | 4.0M | 286k |
| ~18M (d512·10L) | 130M | 1.8M | 127k |
| ~34M (d640·12L) | 69M | 0.95M | 67k |

Data pool is never binding: January alone has ~50M both>1600 games (54.8% acceptance); Jan+Feb+Mar ≈ 150M. Even nobullet (−44%) leaves ≥ 9× the largest need.

## 1. Lineage plan — one fresh, single-stage run

- Fresh init, single cosine cycle, single pass over fresh data, land at 0.97e18 exactly via a-priori step count from the analytic FLOP model (the Exp 65/66 method; it landed 1.0000e17 on the nose).
- **No warm start** (Exp 55/56: continuation at full LR improves imitation, degrades strength; even the corrected recipe is an untested risk with no budget upside for a fresh lineage).
- **No stage-2 elite fine-tune** (Exp 59: stage 2 gated even with stage 1; Exp 67: every elitism step above 1600 hurts; Exp 67b: elo1800/2000 clearly worse at play).
- **No mid-run interventions.** Single-seed necessarily; every delta must be pre-committed.
- If credits allow: a **second full-budget run at a different seed** is separately legal (own lineage, unlimited registration, nominate the gate winner). The noise floor is 0.0042 loss between seeds (Exp 65b); ~$20–45 is cheap insurance against seed luck on the one artifact that matters. **[OPEN — depends on credit balance]**

## 2. Architecture — cap64's recipe, size TBD by pilot

Frozen (all champion-proven, none inside the noise band):
- Transformer encoder, 65-token board, perspective flip canonicalization (Exp 59, oracle-verified).
- Learned attention bias; FFN 1×; 8 heads (capstone-59).
- **Bilinear from·to policy head** (Exp 63: +5 top-1 at small scale, still ahead at 129M).
- **Untied Q/K/V** (Exp 65b: K=V's per-FLOP win retracted; ties buy only parameter count, which we don't need under 100MB).
- Value head, weight 1.0, CE with ply-based sample weighting (Exp 51: the judge decides under search; Exp 67b confirmed brutally).
- Exact remaining knobs (history features, state token, etc.): **identical to cap64's recorded run config** — the run record is authoritative; no undocumented deltas.

**[OPEN — model scale, the biggest unknown.]** No evidence exists above 3.7M at the modern recipe. Chinchilla-style reasoning at 1e18 suggests tens of millions of params; chess-specific counterweights: (a) per-move wall clock — a bigger model evaluates fewer beam nodes, and depth compounds harder than any training lever measured (Exp 52: 4-ply vs 1-ply ≈ +430 Elo); (b) fp16 export caps params at ~48M under the 100MB package limit.

**Scale pilot (off-lineage, ~$15–25):** arms at 1e17 FLOPs each, seed 20260730, frozen April-elo1600 ruler (`shards/slice67-val-april.parquet`):
- S0 = 3.7M — already measured, free (Exp 66 control: 1.4408).
- S1 ≈ 8M (d384·8L), S2 ≈ 18M (d512·10L); S3 ≈ 34M (d640·12L) only if S2 beats S1.
- Per arm, also record: measured pos/s on the chosen GPU, and **ONNX-web ms/eval on the M4** (the search-throughput axis).
- Decision rule: pick N* maximizing predicted match strength = loss-vs-N slope combined with beam-nodes-vs-N under the tournament move clock — not val loss alone (Exp 67b: the ruler inverts under search). If the loss curve is flat-ish across sizes, prefer **smaller** (search throughput wins ties).
- Optional 4th arm: chosen size with **dropout 0** — single-pass fresh data makes regularization mostly cost (gaps already ≤0.9 at 6-epoch reuse); adopt if ≥ noise-floor better. Otherwise keep 0.1 (champion default).

Note scale effects (expected ≥0.02–0.05 loss between S-arms) clear the 0.0042 single-seed noise floor, unlike the micro-architecture band — one seed per arm is defensible here.

## 3. Data

- **Pool:** Jan + Feb + Mar 2026 (all downloaded, 109GB), `both>1600`, draws kept, both sides' positions, flip-canonicalized. Elo floor stays 1600 (Exp 67: monotone worse above; 67b: same at play). **Decisive filter rejected** (67b: −116 Elo, draw-blind judge).
- **Fresh single epoch, no reuse** (Fleet 3: 960k-once ≈ 480k-twice, so reuse is nearly free — but fresh is at least as good and the pool is bottomless). Game-level shuffle across months so shard order carries no chronology drift (Exp 67 caveat).
- **Validation:** April only, never trained on. Keep `slice67-val-april` (1.79M pos) as the continuity ruler; optionally cut a larger April shard for the final low-noise read.
- **[OPEN — nobullet.]** Exp 67b: nobullet +69 Elo over control (~2 SE, suggestive; best judge of the nine; mechanism plausible — real thinking time) and noforfeit +45. Neither is established; composition untested. Proposal: **powered confirmation before adoption** — 1,000 games nobullet-vs-control using the existing slice67 checkpoints (parallel CPU/t4 jobs, ~$10–15; SE ≈ ±1.6%, resolves ~±11 Elo). Adopt nobullet iff it clears; adopt **at most one** filter (composition trap: Fleet 2/3 finalists both failed gates on stacked wins). Noforfeit stays unadopted unless nobullet fails and noforfeit clears its own gate. Default on ambiguity: control (both>1600, no further filter).
- Prep engineering: if N* ≥ ~18M, the existing 2M-game elo1600 shard already covers the need (130M positions) unless nobullet passes; otherwise prep fresh shards (known pipeline, awk pre-skim for speed).

## 4. Optimization

All champion defaults; nothing in the noise band gets changed for the big run:
- AdamW (0.9, 0.999), wd 0.01, eps 1e-8, **fused**, lr 1.2e-3, **batch 1024** (b4096 diverged in Fleet 2 — and no batch scaling to enable multi-GPU).
- Cosine to 0, warmup 2% (Fleet 2's sleeping giant; 2% vs 5% measured small-real — keep cap64's recorded value).
- bf16 autocast, torch.compile on CUDA (+10%, Fleet 2), guarded skip-step on non-finite grad norm (Exp 61 standard), no grad clipping (Fleet 2: clip 1.0 cost 3.3 points).
- No label smoothing, no EMA/SWA, no Muon (value-head interaction; single-seed-band win only), no schedule-free (small-real, untested at this recipe).

## 5. Run mechanics (the Exp 61/65/67b lessons, made binding)

- `save_ckpt` true; **per-mark checkpoints every 5%** with distinct filenames, uploaded with retry; resumable optimizer+scheduler state.
- Self-enforced watchdog (HF's job timeout is not enforced — Exp 61); NaN tripwire that halts rather than trains through poison.
- Streaming shard loader: at 3.7M-param scale the run needs ~630M positions ≈ 60–100GB — load shard k+1 while training k; at N* ≥ 18M everything fits device-resident as today.
- Run record with profiler version, actual examples, run FLOPs, lineage FLOPs (0), seed, config hash — the compliance artifact per TOURNAMENT_RULES.
- **Stability smoke first**: 10 minutes at full config on the target hardware before the budget clock starts (Exp 61 meta-lesson: every speed flag is itself an experiment).

## 6. Hardware — probe, then commit (~$2 of probes)

Measured baseline: A100 does 20.7k pos/s at 3.7M params ≈ 32 TFLOP/s achieved ≈ **10% MFU** — the model is too small to saturate it; we're memory/launch-bound, so paper FLOPs mislead (QKV week lesson 4: only ~25% of FLOP savings converted; the same physics caps FLOP-rich hardware here).

| flavor | $/h | key spec | verdict |
|---|---|---|---|
| a100-large | 2.50 | 312 TF bf16, 2.0 TB/s | baseline: 8.5h / ~$21 per exaFLOP at 3.7M-param size |
| **h200** | 5.00 | 989 TF, **4.8 TB/s**, 141GB | best wall-clock bet: bandwidth-bound workload should gain ~2–2.5×, ≈ cost-neutral per FLOP, VRAM swallows any shard |
| **rtx-pro-6000** | 2.75 | Blackwell, 96GB GDDR7 1.8 TB/s | dark horse on $/exaFLOP: A100-class bandwidth at 1.1× price, newer arch, likely lower launch overhead |
| l40sx1 | 1.80 | 0.86 TB/s | dominated: 0.43× bandwidth for 0.72× price |
| l4x1 / a10g / t4 | ≤1.50 | ≤0.6 TB/s | too slow |
| a100x4/x8, h200x2+ | 10–40 | multi-GPU | **rejected**: needs DDP + global batch ≥4096 (diverged) or gradient accumulation complexity, for a run that is only hours long single-GPU |

Plan: 10-minute timing probes on **a100-large, h200, rtx-pro-6000** at N* (and at 3.7M if N*=3.7M), each with compile default vs `mode="reduce-overhead"` (CUDA graphs — the specific cure for small-model launch overhead; smoke for the Exp 61 compile-poisoning failure mode with the NaN tripwire armed). Commit to the best measured **pos/s per dollar**, tie-broken by wall-clock. Expected outcome: H200 finishes the budget run in ~3.5–5h (~$18–25) at 3.7M size, faster still at larger N* where MFU rises.

## 7. Evaluation, selection, packaging

- During run: val loss + top-1 + value_top1 on the frozen April ruler at every checkpoint mark.
- After: (1) behavioral probes + package validator; (2) **gate vs cap64, 200 games beam-4 both sides** (resolves ~±50 Elo; a full-budget model should clear decisively — if it lands inside noise something is wrong and we debug before nominating; extend games only for a genuinely close call between our own candidates, per QKV lesson 6 gates rank champions, not variants); (3) ladder rating for the record; (4) ONNX fp16 export with bit-parity check, package ≤100MB (trivial below ~48M params).
- Registration: publish as a new HF repo, pin the commit; nominate the gate winner. cap64/capstone-59 remain registered fallbacks — a failed big run costs nothing but dollars.
- Out of scope here but flagged: the inference layer (search depth, time management under `moveTimeLimitMs`) has produced the largest strength jumps all project (+650 fleet-1, +430 Exp 52) and deserves its own proposal after this run ships.

## 8. Explicitly not doing, and why

| idea | killed by |
|---|---|
| warm-start / continuation from cap64 | Exp 55/56 (both failed gates), 20% lineage debt |
| elite fine-tune stage | Exp 59 (even), Exp 67 (elitism hurts) |
| decisive filter | Exp 67b (−116 Elo, draw-blind judge) |
| K=V tying | Exp 65b retraction (seed noise; only a param cut we don't need) |
| CNN trunk | Exp 62 (10× slower per dollar on silicon), Exp 63 (0.5 short at parity) |
| search-labeled value training | Exp 58 (cheap variant null); deep variant unproven — off-lineage v2 experiment at best, never a first-shot passenger |
| self-play data | rules unresolved; draw-wall starves it (fleet 1) |
| curriculum / short-first | Fleet 2 (destroyed value head) |
| batch 4096 / multi-GPU DDP | Fleet 2 divergence; complexity for an hours-long run |
| grad clip 1.0, label smoothing, Muon, schedule-free | Fleet 2/3: harmful or inside the noise band |

## 9. Cost and sequence

| phase | what | cost | wall |
|---|---|---|---|
| 0 | hardware probes (3 flavors × 10 min, compile modes) | ~$2 | half a day, parallel |
| 0 | nobullet powered gate (1,000 games, existing ckpts) | ~$10–15 | overnight, parallel |
| 1 | scale pilot S1–S2 (+S3, +dropout arm) at 1e17 each | ~$15–25 | ~1 day |
| 2 | data prep if needed (nobullet or >2M games) + streaming loader | ~$0 (local/cpu) | ~1 day |
| 3 | **the budget run** at 0.97e18 | ~$20–45 | 4–9h |
| 3′ | optional second seed | ~$20–45 | 4–9h |
| 4 | gates, ladder, package, publish, register | ~$5–10 | ~1 day |

Total ≈ **$50–90** (≈ $75–135 with the second seed). **[OPEN — confirm the current credit balance supports this; the log last recorded credits exhausted at ~$100, then further spend, so the real balance is unknown to the lab.]**

## Sign-off checklist (the [OPEN] items)

1. Spend target 0.97e18 vs 0.99e18.
2. Model scale decision rule (and whether S3/34M gets a slot).
3. Dropout-0 pilot arm: run it?
4. nobullet: adopt on a 1,000-game win, or freeze data policy at control regardless?
5. Second full-budget seed: yes/no given credits.
6. Credit balance / dollar ceiling for the whole phase.
