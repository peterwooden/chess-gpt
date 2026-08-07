# The full-budget tournament run — settled decision record (2026-08-07)

Grilled to alignment 2026-08-07; every item below is agreed. Decisions marked **[override]** are the learner's explicit calls against the teacher's recommendation, recorded as such at decision time. The only remaining gate before launch is the WASM speed validation of the model size (in flight).

## Settled decisions

1. **Lineage:** fresh init, single stage, single cosine cycle, single pass over fresh data. No warm start (Exp 55/56), no elite stage (Exp 59/67), no mid-run intervention. Seed 20260730. One run; **no second seed**.
2. **Spend: 0.97e18** profiler-FLOPs, step count fixed a priori from the analytic FLOP model on the instantiated architecture (the Exp 65/66 method). Checkpoints (full resume state, ~120 MB) every **2%** with per-mark filenames and upload retry; 3% margin absorbs one crash window plus accounting drift. Crash-lost work is counted against lineage honestly.
3. **Architecture: transformer d384 · 12L · 8h · FFN1×, K=V tied** (learner's call; 65b supports as a param/wall win at indistinguishable loss), learned attention bias, bilinear from·to head, perspective flip, history_k 8, per cap64's recorded config in all remaining knobs. ~9.7M params (~39 MB fp32 — package cap irrelevant). ≈2.6× champion scale keeps the d/L aspect ratio; chosen by scaling judgment after the scale pilot was cancelled. **Pending:** local ONNX-web speed bench (random weights, free) must confirm beam-4 fits the ~10s move clock at this size before launch.
4. **Data: Jan+Feb+Mar 2026, both>1600 AND base time ≥180s (nobullet), draws kept, both sides' moves, flip-canonicalized, game-level shuffle across months.** Single fresh epoch, no reuse. nobullet is an **[override]**: adopted on Exp 67b's +44 Elo point estimate and mechanism (real thinking time, best judge of the screen) against the recommendation to treat a top-of-9 result at ~2 SE as unadopted until powered. Pool ≈ 75M games; need ≈ 3.6M; ~270M positions ≈ 26 GB device-resident (no streaming loader needed).
5. **Optimization:** AdamW (0.9, 0.999) wd 0.01 eps 1e-8 fused, **lr 1.2e-3**, **batch 1024**, cosine→0 with **5% warmup** (cap64's recorded values — the earlier draft's 2% was wrong), **dropout 0.1**, value_weight 1.0 CE with **ply-weighting off** (also per the recorded recipe), bf16 autocast, torch.compile, no grad clip, guarded skip-step on non-finite norm, NaN tripwire that halts. LR kept at champion value deliberately: 2× under the known cliff, protected by warmup + skip-step + 2% checkpoint recovery (resume at reduced LR if a late divergence ever fires).
6. **Hardware: trialled 2026-08-07 (8 jobs, ~$2.50) — winner rtx-pro-6000, default compile.** Measured at the final config (d384·12L tied, 9,708,819 params, fwd 1,403,904,256 FLOPs/pos): rtx-pro-6000 20.14 steps/s ($8.50 and 3.1 h per 0.97e18 — cheapest and clean), h200 28.99 steps/s but **disqualified on quality** (val loss 3.33 at 4,574 steps where rtx hit 1.94 at 3,188 — silent Hopper-path degradation at default compile), a100 13.79, l40s 9.99. `reduce-overhead` compile **silently poisoned training on h200 and rtx-pro-6000** (val loss 10.7 / 27.2 at full speed, no errors) while gaining ≤1.6% where it worked — rejected permanently; the smoke-with-metrics protocol caught it, vindicating the Exp 61 meta-lesson twice over. Budget run: **224,929 steps** (230.3M positions, 0.97e18 exactly) on rtx-pro-6000, ~3.3 h wall ≈ $9. These trials plus the budget run are the **only** cloud compute in this phase — all pilots (scale, dropout, nobullet gate, second seed) are cancelled.
7. **Monitoring:** two frozen April rulers logged at every mark — `slice67-val-april` (both>1600, continuity with all prior numbers) and a new nobullet-filtered April shard (distribution-matched). The continuity ruler is *expected* to read ~+0.012 worse for distribution reasons (Exp 67); that is not a regression signal.
8. **Nomination: val curves + probe suite before upload — no pre-publication gate games.** **[override, confirmed twice]** The teacher's recommendation was a 200-game local CPU gate vs cap64 (the 55/f2/67b inversion record, plus breakage detection); the learner deliberately accepts the inversion risk and publishes on curves, the package validator, and tactical probes. **Amended 2026-08-07: after upload, the model IS play-tested on the arena site** (chess-gpt-lab.peter-r-wooden.chatgpt.site) — 4 games per registered model at the 10s limit (~24 opponents registered at time of writing, incl. burrowdweller/minichess-gpt-v1-final at 115-1-5). cap64/capstone-59 remain registered as fallbacks; nomination can be revisited any time before registration closes since entries are pinned only at freeze.
9. **Packaging:** fp32 ONNX export with bit-parity check, existing flip-aware beam adapter, publish as a new pinned HF repo, run record with profiler version / actual examples / run FLOPs / lineage 0 / seed / config hash per TOURNAMENT_RULES.

## Cost and sequence

| step | what | cost | wall |
|---|---|---|---|
| 0 | WASM speed bench of d384·12L (local, running) | $0 | hours |
| 1 | prep Jan+Feb+Mar nobullet shards + April nobullet ruler (local) | $0 | ~1 day |
| 2 | hardware trials × 4 flavors | ~$3 | half a day |
| 3 | **budget run** 0.97e18 on the winner | ~$11–25 | 2–6 h |
| 4 | curves + probes, package, publish, register | $0 | ~half a day |

Total ≈ **$15–30**.

## Not doing (unchanged from the draft, plus the grill's cancellations)

Warm start (55/56) · elite stage (59/67) · decisive filter (67b) · untied QKV (superseded by the K=V call) · CNN (62/63) · search-labeled value (58) · self-play (rules + draw-wall) · curriculum (Fleet 2) · batch 4096 / multi-GPU (Fleet 2 divergence, DDP complexity) · grad clip 1.0, label smoothing, Muon, schedule-free (Fleet 2/3) · scale pilot, dropout-0 arm, nobullet powered gate, second seed, gate matches (cancelled/overridden in grilling, 2026-08-07).
