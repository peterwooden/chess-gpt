# Chapter 1 adaptive plan — What does it mean to learn?

This is the direct path selected after diagnostic `CGPT-D0-7-7F-H0` and the learner's correct follow-up explanation of test-set selection bias. It is deliberately narrow: each mission adds one causal lever to the existing SAN n-gram baseline, and mastery is recorded only after a prediction, evidence, and an explanation from memory.

## Mission sequence

- [ ] **M1 — Split games, not positions (10–15 minutes).** Predict the direction of leakage bias, compare three split implementations, and identify why the stable game-ID hash is the trustworthy choice. Completion evidence: gated site code plus an explanation of why apparent validation accuracy changes.
- [ ] **M2 — What exactly is one prediction? (10–15 minutes).** Map one chess game into contexts and next-move targets, then trace one n-gram count from the pinned data to a SAN prediction. Completion evidence: label context, target, and learned state in a real baseline example.
- [ ] **M3 — What does the baseline believe? (10–15 minutes).** Turn counts into conditional probabilities, compare exact-context, backoff, and deterministic-fallback implementations, and predict where each will fail. Completion evidence: calculate one tiny probability and diagnose one baseline miss.
- [ ] **M4 — How much data should validation receive? (10–15 minutes).** Forecast the uncertainty/training-data trade-off at several split sizes, then specify one controlled experiment before it runs. Completion evidence: a direction, rough magnitude, assumptions, confidence, and likely failure mode.
- [ ] **Source assignment (20–40 minutes).** Inspect the pinned dataset's `site` and `tokens` fields and the baseline's `is_validation_game` function; answer: “What identity must remain stable for our current split to remain reproducible?”
- [ ] **Chapter experiment (45–75 minutes).** Compare two validation percentages with fixed dataset revision, game count, seed, model, and evaluation semantics. Record the prediction before execution and interpret both accuracy and estimate uncertainty.
- [ ] **Mastery checkpoint.** Explain example, context, target, parameter, baseline, leakage, and the training/validation/test contract without the lesson open.

## Mission 1 sources

- **Project evidence:** [`src/chess_gpt/baseline.py`](../src/chess_gpt/baseline.py) for `GameRecord` and `is_validation_game`; [`data/dataset-candidate.toml`](../data/dataset-candidate.toml) for the candidate revision.
- **Primary source:** [Kohavi, “A Study of Cross-Validation and Bootstrap”](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf), especially the motivation and experimental comparison.
- **Secondary source:** [Goodfellow, Bengio & Courville, *Deep Learning*, §§5.1–5.3](https://www.deeplearningbook.org/contents/ml.html), especially generalization and data partitions.

## Teaching constraint

Do not mark a mission complete merely because its page was opened. The learner must return its code and explain the central causal prediction in their own words; wrong answers can be retried without penalty.
