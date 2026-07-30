# Lab

A fresh, minimal rebuild of the core pipeline — data → tensors → model → loss → training → evaluation — designed for **minutes-long experiment loops** on the M4 Mac Mini. Everything here is small enough to read in full and fast enough to be wrong cheaply.

Ground rules:

- Every experiment finishes in under ~5 minutes end to end.
- Every model is measured with the same frozen ruler (see DECISIONS.md once decided).
- Every consequential choice is made deliberately and logged in [DECISIONS.md](DECISIONS.md) with the options considered and the reasoning.
- A run gets a prediction before it gets a result.
- Bench conclusions are hypotheses: the winners get re-verified once at medium scale before tournament bets.

The tournament-grade pipeline in `src/chess_gpt/` is unchanged and serves as reference material and the match arena.
