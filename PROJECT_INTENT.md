# Project intent

Recorded on 2026-07-24. This file preserves the founding brief. It is the durable source for why this repository exists; [MISSION.md](MISSION.md) is its shorter working compass.

## Founding brief (verbatim)

> My friend trained a small chess llm and hosted it on https://shahazmat.github.io/chess-transformer-model/. It uses a novel technique which trains blunders/innaccuracy signals as tokens and downsamples them and the following token. I suggested that three of us friends all try to train chess models and they play off against eachother. To keep it fair we will limit the model size and use the same training data. This is the training dataset https://huggingface.co/datasets/shazmate/lichess-chess-tokens . I anticipate that I will try all sorts of different experiments. I'm not an ML engineer but I want to use this as an opportunity to learn ML fundamentals, including LLMs. I want you to set up this repo from scratch and teach me how to do it properly. Follow 2026 best practices, don't take shortcuts. I would expect we do things like versioned experiments, training and test sets, etc. I really like how Karpathy teaches things and he probably has a good approach to teaching ML and/or LLMs - breaking down the fundamentals into incremental intuitions. Look up online his teaching materials (perhaps eureka labs?) and anything he recommends - and use that to guide me. So overall, I want to win this tournament, second, I want to learn to be an ML engineer, third, I want to express my creativity thru different ideas. First thing, document my intentions in durable context in this repo (verbatim as possible), second, what do you need from me? Treat me as your student.

## Priority order

1. Win the tournament.
2. Learn to be an ML engineer, including LLM fundamentals.
3. Express creativity through original experiments.

These priorities are complementary: a strong baseline makes creative experiments measurable, and understanding the system makes good competitive ideas more likely.

## Working commitments

- Freeze the tournament rules, dataset revision, tokenizer, and evaluation protocol before comparing models.
- Establish a deliberately simple end-to-end baseline before adding novel techniques.
- Treat an experiment as a reproducible record: hypothesis, one intentional change, code revision, data revision, environment, configuration, seed, metrics, artifacts, cost, and conclusion.
- Use validation results for iteration, keep a test set untouched for infrequent decisions, and keep tournament matches independent of both.
- Teach concepts just before they become useful, then require prediction, implementation, measurement, and explanation.
- Prefer code that the learner can explain over machinery that merely produces a run.

## Verified starting context

- The repository began empty except for Git on branch `main`.
- Local development machine: Apple M1 Pro with 16 GB unified memory. It is suitable for data inspection, tests, and tiny teaching runs; serious tournament training will probably need rented or otherwise available accelerator compute.
- Python 3.13.0 and `uv` 0.10.4 are installed locally.
- The public dataset revision observed on 2026-07-24 was `cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94`, about 13.85 GB. This is an observation, not yet the agreed tournament freeze.
- The dataset card and the newer GPCT repository documentation currently describe different vocabulary generations (5,267 versus 5,273 tokens). That must be reconciled and pinned before training comparisons are meaningful.
- The only proposed tournament limit so far is the baseline author's: "So my current model is 85mish - I think we can cap at 50m?" This is recorded as a proposal, not an agreed or fully defined rule.
- Local-only work is the current compute plan. The learner has about one hour per week and is rated approximately 1350 blitz on chess.com.
- The GPCT source is [`shahazmat/chess-transformer-model`](https://github.com/shahazmat/chess-transformer-model). Commit `38536855597d064f4b5d04005ce1587f45359881` was inspected on 2026-07-24.
- At that commit, GPCT's `full` profile is 12 layers, 12 heads, width 768, context 512, and batch size 128. Its transformer blocks contain about 85.1M parameters; the complete tied-embedding nanoGPT model is about 89.5M parameters with a 5,273-token vocabulary. This explains why the meaning of "50M" must be explicit.

## Decisions still needed

- The complete fairness contract: parameter-count formula, context length, training-token or compute budget, allowed data transformations, inference method, legal-move masking, search, time controls, hardware normalization, and submission format.
- The authoritative frozen dataset/tokenizer revision and whether competitors may create their own train/validation/test split from it.
- The match protocol: colors, openings, number of games, sampling settings, illegal-move handling, adjudication, and rating/statistical method.
- Any eventual non-laptop training budget, deadline, and cloud provider constraints.
- The learner's current Python, calculus, linear algebra, probability, chess, and command-line comfort.
- Whether experiment tracking should remain fully local or may use a hosted service.
