# Teaching notes

## Learner preferences

- Treat the user as a student and collaborative researcher, not as a passive recipient of generated code.
- Teach in Karpathy-like increments: start from a small mechanism that can be inspected, build intuition in code, and add complexity only after the previous step is understood.
- Tie theory to the chess tournament whenever possible.
- Do not hide important behavior behind a framework before the learner has implemented or inspected the underlying idea.
- Use predictions, small exercises, and explanations from memory to build durable understanding—not only fluent recognition.
- Preserve room for creative hypotheses, but demand controlled comparisons and negative results.
- Plan for about five hours per week, preferably spread across several sessions. Organize chapters into three or four focused 10–15 minute missions plus primary-source study, experiment work, and spaced review.
- Work locally on the 16 GB M1 Pro for now. Use CPU/MPS for tiny models, tests, and profiling; do not design the early curriculum around cloud infrastructure.
- The learner reports an approximately 1350 chess.com blitz rating. Use real chess examples, but verify familiarity with SAN rather than assuming it from rating alone.
- The latest placement diagnostic attempt scored 7/8 on the direct track: it showed working intuition for a simple Python loop, negative log-likelihood, gradient direction, learning-rate instability, validation-size and capacity trade-offs, and tensor shapes. A follow-up explanation demonstrated why repeated test-set model selection creates optimistic bias without test-set gradient updates, and how more candidates or a larger test set change the effect. Broader Python, terminal/Git, calculus, and linear-algebra comfort remain only lightly sampled.
- The learner usually accesses Codex from a phone through a remote session, where local HTML links display source instead of a rendered page; publish interactive lessons through Sites and return a normal web URL.
- Interactive lessons should gate a short completion code behind correct retrieval answers so the learner can paste it back as lightweight evidence of completion.
- Treat causal prediction as the central learning skill. Before changing a fundamental lever, require the learner to predict direction, rough magnitude, assumptions, confidence, and likely failure modes.
- Do not require hand-written implementations as proof of understanding. Prefer exercises that compare three candidate implementations, identify defects, specify work for a coding agent, diagnose evidence, and explain surprises.
- Teach mathematics to equation-literacy depth: understand every symbol, calculate tiny examples, and use equations to make predictions; omit formal proofs unless they unlock intuition.
- Structure the curriculum as causal questions in dependency order. Show the whole roadmap, unlock lessons sequentially, and keep core outcomes fixed while adapting missions to diagnostic and experiment evidence.
- Gate each chapter with a causal prediction, implementation-comparison challenge, and explanation of an observed result. Unlimited retries are allowed; record mastery only after the learner pastes the completion code and explains one prediction in their own words.
- Use one purposeful 20–40 minute primary-source assignment per chapter rather than assigning whole books or multi-hour lectures without a concrete question.
- Keep site progress device-local and account-free. Completion codes and learning records are the durable cross-device record.
- Teach each mechanism first with a tiny fully inspectable example, then apply it immediately to the real chess data or model.
- Collaborate on consequential choices involving data, targets, architecture, optimization, compute allocation, or evaluation. Routine agreed mechanical work may proceed without interruption.
- Record the learner's prediction in every consequential versioned experiment before running it, then compare prediction with evidence afterward.
- Present the site as a serious interactive field notebook: causal diagrams, small numerical simulators, implementation comparisons, experiment records, and restrained progress indicators rather than gamification.
- Include a post-core reinforcement-learning extension grounded in chess, with self-play gated on an explicit tournament ruling about generated training experience and compute accounting.
- Give every core and extension chapter a curated reading trail of one to three primary sources and one to three secondary sources, pinpointing the useful section rather than assigning whole works by default.
- Commit and push completed repository changes to `main` by default unless the learner explicitly asks to keep them local.
- Do not incorporate competitors' distinctive techniques into lessons or experiments unless the learner explicitly chooses to study them.

## Default teaching loop

1. State the concrete chess or engineering question.
2. Ask the learner to predict direction, magnitude, assumptions, confidence, and failure modes.
3. Have a coding agent build the smallest observable version after the consequential choices are agreed.
4. Measure it and inspect failures.
5. Have the learner explain the result in their own words.
6. Record the non-obvious lesson and only then add complexity.

## Curriculum direction

Use Karpathy's public material as a spine, adapted to this project rather than copied as a detached course:

1. Become one with the chess data and tokenizer.
2. Learn scalar computation graphs and backpropagation.
3. Learn tensors, logits, softmax, cross-entropy, batching, and optimization with a tiny next-move model.
4. Learn train/validation/test discipline, overfitting, regularization, and diagnostics.
5. Build causal self-attention and a tiny GPT-style chess model from understandable pieces.
6. Turn the teaching implementation into a tested, profiled experiment harness.
7. Establish a strong baseline, then run creative ablations and tournament matches.

## Guardrails for the teacher

- Never use the final test set for routine model selection.
- Never call two runs comparable unless their data, evaluation semantics, and budgets match.
- Never accept a training curve without checking data examples, a trivial baseline, small-batch overfitting, and task-level chess metrics.
- Distinguish deterministic debugging from statistical reproducibility; identical seeds do not guarantee identical results across PyTorch versions or hardware.
- When code is generated for the learner, teach the important parts and leave meaningful retrieval or implementation work for the learner.
