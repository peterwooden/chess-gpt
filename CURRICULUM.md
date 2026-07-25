# Chess GPT curriculum

This is the durable learning roadmap. Core outcomes are fixed; individual missions may change after diagnostics and experiments reveal the learner's zone of proximal development.

## Progress rules

- `[x]` means demonstrated mastery or a completed project prerequisite, as labelled—not merely exposure.
- A chapter is mastered only after its primary-source assignment, causal prediction, implementation comparison, experiment, checkpoint code, and explanation from memory are complete.
- Consequential experiments record the learner's predicted direction, rough magnitude, assumptions, confidence, and likely failure modes before execution.
- Lessons use a tiny inspectable example first and the real chess system second.
- The weekly cadence assumes five hours per week across several short missions, one focused source assignment, one experiment, and review.
- Chapter 1 will be written only after the placement diagnostic is reviewed.

## Reading strategy for *Deep Learning*

Goodfellow, Bengio, and Courville is a companion reference, not a cover-to-cover prerequisite. Chapters 3–8 and 11 support the core probability, optimization, generalization, and debugging work; Part III contributes section 15.4 on distributed representations to Chapter 3 and section 17.1 on Monte Carlo estimation to the RL extension. The rest of Part III is optional because its autoencoder, graphical-model, partition-function, approximate-inference, and older generative-model material does not directly advance our small Transformer unless a later experiment creates a reason to study it.

### How to use the reading lists

- **Primary sources** are original evidence or definitions: papers, specifications, official documentation, datasets, and reference implementations.
- **Secondary sources** are carefully chosen explanations: textbooks, lectures, and technical guides that help build the mental model before reading original work.
- **Reading budget** defaults to one primary and one secondary selection per chapter, chosen with the teacher for the current mission; the remaining links are for clarification or deeper study.

## Course setup

- [x] Learning mission and tournament priorities recorded
- [x] Teaching contract agreed through learner interview
- [x] High-trust primary sources curated in [`RESOURCES.md`](RESOURCES.md)
- [x] First reproducible chess baseline trained as experiment `0001`
- [x] Placement diagnostic completed on the learning site
- [ ] Diagnostic code and one prediction explained to the teacher
- [ ] Adaptive Chapter 1 missions generated

## Chapter 1 — What does it mean for a chess model to learn?

**Motivation:** distinguish learning a reusable pattern from memorizing games or copying a frequent move.

- [ ] Define example, context, target, prediction, parameter, and baseline
- [ ] Read a deterministic data split without leaking validation or test games
- [ ] Predict effects of changing training, validation, and test-set sizes
- [ ] Explain conditional probability through the existing SAN n-gram baseline
- [ ] Compare three data-and-split implementations and choose the trustworthy one
- [ ] Run the Chapter 1 chess experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [the pinned Lichess dataset artifact](https://huggingface.co/datasets/shazmate/lichess-chess-tokens/tree/cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94) — inspect the actual schema and provenance; [Kohavi, “A Study of Cross-Validation and Bootstrap”](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf) — focus on the experimental comparison of validation methods.
- **Secondary:** [Jurafsky & Martin, *Speech and Language Processing*, Chapter 3](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — n-grams, evaluation, and perplexity; [Goodfellow et al., *Deep Learning*, §§5.1–5.5](https://www.deeplearningbook.org/contents/ml.html) — learning algorithms, capacity, splits, estimators, and likelihood.

## Chapter 2 — How can a number learn to prefer a move?

**Motivation:** replace fixed counts with parameters that improve when predictions are wrong.

- [ ] Explain objectives, hill climbing, loss functions, and optimization landscapes
- [ ] Calculate squared error and negative log-likelihood on tiny examples
- [ ] Predict parameter motion from the sign and size of a derivative
- [ ] Trace a scalar computation graph forward and backward
- [ ] Distinguish gradient descent from blind search and finite differences
- [ ] Compare three optimization implementations and locate the broken update
- [ ] Run the Chapter 2 chess experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Rumelhart, Hinton & Williams, “Learning representations by back-propagating errors”](https://doi.org/10.1038/323533a0) — the original short backpropagation account; [Karpathy's `micrograd`](https://github.com/karpathy/micrograd) — inspect the scalar autograd engine and its tests as a reference implementation.
- **Secondary:** [Karpathy, “The spelled-out intro to neural networks and backpropagation”](https://karpathy.ai/zero-to-hero.html) — watch Lecture 1; [Goodfellow et al., §4.3](https://www.deeplearningbook.org/contents/numerical.html) — gradient-based optimization; [Goodfellow et al., §6.5](https://www.deeplearningbook.org/contents/mlp.html) — backpropagation.

## Chapter 3 — How does a model turn move history into probabilities?

**Motivation:** generalize beyond exact histories by representing moves as learned vectors.

- [ ] Read tensor shapes, indexing, broadcasting, and matrix multiplication
- [ ] Explain embeddings as learned lookup tables
- [ ] Convert logits to probabilities with softmax
- [ ] Connect cross-entropy to assigning probability to the played move
- [ ] Predict effects of vocabulary size, embedding width, and batch size
- [ ] Compare three batched next-move implementations
- [ ] Train the first neural chess language model
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Bengio et al., “A Neural Probabilistic Language Model”](https://www.jmlr.org/papers/v3/bengio03a.html) — focus on learned distributed representations and unseen sequences; [Mikolov et al., “Efficient Estimation of Word Representations in Vector Space”](https://arxiv.org/abs/1301.3781) — focus on how a prediction task shapes embeddings.
- **Secondary:** [Karpathy's makemore lectures](https://karpathy.ai/zero-to-hero.html) — Lectures 2–4 on language models, tensors, and activations; [Jurafsky & Martin, Chapters 4–7](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — classifiers through neural language models; [Goodfellow et al., Part III §15.4](https://www.deeplearningbook.org/contents/representation.html) — why distributed representations share statistical strength.

## Chapter 4 — Did the model learn or memorize?

**Motivation:** decide whether an apparent improvement will survive unseen games.

- [ ] Distinguish training, validation, test, and tournament evidence
- [ ] Explain capacity, underfitting, overfitting, and generalization
- [ ] Predict how more data or a larger model changes train and validation loss
- [ ] Overfit one tiny batch as a debugging test
- [ ] Explain regularization, early stopping, and weight decay
- [ ] Diagnose representative learning curves
- [ ] Run a controlled capacity experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Zhang et al., “Understanding Deep Learning Requires Rethinking Generalization”](https://research.google/pubs/understanding-deep-learning-requires-rethinking-generalization/) — the random-label memorization experiments; [Nakkiran et al., “Deep Double Descent”](https://arxiv.org/abs/1912.02292) — inspect the model-size, data-size, and training-time curves.
- **Secondary:** [Karpathy, “A Recipe for Training Neural Networks”](https://karpathy.github.io/2019/04/25/recipe/) — data inspection, baselines, overfitting one batch, then regularization; [Goodfellow et al., §§5.2–5.4 and Chapter 7](https://www.deeplearningbook.org/contents/regularization.html) — capacity, bias/variance, and regularization.

## Chapter 5 — How fast should the model learn?

**Motivation:** make optimization fast enough to use the budget without becoming unstable.

- [ ] Explain stochastic gradient descent, batches, epochs, and training tokens
- [ ] Predict under- and over-sized learning-rate behaviour
- [ ] Interpret activation and gradient statistics
- [ ] Explain initialization and why scale propagates through depth
- [ ] Compare SGD, momentum, and AdamW at mechanism level
- [ ] Predict effects of batch size and learning-rate schedules
- [ ] Run a learning-rate range experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Kingma & Ba, “Adam”](https://arxiv.org/abs/1412.6980) — Algorithm 1 and moment estimates; [Loshchilov & Hutter, “Decoupled Weight Decay Regularization”](https://arxiv.org/abs/1711.05101) — why AdamW is not Adam plus an L2 penalty; [Smith, “Cyclical Learning Rates”](https://arxiv.org/abs/1506.01186) — the learning-rate range test.
- **Secondary:** [Karpathy's “Activations & Gradients” and “Backprop Ninja”](https://karpathy.ai/zero-to-hero.html) — Lectures 5–6; [Goodfellow et al., Chapter 8](https://www.deeplearningbook.org/contents/optimization.html) — optimization for deep models; [Prince, *Understanding Deep Learning*, Chapters 6–7](https://udlbook.github.io/udlbook/) — gradients and fitting models.

## Chapter 6 — How can the model use any earlier move?

**Motivation:** let a prediction depend on relevant game history rather than a fixed short window.

- [ ] Explain context length and positional information
- [ ] Calculate one query-key similarity and weighted value sum
- [ ] Explain causal masking without implementation folklore
- [ ] Predict effects of context length, head count, and head dimension
- [ ] Distinguish attention capacity from guaranteed chess understanding
- [ ] Compare three causal self-attention implementations
- [ ] Run an attention toy and chess-context experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Vaswani et al., “Attention Is All You Need”](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) — §§3.1–3.2 and Figure 1; [Su et al., “RoFormer”](https://arxiv.org/abs/2104.09864) — the motivation and equations for rotary position embeddings.
- **Secondary:** [Alammar, “The Illustrated Transformer”](https://jalammar.github.io/illustrated-transformer/) — visual query/key/value intuition; [Prince, *Understanding Deep Learning*, Chapter 12](https://udlbook.github.io/udlbook/) — Transformers; [Jurafsky & Martin, Chapter 8](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — attention and Transformers for language.

## Chapter 7 — What makes a Transformer trainable?

**Motivation:** assemble attention into a stable model that fits the tournament size limit.

- [ ] Explain the Transformer block as attention, MLP, residual paths, and normalization
- [ ] Track shapes and parameter counts through a complete block
- [ ] Predict effects of depth, width, context, and MLP expansion
- [ ] Explain residual connections and layer normalization mechanistically
- [ ] Diagnose initialization, gradient, and numerical-instability failures
- [ ] Compare three small GPT architectures under the same size budget
- [ ] Train the first tiny Transformer baseline
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Ba, Kiros & Hinton, “Layer Normalization”](https://arxiv.org/abs/1607.06450) — the normalization rule; [He et al., “Deep Residual Learning”](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) — why residual paths ease optimization; [Radford et al., “Language Models are Unsupervised Multitask Learners”](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — §2 for the GPT-2 architecture.
- **Secondary:** [Karpathy, “Let's build GPT”](https://karpathy.ai/zero-to-hero.html) — Lecture 7; [Karpathy's `build-nanogpt`](https://github.com/karpathy/build-nanogpt) — follow the architecture-building commits; [Dive into Deep Learning, “The Transformer Architecture”](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) — shapes and block composition.

## Chapter 8 — How should we spend 100 MB and one exaFLOP?

**Motivation:** allocate a fixed tournament budget between model capacity, data, precision, and training duration.

- [ ] Calculate parameter memory at common numeric precisions
- [ ] Estimate training FLOPs and tokens from a run configuration
- [ ] Explain throughput, utilization, and bottlenecks
- [ ] Predict effects of mixed precision and quantization
- [ ] Explain the model-size/data-size trade-off behind compute-optimal scaling
- [ ] Compare three budget allocations before profiling them
- [ ] Run a local scaling and profiling experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Hoffmann et al., “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556) — model/data allocation under fixed compute; [Micikevicius et al., “Mixed Precision Training”](https://arxiv.org/abs/1710.03740) — precision, master weights, and loss scaling; [Frantar et al., “GPTQ”](https://arxiv.org/abs/2210.17323) — post-training weight quantization.
- **Secondary:** [DeepMind's Chinchilla explainer](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/) — read the model-size versus token trade-off; [Karpathy's `nanochat`](https://github.com/karpathy/nanochat) — inspect speed, depth, and scaling documentation; [Hugging Face, “Model memory anatomy”](https://huggingface.co/docs/transformers/model_memory_anatomy) — account for parameters, gradients, optimizer state, and activations.

## Chapter 9 — Does lower loss mean stronger chess?

**Motivation:** select the model most likely to win games, not merely the model with the prettiest training curve.

- [ ] Interpret loss, perplexity, top-k accuracy, legality, and calibration
- [ ] Explain why the played human move is not the only reasonable move
- [ ] Predict how decoding, temperature, legal masking, and search affect play
- [ ] Design paired matches with reversed colors and shared openings
- [ ] Interpret score, Elo estimates, uncertainty, and practical significance
- [ ] Compare three evaluation protocols and reject biased designs
- [ ] Run a paired baseline match
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Guo et al., “On Calibration of Modern Neural Networks”](https://proceedings.mlr.press/v70/guo17a.html) — reliability diagrams and temperature scaling; [Glickman, “The US Chess Rating System”](https://www.glicko.net/ratings/rating.system.pdf) — expected scores, updates, and uncertainty; [`cutechess-cli` reference](https://github.com/cutechess/cutechess/blob/master/docs/cutechess-cli.6) — the actual automated-match controls.
- **Secondary:** [Jurafsky & Martin, Chapter 3 evaluation sections](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — loss and perplexity; [Chessprogramming Wiki, “Sequential Probability Ratio Test”](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) — how engine developers test Elo changes efficiently.

## Chapter 10 — How do we turn an idea into evidence?

**Motivation:** make creative model ideas scientifically credible and competitively useful.

- [ ] State a falsifiable hypothesis and one intentional change
- [ ] Separate exploratory validation from final test evidence
- [ ] Design ablations and matched-budget comparisons
- [ ] Record seeds, revisions, environment, compute, metrics, and artifact hashes
- [ ] Explain negative and surprising results without hindsight editing
- [ ] Review three experiment proposals and select the identifiable one
- [ ] Complete an original end-to-end experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness) — platform limits, seeds, and deterministic algorithms; [Pineau et al., “Improving Reproducibility in Machine Learning Research”](https://www.jmlr.org/papers/v22/20-303.html) — the NeurIPS reproducibility programme and checklist; [Henderson et al., “Deep Reinforcement Learning That Matters”](https://ojs.aaai.org/index.php/AAAI/article/view/11694) — seeds, reporting, and fair empirical comparisons.
- **Secondary:** [Karpathy, “A Recipe for Training Neural Networks”](https://karpathy.github.io/2019/04/25/recipe/) — disciplined iteration; [Goodfellow et al., Chapter 11](https://www.deeplearningbook.org/contents/guidelines.html) — metrics, baselines, and debugging; [Spinning Up, “Doing Rigorous Research in RL”](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl) — matched baselines, seeds, final runs, and ablations.

## Phase II — Reinforcement-learning extension

This optional extension begins only after Chapter 9, when the learner can already train and evaluate the supervised policy. Any self-play or environment interaction used to update the tournament model counts as training; before doing that, the three competitors must explicitly agree that it satisfies the shared-data rule and that all such compute counts toward the one-exaFLOP budget. If it is not allowed, the conceptual missions still run on tiny environments and archived chess evidence without changing the tournament submission.

### RL extension 1 — When is predicting human moves the wrong objective?

**Motivation:** distinguish imitating the moves in the dataset from choosing moves that maximize the eventual game result.

- [ ] Map chess history, legal SAN moves, policy outputs, trajectories, rewards, and returns onto RL notation
- [ ] Distinguish supervised behaviour cloning, offline RL, online RL, and planning
- [ ] Explain why the existing next-move model is already a categorical policy
- [ ] Predict when optimizing game result should help or damage an imitation-trained model
- [ ] Compare three proposed reward specifications and identify exploitable proxies
- [ ] Run a toy bandit before mapping the mechanism to chess
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Ross, Gordon & Bagnell, “A Reduction of Imitation Learning…”](https://proceedings.mlr.press/v15/ross11a.html) — why sequential imitation violates i.i.d. assumptions; [Silver et al., “Mastering Chess and Shogi by Self-Play”](https://arxiv.org/abs/1712.01815) — contrast human-data imitation with outcome-driven self-play.
- **Secondary:** [Spinning Up, “Key Concepts in RL”](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — states, actions, policies, trajectories, rewards, and returns; [Sutton & Barto, Chapters 2–3](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) — bandits and finite Markov decision processes.

### RL extension 2 — How does a final result teach earlier moves?

**Motivation:** assign credit through a long game when the clearest reward arrives at checkmate.

- [ ] Calculate finite and discounted returns from a tiny trajectory
- [ ] Explain state values, action values, Bellman relationships, and advantage
- [ ] Compare Monte Carlo and temporal-difference targets by bias, variance, and data efficiency
- [ ] Predict how reward sparsity and discounting change what the model learns
- [ ] Diagnose three value-target implementations, including one with information leakage
- [ ] Estimate a toy chess position's value from sampled continuations
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Sutton, “Learning to Predict by the Methods of Temporal Differences”](https://doi.org/10.1007/BF00115009) — the original TD formulation; [Watkins & Dayan, “Q-learning”](https://doi.org/10.1007/BF00992698) — action values and convergence conditions.
- **Secondary:** [Sutton & Barto, Chapters 3–6](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) — MDPs through temporal-difference learning; [Spinning Up on value functions](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#value-functions) — concise notation and relationships; [Goodfellow et al., Part III §17.1](https://www.deeplearningbook.org/contents/monte_carlo.html) — sampling and Monte Carlo estimates.

### RL extension 3 — Should we learn values, actions, or both?

**Motivation:** choose an algorithm family that matches chess's discrete actions, sparse outcomes, pretrained policy, and small compute budget.

- [ ] Distinguish value-based, policy-gradient, actor-critic, model-based, and search-assisted approaches
- [ ] Explain the policy-gradient update as log-probability weighted by estimated advantage
- [ ] Predict how a baseline changes gradient variance without changing the expected gradient
- [ ] Explain exploration, entropy, on-policy data, and distribution shift
- [ ] Compare three policy-gradient implementations and find the silent mathematical error
- [ ] Run a tiny policy-gradient experiment, then specify a supervised-checkpoint fine-tuning experiment
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Williams, “Simple Statistical Gradient-Following Algorithms…”](https://doi.org/10.1007/BF00992696) — REINFORCE; [Schulman et al., “Generalized Advantage Estimation”](https://arxiv.org/abs/1506.02438) — the bias/variance trade-off in advantage estimates; [Schulman et al., “Proximal Policy Optimization Algorithms”](https://arxiv.org/abs/1707.06347) — the clipped surrogate objective.
- **Secondary:** [Spinning Up, “Intro to Policy Optimization”](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — derive the simplest policy gradient; [Spinning Up's PPO guide](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — mechanism and pseudocode; [Sutton & Barto, Chapter 13](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf) — policy-gradient methods.

### RL extension 4 — Can self-play make the model stronger without fooling us?

**Motivation:** improve actual match performance while controlling opponent drift, random-seed variance, compute, and evaluation bias.

- [ ] Explain self-play as a changing data distribution and opponent population
- [ ] Distinguish the roles of policy, value model, tree search, and generated experience in AlphaZero
- [ ] Predict failure modes including collapse, cycling, reward hacking, and catastrophic forgetting
- [ ] Design opponent snapshots, shared openings, reversed colors, multiple seeds, and rollback criteria
- [ ] Account for environment inference and updates inside the training-compute budget
- [ ] Compare the supervised checkpoint and RL-tuned checkpoint with a preregistered paired match
- [ ] Complete the mastery checkpoint and explanation

**Further reading**

- **Primary:** [Silver et al., “Mastering Chess and Shogi by Self-Play”](https://arxiv.org/abs/1712.01815) — policy/value/search/self-play loop; [Lanctot et al., “A Unified Game-Theoretic Approach to Multiagent RL”](https://proceedings.neurips.cc/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html) — training against policy populations rather than one drifting opponent; [Henderson et al., “Deep Reinforcement Learning That Matters”](https://ojs.aaai.org/index.php/AAAI/article/view/11694) — variance and experimental integrity.
- **Secondary:** [DeepMind, “AlphaZero: Shedding new light on chess…”](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) — high-level system explanation; [Spinning Up, “Doing Rigorous Research in RL”](https://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl) — seeds, preregistered final runs, and ablations; [Chessprogramming Wiki on SPRT](https://www.chessprogramming.org/Sequential_Probability_Ratio_Test) — connect learning claims to match evidence.

## RL extension graduation

- [ ] Explain when RL is the right tool and when supervised learning or search is the cleaner choice
- [ ] Predict how reward, discounting, exploration, opponent choice, and algorithm family affect learning
- [ ] Specify a budgeted, rules-compliant RL experiment clearly enough for a coding agent to implement
- [ ] Diagnose a plausible RL failure using learning curves, value estimates, entropy, and match evidence
- [ ] Demonstrate a trustworthy playing-strength comparison or explain why the experiment should be rejected

## Graduation

- [ ] Independently explain every fundamental lever used by the submitted model
- [ ] Predict the likely consequences of changing data, architecture, optimization, compute, or evaluation
- [ ] Specify consequential experiments clearly enough for a coding agent to implement
- [ ] Select the strongest of multiple implementations and diagnose plausible failures
- [ ] Take an original chess-model idea from hypothesis to trustworthy match evidence
- [ ] Submit a rules-compliant tournament model and explain why it should win
