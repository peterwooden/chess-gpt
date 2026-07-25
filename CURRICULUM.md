# Chess GPT curriculum

This is the durable learning roadmap. Core outcomes are fixed; individual missions may change after diagnostics and experiments reveal the learner's zone of proximal development.

## Progress rules

- `[x]` means demonstrated mastery or a completed project prerequisite, as labelled—not merely exposure.
- A chapter is mastered only after its primary-source assignment, causal prediction, implementation comparison, experiment, checkpoint code, and explanation from memory are complete.
- Consequential experiments record the learner's predicted direction, rough magnitude, assumptions, confidence, and likely failure modes before execution.
- Lessons use a tiny inspectable example first and the real chess system second.
- The weekly cadence assumes five hours per week across several short missions, one focused source assignment, one experiment, and review.
- Chapter 1 will be written only after the placement diagnostic is reviewed.

## Course setup

- [x] Learning mission and tournament priorities recorded
- [x] Teaching contract agreed through learner interview
- [x] High-trust primary sources curated in [`RESOURCES.md`](RESOURCES.md)
- [x] First reproducible chess baseline trained as experiment `0001`
- [ ] Placement diagnostic completed on the learning site
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

Primary sources: Jurafsky & Martin, *Speech and Language Processing*, Chapter 3; Karpathy, “building makemore.”

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

Primary sources: Karpathy, “building micrograd”; Prince, *Understanding Deep Learning*, Chapters 5–7.

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

Primary sources: Karpathy, “building makemore” Parts 1–2; Jurafsky & Martin, Chapters 4–6.

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

Primary sources: Prince, Chapters 8–9; Karpathy, “A Recipe for Training Neural Networks.”

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

Primary sources: Karpathy, “Activations & Gradients” and “Backprop Ninja”; Prince, Chapters 6–7.

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

Primary sources: Prince, Chapter 12; Jurafsky & Martin, Chapter 8; Vaswani et al., “Attention Is All You Need.”

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

Primary sources: Karpathy, “Let’s build GPT”; Prince, Chapters 4, 7, 11–12.

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

Primary sources: Hoffmann et al., “Training Compute-Optimal Large Language Models”; Karpathy, `nanochat` implementation notes.

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

Primary sources: Jurafsky & Martin on language-model evaluation; project tournament protocol and measured match evidence.

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

Primary sources: Karpathy, “A Recipe for Training Neural Networks”; this repository's versioned experiment records.

## Graduation

- [ ] Independently explain every fundamental lever used by the submitted model
- [ ] Predict the likely consequences of changing data, architecture, optimization, compute, or evaluation
- [ ] Specify consequential experiments clearly enough for a coding agent to implement
- [ ] Select the strongest of multiple implementations and diagnose plausible failures
- [ ] Take an original chess-model idea from hypothesis to trustworthy match evidence
- [ ] Submit a rules-compliant tournament model and explain why it should win
