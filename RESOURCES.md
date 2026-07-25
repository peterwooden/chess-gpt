# Chess language-model engineering resources

## Knowledge

- [Dataset: `shazmate/lichess-chess-tokens`](https://huggingface.co/datasets/shazmate/lichess-chess-tokens)
  The shared candidate training corpus and tokenizer pipeline. Use for: schema inspection, data provenance, split design, vocabulary pinning, and the eventual frozen tournament snapshot.
- [GPCT: Generative Pretrained Chess Transformer](https://github.com/shahazmat/chess-transformer-model)
  A competitor's public model and inference interface. Use for: agreeing tournament compatibility and establishing a playing-strength benchmark—not as curriculum material or unquestioned ground truth.
- [Neural Networks: Zero to Hero — Andrej Karpathy](https://karpathy.ai/zero-to-hero.html)
  The primary teaching spine from scalar backpropagation through language modeling, train/dev/test splits, diagnostics, and GPT. Use for: first-principles lessons and exercises in the same conceptual order.
- [Code and exercises for Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
  Karpathy's notebooks and exercise links. Use for: active practice and checking our chess-adapted implementations against the original teaching examples.
- [A Recipe for Training Neural Networks — Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/)
  A data-first, simple-to-complex debugging discipline: inspect data, establish baselines, overfit small batches, then regularize and tune. Use for: the operating procedure behind every model milestone.
- [Build nanoGPT](https://github.com/karpathy/build-nanogpt)
  A clean commit-by-commit construction of GPT-2. Use for: seeing how a transparent teaching implementation grows into a recognizable training program.
- [nanochat](https://github.com/karpathy/nanochat)
  Karpathy's current minimal experimental harness; nanoGPT is deprecated in its favor. Use for: modern PyTorch patterns, profiling, scaling experiments, and evaluation ideas after fundamentals are secure—not as a dependency to copy wholesale.
- [Eureka Labs and the LLM101n announcement](https://eurekalabs.ai/)
  Useful statement of the teacher-plus-AI educational vision. The linked LLM101n repository is archived and explicitly says the course does not yet exist, so it is inspiration rather than available curriculum.
- [Textbook: *Understanding Deep Learning* — Simon J. D. Prince](https://udlbook.github.io/udlbook/)
  A free, visual, implementation-minded textbook with short chapters. Use chapters 2–9 for supervised learning, networks, loss, optimization, performance, and regularization; use chapter 12 when we reach transformers.
- [Textbook: *Speech and Language Processing*, 3rd-edition draft — Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
  A current language-modeling spine that progresses through tokens, n-grams, logistic regression, embeddings, neural networks, LLMs, and transformers. Use for: the probabilistic meaning and motivation behind each chess next-move model.
- [Textbook: *Mathematics for Machine Learning* — Deisenroth, Faisal, and Ong](https://mml-book.github.io/)
  A free bridge from intuition to the necessary linear algebra, vector calculus, probability, and optimization. Use selectively when an experiment creates a concrete mathematical question, not as a cover-to-cover prerequisite.
- [Interactive book: *Dive into Deep Learning*](https://d2l.ai/)
  Code-first chapters on tensors, automatic differentiation, softmax, multilayer perceptrons, optimization, attention, and transformers. Use for: a second executable explanation when our own tiny implementation needs another angle.
- [Paper: “Attention Is All You Need”](https://arxiv.org/abs/1706.03762)
  The original Transformer paper. Use for: checking our eventual causal self-attention implementation against the architecture's primary source after the mechanism is intuitive.
- [Paper: “Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556)
  The Chinchilla scaling study relating model parameters, training tokens, and fixed compute. Use for: motivating tournament-budget allocation once we can measure our own tokens and FLOPs.
- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness)
  Official limits and controls for randomness and deterministic algorithms. Use for: experiment manifests, seed handling, debugging mode, and honest reproducibility claims.
- [PyTorch MPS backend notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
  Official guide to running tensors and models through Metal Performance Shaders on Apple silicon. Use for: small local training runs and understanding where laptop results may differ from later CUDA runs.
- [`uv` project locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
  Current environment-locking behavior and reproducible install commands. Use for: dependency management and CI runs from the checked-in lockfile.

## Wisdom (Communities)

- [Karpathy's nanochat Discussions](https://github.com/karpathy/nanochat/discussions)
  A practitioner forum around small-model training, scaling, evaluation, and current implementation trade-offs. Use for: checking hard-won practical assumptions after we can formulate a precise question.
- [GPCT repository issues](https://github.com/shahazmat/chess-transformer-model/issues)
  The closest project-specific place to resolve tokenizer, data, and match-interface ambiguities with the baseline author.

## Gaps

- A written, agreed tournament fairness and match protocol from all three competitors.
- A frozen dataset/tokenizer revision whose files, hashes, and vocabulary generation agree.
- A primary tournament baseline report with exact parameter count, training budget, seeds, checkpoint, and match results.
- The learner's compute budget and deadline, needed before choosing the production training plan.
