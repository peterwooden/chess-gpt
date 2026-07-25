# Chess language-model engineering resources

## Knowledge

- [Dataset: `shazmate/lichess-chess-tokens`](https://huggingface.co/datasets/shazmate/lichess-chess-tokens)
  The shared candidate training corpus and tokenizer pipeline. Use for: schema inspection, data provenance, split design, vocabulary pinning, and the eventual frozen tournament snapshot.
- [GPCT: Generative Pretrained Chess Transformer](https://github.com/shahazmat/chess-transformer-model)
  The friend's public harness and training design, including legal-token masking, quality tokens, bare-history batching, and the browser inference interface. Use for: understanding the incumbent idea and agreeing compatibility rules—not as unquestioned ground truth.
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
