# Versioned experiments

Each numbered TOML file is a planned experiment and belongs in Git. It states the question and hypothesis before the result is known. Material changes get a new number; do not quietly rewrite history after a run.

Generated metrics, logs, and checkpoints go under `runs/<experiment-id>/` and stay out of Git. A later runner will capture the Git commit, `uv.lock` hash, dataset revision, seeds, hardware, duration, token count, metrics, and artifact hashes automatically.
