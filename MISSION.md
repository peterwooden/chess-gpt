# Mission: Build a tournament-winning small chess language model

## Why

Build the strongest fair chess model in a three-friend tournament while learning to work like an ML engineer from first principles. Use the project as a creative laboratory where chess ideas become clear, reproducible experiments rather than guesses.

## Success looks like

- Field a rules-compliant model and give it the best defensible chance of winning the tournament.
- Explain, implement, and debug the path from chess data and tokens through loss, gradients, training, evaluation, inference, and match play.
- Reproduce any reported result from a pinned data revision, Git commit, locked environment, explicit configuration, and recorded random seeds.
- Maintain honest train, validation, test, and tournament evaluation boundaries, with the exact split policy agreed before tuning.
- Turn original ideas into controlled experiments with stated hypotheses, measurements, and conclusions.

## Constraints

- All three competitors will use the same frozen training data and an agreed model-size limit; the exact fairness contract is still to be defined.
- The learner is not an ML engineer yet, so teaching must build intuition incrementally and connect every abstraction to code and evidence.
- Competition comes first, learning second, and creative expression third; good work should serve all three where possible.
- Use current, reproducible ML engineering practices and avoid unexplained shortcuts.

## Out of scope

- Using outside training data or pretrained weights unless the tournament rules explicitly allow them.
- Building elaborate infrastructure before a small, trustworthy end-to-end baseline works.
- Treating a lower validation loss as proof of a stronger chess player without match-based evaluation.
