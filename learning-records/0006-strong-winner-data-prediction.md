# Strong-winner data-selection prediction

Recorded before material training on 2026-07-30.

The learner predicted that retraining both policies only on moves made by decisive-game winners rated at least 1600 would make each new model win 80% of games against its corresponding old model, with 40% confidence. The losing player's rating was deliberately left unrestricted. The stated assumption was that players below ELO 1600 would not generate useful data for modelling chess at the lower end. The predicted failure mode was that the filtered models would play very poorly against poor players.

This is evidence of a complete causal prediction—direction, magnitude, assumption, confidence, and failure mode—not evidence that the data-selection mechanism is understood or that the prediction was correct. The paired matches support the predicted direction but not the magnitude. They did not test the stated weak-opponent failure mode.
