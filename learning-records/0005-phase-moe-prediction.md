# Phase-MoE prediction

Recorded on 2026-07-29 before material training.

The learner predicts that the phase-routed mixture-of-experts model will win 70% of its head-to-head games against the shared snapshot model, with 70% confidence. The stated assumption is that the shared encoder represents board state in an information-efficient form that the experts can use. The predicted failure mode is a router failure at tournament inference.

The learner also identified an unresolved decoder question: whether the policy should return probabilities and where board representation becomes a move. The implemented answer is now available for retrieval but is not yet recorded as learned: the network's linear policy head maps the encoded position to 4,272 move logits; cross-entropy consumes logits during training; the adapter masks illegal moves and maps the winning move identity to exact SAN. Softmax probabilities are optional because they preserve argmax ordering.
