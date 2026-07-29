from __future__ import annotations

import chess
import torch

from chess_gpt.snapshot_match import choose_move, play_game
from chess_gpt.snapshot_model import ModelConfig, SnapshotPolicy


def test_checkpoint_policy_always_selects_a_legal_move() -> None:
    board = chess.Board()
    model = SnapshotPolicy(ModelConfig(d_model=16, layers=1, heads=2, ff_multiplier=2))

    move = choose_move(model, board, torch.device("cpu"))

    assert move in board.legal_moves


def test_game_length_cap_is_reported_as_a_draw() -> None:
    board = chess.Board()
    model = SnapshotPolicy(ModelConfig(d_model=16, layers=1, heads=2, ff_multiplier=2))

    winner, plies, termination = play_game(
        board, white=model, black=model, device=torch.device("cpu"), max_plies=1
    )

    assert winner is None
    assert plies == 1
    assert termination == "max_plies"
