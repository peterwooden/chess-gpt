"""Pit a lab checkpoint against a legacy tournament snapshot checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import torch

from chess_gpt.snapshot_model import ModelConfig, SnapshotPolicy, encode_board, move_index
from lab.match import Player, make_player, play_series


def legacy_player(checkpoint: Path) -> Player:
    """The legacy model exactly as submitted: greedy argmax over legal moves."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("model_type") != "board_snapshot_policy":
        raise SystemExit("not a legacy board snapshot checkpoint")
    model = SnapshotPolicy(ModelConfig(**raw["model_config"]))
    model.load_state_dict(raw["state_dict"])
    model.eval()

    def choose(board: chess.Board) -> chess.Move:
        snapshot = encode_board(board)
        with torch.no_grad():
            logits = model(
                torch.tensor([snapshot.squares], dtype=torch.long),
                torch.tensor([snapshot.state], dtype=torch.long),
                torch.tensor([snapshot.phase], dtype=torch.long),
            )[0]
        legal = list(board.legal_moves)
        indices = torch.tensor([move_index(move) for move in legal])
        return legal[int(logits[indices].argmax())]

    return choose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--search", choices=("none", "value1", "beam"), default="beam")
    parser.add_argument("--contempt", type=float, default=0.15)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    candidate = make_player(args.checkpoint, search=args.search, contempt=args.contempt)
    opponent = legacy_player(args.legacy)
    rng = random.Random(args.seed)
    tally = play_series(candidate, opponent, args.games, rng)
    score = tally["win"] + 0.5 * tally["draw"]
    print(json.dumps({**tally, "score": score, "games": sum(tally.values())}))


if __name__ == "__main__":
    main()
