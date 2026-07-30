"""Play lab checkpoints: against random, each other, or engine rungs, with decode options."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import chess
import torch

from chess_gpt.snapshot_model import encode_board, move_index
from lab.model import TinyPolicy

Player = Callable[[chess.Board], chess.Move]


def load_model(checkpoint: Path) -> TinyPolicy:
    saved = torch.load(checkpoint, weights_only=True)
    if isinstance(saved, dict) and "config" in saved:
        model = TinyPolicy(**saved["config"])
        model.load_state_dict(saved["model"])
    else:  # legacy bench-S checkpoint
        model = TinyPolicy()
        model.load_state_dict(saved, strict=False)
    model.eval()
    return model


def _inputs(model: TinyPolicy, board: chess.Board) -> dict[str, torch.Tensor]:
    snapshot = encode_board(board)
    inputs: dict[str, torch.Tensor] = {
        "squares": torch.tensor([snapshot.squares], dtype=torch.long),
        "state": torch.tensor([snapshot.state], dtype=torch.long),
    }
    if model.history:
        moves = board.move_stack[-model.history :]
        pad = model.history - len(moves)
        inputs["history_from"] = torch.tensor(
            [[64] * pad + [m.from_square for m in moves]], dtype=torch.long
        )
        inputs["history_to"] = torch.tensor(
            [[64] * pad + [m.to_square for m in moves]], dtype=torch.long
        )
    if model.use_repetition:
        count = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
        inputs["repetition"] = torch.tensor([count], dtype=torch.long)
    return inputs


def _white_score(value_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(value_logits, dim=-1)
    return p[..., 0] + 0.5 * p[..., 1]


def make_player(
    checkpoint: Path,
    temperature: float = 0.0,
    top_k: int = 0,
    search: str = "none",
    contempt: float = 0.0,
    seed: int = 0,
) -> Player:
    model = load_model(checkpoint)
    rng = random.Random(seed)

    def policy_choice(board: chess.Board) -> chess.Move:
        with torch.no_grad():
            logits = model(**_inputs(model, board))["policy"][0]
        legal = list(board.legal_moves)
        scores = logits[torch.tensor([move_index(m) for m in legal])]
        if temperature > 0:
            keep = scores.topk(min(top_k, len(legal))).indices if top_k else torch.arange(len(legal))
            probabilities = torch.softmax(scores[keep] / temperature, dim=0)
            pick = keep[torch.multinomial(probabilities, 1, generator=None)].item()
            return legal[pick]
        return legal[int(scores.argmax())]

    def value_search_choice(board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        my_turn_is_white = board.turn == chess.WHITE
        terminal: dict[int, float] = {}
        batched: list[dict[str, torch.Tensor]] = []
        batch_slots: list[int] = []
        repetition_flags: list[bool] = []
        for slot, move in enumerate(legal):
            board.push(move)
            if board.is_checkmate():
                terminal[slot] = 1.0
            elif board.is_game_over(claim_draw=True):
                terminal[slot] = 0.5
            else:
                batched.append(_inputs(model, board))
                batch_slots.append(slot)
                repetition_flags.append(board.is_repetition(2))
            board.pop()
        scores = [0.0] * len(legal)
        for slot, value in terminal.items():
            scores[slot] = value
        if batched:
            stacked = {
                key: torch.cat([b[key] for b in batched]) for key in batched[0]
            }
            with torch.no_grad():
                white = _white_score(model(**stacked)["value"])
            for i, slot in enumerate(batch_slots):
                mine = float(white[i]) if my_turn_is_white else 1.0 - float(white[i])
                if contempt > 0 and repetition_flags[i] and mine > 0.55:
                    mine -= contempt
                scores[slot] = mine
        best = max(range(len(legal)), key=scores.__getitem__)
        return legal[best]

    return value_search_choice if search == "value1" else policy_choice


def load_policy(checkpoint: Path) -> Player:
    """Greedy argmax player — the ladder's standard decode."""
    return make_player(checkpoint)


def random_player(rng: random.Random) -> Player:
    return lambda board: rng.choice(list(board.legal_moves))


def random_opening(rng: random.Random, plies: int) -> list[chess.Move]:
    """A seeded random opening so deterministic players still produce distinct games."""
    while True:
        board = chess.Board()
        moves: list[chess.Move] = []
        for _ in range(plies):
            moves.append(rng.choice(list(board.legal_moves)))
            board.push(moves[-1])
        if not board.is_game_over():
            return moves


def play(white: Player, black: Player, opening: list[chess.Move], max_plies: int) -> str:
    board = chess.Board()
    for move in opening:
        board.push(move)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover(board))
    return board.result(claim_draw=True)


def play_series(
    candidate: Player,
    opponent: Player,
    games: int,
    rng: random.Random,
    opening_plies: int = 6,
    max_plies: int = 200,
) -> dict[str, int]:
    """Color-reversed opening pairs; returns the candidate's win/draw/loss tally."""
    tally = {"win": 0, "draw": 0, "loss": 0}
    for _ in range((games + 1) // 2):
        opening = random_opening(rng, opening_plies)
        for candidate_is_white in (True, False):
            white, black = (candidate, opponent) if candidate_is_white else (opponent, candidate)
            result = play(white, black, opening, max_plies)
            if result == "1-0":
                tally["win" if candidate_is_white else "loss"] += 1
            elif result == "0-1":
                tally["loss" if candidate_is_white else "win"] += 1
            else:
                tally["draw"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, help="Opponent checkpoint; random-legal if omitted")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--search", choices=("none", "value1"), default="none")
    parser.add_argument("--contempt", type=float, default=0.0)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidate = make_player(
        args.checkpoint, args.temperature, args.top_k, args.search, args.contempt, args.seed
    )
    opponent = make_player(args.checkpoint_b) if args.checkpoint_b else random_player(rng)
    tally = play_series(candidate, opponent, args.games, rng, args.opening_plies, args.max_plies)
    score = tally["win"] + 0.5 * tally["draw"]
    print(json.dumps({**tally, "score": score, "games": sum(tally.values())}))


if __name__ == "__main__":
    main()
