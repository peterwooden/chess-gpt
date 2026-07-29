"""Run a deterministic paired match between two snapshot-policy checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import pyarrow.parquet as pq
import torch

from chess_gpt.snapshot_model import ModelConfig, SnapshotPolicy, encode_board, move_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_worktree_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )


def _board_from_row(row: dict[str, Any]) -> chess.Board:
    board = chess.Board.empty()
    for square, code in enumerate(row["squares"]):
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            board.set_piece_at(square, chess.Piece(code if code <= 6 else code - 6, color))
    state = row["state"]
    board.turn = chess.BLACK if state[0] else chess.WHITE
    board.castling_rights = sum(
        chess.BB_SQUARES[square]
        for enabled, square in zip(
            state[1:5], (chess.H1, chess.A1, chess.H8, chess.A8), strict=True
        )
        if enabled
    )
    board.ep_square = None if state[5] == 64 else state[5]
    board.halfmove_clock = state[6]
    board.fullmove_number = int(row["ply"]) // 2 + 1
    return board


def load_openings(path: Path, *, opening_ply: int, count: int) -> list[tuple[str, chess.Board]]:
    """Load one frozen validation position per game at an exact ply."""
    table = pq.read_table(path, columns=["game_id", "ply", "squares", "state"])
    openings: list[tuple[str, chess.Board]] = []
    seen: set[str] = set()
    for row in table.to_pylist():
        game_id = str(row["game_id"])
        if int(row["ply"]) != opening_ply or game_id in seen:
            continue
        board = _board_from_row(row)
        if not board.is_valid() or board.is_game_over():
            continue
        seen.add(game_id)
        openings.append((game_id, board))
        if len(openings) == count:
            break
    if len(openings) != count:
        raise ValueError(f"requested {count} openings at ply {opening_ply}, found {len(openings)}")
    return openings


def load_model(path: Path, device: torch.device) -> tuple[SnapshotPolicy, dict[str, Any]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("model_type") != "board_snapshot_policy":
        raise ValueError(f"{path} is not a board snapshot policy")
    model = SnapshotPolicy(ModelConfig(**raw["model_config"]))
    model.load_state_dict(raw["state_dict"])
    model.eval()
    metadata = {
        "architecture": raw["model_config"]["architecture"],
        "experiment_id": raw["train_config"]["experiment_id"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model.to(device), metadata


def choose_move(model: SnapshotPolicy, board: chess.Board, device: torch.device) -> chess.Move:
    snapshot = encode_board(board)
    with torch.no_grad():
        logits = model(
            torch.tensor([snapshot.squares], dtype=torch.long, device=device),
            torch.tensor([snapshot.state], dtype=torch.long, device=device),
            torch.tensor([snapshot.phase], dtype=torch.long, device=device),
        )[0]
    legal = list(board.legal_moves)
    legal_indices = torch.tensor(
        [move_index(move) for move in legal], dtype=torch.long, device=device
    )
    return legal[int(logits.index_select(0, legal_indices).argmax().item())]


def play_game(
    board: chess.Board,
    *,
    white: SnapshotPolicy,
    black: SnapshotPolicy,
    device: torch.device,
    max_plies: int,
) -> tuple[chess.Color | None, int, str]:
    """Return winner, played plies, and termination; max-length games are draws."""
    played = 0
    while played < max_plies:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return outcome.winner, played, outcome.termination.name.lower()
        board.push(choose_move(white if board.turn else black, board, device))
        played += 1
    return None, played, "max_plies"


def paired_match(
    *,
    checkpoint_a: Path,
    checkpoint_b: Path,
    validation: Path,
    opening_ply: int,
    opening_count: int,
    max_game_plies: int,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(device_name)
    model_a, metadata_a = load_model(checkpoint_a, device)
    model_b, metadata_b = load_model(checkpoint_b, device)
    openings = load_openings(validation, opening_ply=opening_ply, count=opening_count)
    a_wins = b_wins = draws = total_plies = 0
    terminations: dict[str, int] = {}
    games: list[dict[str, Any]] = []
    for game_id, opening in openings:
        for a_color in (chess.WHITE, chess.BLACK):
            winner, plies, termination = play_game(
                opening.copy(),
                white=model_a if a_color else model_b,
                black=model_b if a_color else model_a,
                device=device,
                max_plies=max_game_plies,
            )
            if winner is None:
                draws += 1
                result = "draw"
            elif winner == a_color:
                a_wins += 1
                result = "a_win"
            else:
                b_wins += 1
                result = "b_win"
            total_plies += plies
            terminations[termination] = terminations.get(termination, 0) + 1
            games.append(
                {
                    "opening_id_sha256": hashlib.sha256(game_id.encode()).hexdigest(),
                    "model_a_color": "white" if a_color else "black",
                    "plies": plies,
                    "result": result,
                    "termination": termination,
                }
            )
    game_count = len(games)
    return {
        "schema": "chess-gpt-paired-match-v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(),
        "code_worktree_dirty": _git_worktree_dirty(),
        "evaluator_source": {
            "path": "src/chess_gpt/snapshot_match.py",
            "sha256": _sha256(Path(__file__)),
        },
        "environment_lock_sha256": _sha256(Path("uv.lock")),
        "checkpoint_a": {
            "path": checkpoint_a.as_posix(),
            "sha256": _sha256(checkpoint_a),
            **metadata_a,
        },
        "checkpoint_b": {
            "path": checkpoint_b.as_posix(),
            "sha256": _sha256(checkpoint_b),
            **metadata_b,
        },
        "validation": {"path": validation.as_posix(), "sha256": _sha256(validation)},
        "method": {
            "device": device_name,
            "opening_count": opening_count,
            "opening_ply": opening_ply,
            "paired_color_reversal": True,
            "max_game_plies_after_opening": max_game_plies,
            "opening_history": (
                "position-only validation snapshot; repetition history before opening ply "
                "is unavailable"
            ),
            "selection": "first distinct valid games in frozen validation shard",
            "test_split_used": False,
            "seed": None,
        },
        "hardware": {
            "device": device_name,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "result": {
            "games": game_count,
            "model_a_wins": a_wins,
            "model_b_wins": b_wins,
            "draws": draws,
            "model_a_score": a_wins + 0.5 * draws,
            "model_b_score": b_wins + 0.5 * draws,
            "model_a_score_rate": (a_wins + 0.5 * draws) / game_count,
            "total_plies": total_plies,
            "terminations": terminations,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "games": games,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opening-ply", type=int, default=12)
    parser.add_argument("--opening-count", type=int, default=50)
    parser.add_argument("--max-game-plies", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = (
        "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    result = paired_match(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        validation=args.validation,
        opening_ply=args.opening_ply,
        opening_count=args.opening_count,
        max_game_plies=args.max_game_plies,
        device_name=device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
