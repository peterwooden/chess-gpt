"""Opening-play experiments: corpus book, conformity metric, and opening-aware players."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import chess
import chess.polyglot
import numpy as np
import pyarrow.parquet as pq
import torch

from chess_gpt.snapshot_model import PROMOTION_UCI_MOVES, encode_board, move_index
from lab.match import Player, _inputs, _white_score, load_model, make_player, play_series

BOOK_PATH = Path("runs/lab/opening-book.pkl")
BOOK_PLIES = 12


def move_from_index(index: int, board: chess.Board) -> chess.Move:
    if index >= 4096:
        return chess.Move.from_uci(PROMOTION_UCI_MOVES[index - 4096])
    move = chess.Move(index // 64, index % 64)
    # Re-attach promotion metadata if this from/to is actually a pawn promotion push
    for legal in board.legal_moves:
        if legal.from_square == move.from_square and legal.to_square == move.to_square:
            return legal
    raise ValueError(f"index {index} is not legal here")


def build(shard: Path, games: int) -> None:
    parquet = pq.ParquetFile(shard)
    counts: dict[int, Counter] = {}
    paths: dict[int, list[str]] = {}
    seen_games: set[str] = set()
    replayed = 0
    board = chess.Board()
    current_game: str | None = None
    for batch in parquet.iter_batches(columns=["game_id", "ply", "target"]):
        ids = batch["game_id"].to_pylist()
        plies = batch["ply"].to_numpy()
        targets = batch["target"].to_numpy()
        for gid, ply, target in zip(ids, plies, targets):
            if gid != current_game:
                if replayed >= games:
                    break
                current_game = gid
                seen_games.add(gid)
                replayed += 1
                board = chess.Board()
            if ply >= BOOK_PLIES or board.ply() != ply:
                continue
            key = chess.polyglot.zobrist_hash(board)
            move = move_from_index(int(target), board)
            counts.setdefault(key, Counter())[move.uci()] += 1
            if key not in paths:
                paths[key] = [m.uci() for m in board.move_stack]
            board.push(move)
        else:
            continue
        break
    book = {}
    for key, counter in counts.items():
        total = sum(counter.values())
        if total < 20:
            continue
        ranked = counter.most_common(3)
        if ranked[0][1] / total < 0.15:
            continue
        book[key] = {"top": [uci for uci, _ in ranked], "total": total, "path": paths[key]}
    BOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOOK_PATH.write_bytes(pickle.dumps(book))
    print(json.dumps({"games": replayed, "book_positions": len(book)}))


def load_book() -> dict:
    return pickle.loads(BOOK_PATH.read_bytes())


def opening_player(base: Player, checkpoint: Path, mode: str) -> Player:
    """Wrap a search player with an opening policy. Modes: book, flat, policy."""
    book = load_book() if mode == "book" else None
    model = load_model(checkpoint) if mode in ("flat", "policy") else None

    def choose(board: chess.Board) -> chess.Move:
        if board.ply() < BOOK_PLIES and mode == "book" and book is not None:
            entry = book.get(chess.polyglot.zobrist_hash(board))
            if entry is not None:
                move = chess.Move.from_uci(entry["top"][0])
                if move in board.legal_moves:
                    return move
        if board.ply() < 16 and mode in ("flat", "policy") and model is not None:
            legal = list(board.legal_moves)
            with torch.no_grad():
                out = model(**_inputs(model, board))
            policy = out["policy"][0]
            indices = torch.tensor([move_index(m) for m in legal])
            if mode == "policy":
                return legal[int(policy[indices].argmax())]
            # flat mode: value-screen all successors; fall back to policy when flat
            values = []
            for move in legal:
                board.push(move)
                snapshot_inputs = _inputs(model, board)
                board.pop()
                values.append(snapshot_inputs)
            stacked = {k: torch.cat([v[k] for v in values]) for k in values[0]}
            with torch.no_grad():
                white = _white_score(model(**stacked)["value"])
            mine = white if board.turn == chess.WHITE else 1 - white
            if float(mine.max() - mine.min()) < 0.04:
                return legal[int(policy[indices].argmax())]
        return base(board)

    return choose


def conformity(player: Player, probes: int, seed: int) -> float:
    book = load_book()
    popular = sorted(book.items(), key=lambda kv: -kv[1]["total"])[:probes]
    hits = 0
    for _, entry in popular:
        board = chess.Board()
        for uci in entry["path"]:
            board.push(chess.Move.from_uci(uci))
        hits += int(player(board).uci() in entry["top"])
    return hits / len(popular)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "evaluate"))
    parser.add_argument("--shard", type=Path, default=Path("data/downloads/lab/enriched-240k.parquet"))
    parser.add_argument("--games", type=int, default=120000)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/lab/fleet/57-cloud.pt"))
    parser.add_argument("--mode", choices=("base", "book", "flat", "policy"), default="base")
    parser.add_argument("--match-games", type=int, default=20)
    parser.add_argument("--probes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.command == "build":
        build(args.shard, args.games)
        return
    base = make_player(args.checkpoint, search="beam", contempt=0.15)
    player = base if args.mode == "base" else opening_player(base, args.checkpoint, args.mode)
    score = conformity(player, args.probes, args.seed)
    result = {"mode": args.mode, "conformity_top3": round(score, 3)}
    if args.mode != "base":
        opponent = make_player(args.checkpoint, search="beam", contempt=0.15)
        tally = play_series(player, opponent, args.match_games, random.Random(args.seed))
        result.update(tally)
        result["score"] = tally["win"] + 0.5 * tally["draw"]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
