"""Prepare enriched lab shards from a raw Lichess PGN archive, with data-policy filters."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
from pathlib import Path

import chess.pgn
import chess.polyglot
import pyarrow as pa
import pyarrow.parquet as pq

from chess_gpt.snapshot_model import encode_board, move_index

RESULTS = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
PIECE_VALUES = [0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0]
HISTORY = 8

SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.uint16()),
        pa.field("squares", pa.list_(pa.uint8(), 64)),
        pa.field("state", pa.list_(pa.uint8(), 7)),
        pa.field("target", pa.uint16()),
        pa.field("result", pa.uint8()),
        pa.field("history_from", pa.list_(pa.uint8(), HISTORY)),
        pa.field("history_to", pa.list_(pa.uint8(), HISTORY)),
        pa.field("repetition", pa.uint8()),
        pa.field("plies_remaining", pa.uint16()),
        pa.field("future_material", pa.int8()),
        pa.field("white_elo", pa.uint16()),
        pa.field("black_elo", pa.uint16()),
    ]
)


def rows_from_moves(
    moves: list[chess.Move],
    game_id: str,
    result: int,
    keep_color: chess.Color | None,
    global_counts: dict[int, int] | None,
    dedup_cap: int,
    elos: tuple[int, int] = (0, 0),
) -> list[dict[str, object]] | None:
    """Replay a finished game into enriched training rows.

    keep_color limits target rows to one side's moves; global_counts (if given)
    caps how often the exact same position may appear across the whole shard.
    """
    board = chess.Board()
    snapshots = []
    balances = []
    seen: dict[int, int] = {}
    for move in moves:
        if move not in board.legal_moves:
            return None
        snapshot = encode_board(board)
        key = chess.polyglot.zobrist_hash(board)
        balances.append(sum(PIECE_VALUES[code] for code in snapshot.squares))
        snapshots.append((snapshot, move, board.turn, seen.get(key, 0), key))
        seen[key] = seen.get(key, 0) + 1
        board.push(move)
    balances.append(sum(PIECE_VALUES[code] for code in encode_board(board).squares))

    total = len(moves)
    rows: list[dict[str, object]] = []
    for ply, (snapshot, move, turn, repetition, key) in enumerate(snapshots):
        if keep_color is not None and turn != keep_color:
            continue
        if global_counts is not None:
            if global_counts.get(key, 0) >= dedup_cap:
                continue
            global_counts[key] = global_counts.get(key, 0) + 1
        window = moves[max(0, ply - HISTORY) : ply]
        pad = HISTORY - len(window)
        rows.append(
            {
                "game_id": game_id,
                "ply": ply,
                "squares": snapshot.squares,
                "state": snapshot.state,
                "target": move_index(move),
                "result": result,
                "history_from": [64] * pad + [m.from_square for m in window],
                "history_to": [64] * pad + [m.to_square for m in window],
                "repetition": min(repetition, 3),
                "plies_remaining": total - ply,
                "future_material": max(-127, min(127, balances[min(ply + 6, total)])),
                "white_elo": elos[0],
                "black_elo": elos[1],
            }
        )
    return rows or None


def _elo(game: chess.pgn.Game, header: str) -> int | None:
    try:
        return int(game.headers[header])
    except (KeyError, ValueError):
        return None


def _base_seconds(game: chess.pgn.Game) -> int | None:
    control = game.headers.get("TimeControl", "")
    if "+" not in control:
        return None
    try:
        return int(control.split("+")[0])
    except ValueError:
        return None


def accept(game: chess.pgn.Game, result: int, args: argparse.Namespace) -> chess.Color | None | str:
    """Return 'reject', or the keep_color for rows (None = both sides)."""
    if args.draws_only and result != 1:
        return "reject"
    if args.decisive_only and result == 1:
        return "reject"
    if args.no_bullet and ((_base_seconds(game) or 0) < 180):
        return "reject"
    if args.both_min_elo is not None:
        white, black = _elo(game, "WhiteElo"), _elo(game, "BlackElo")
        if white is None or black is None or min(white, black) <= args.both_min_elo:
            return "reject"
    if args.min_winner_elo is not None:
        if result == 1:
            return "reject"
        winner = chess.WHITE if result == 0 else chess.BLACK
        elo = _elo(game, "WhiteElo" if winner == chess.WHITE else "BlackElo")
        if elo is None or elo <= args.min_winner_elo:
            return "reject"
        return winner
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--skip-games", type=int, default=0, help="Discard this many games from the archive first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-winner-elo", type=int)
    parser.add_argument("--both-min-elo", type=int)
    parser.add_argument("--decisive-only", action="store_true")
    parser.add_argument("--draws-only", action="store_true")
    parser.add_argument("--no-bullet", action="store_true")
    parser.add_argument("--dedup-cap", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(["zstd", "-dc", str(args.source)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace")
    global_counts: dict[int, int] | None = {} if args.dedup_cap else None

    kept = read = positions = 0
    buffered: list[dict[str, object]] = []
    with pq.ParquetWriter(args.output, SCHEMA, compression="zstd") as writer:
        while kept < args.games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            read += 1
            if read <= args.skip_games:
                continue
            result = RESULTS.get(game.headers.get("Result", ""))
            if result is None or game.errors:
                continue
            decision = accept(game, result, args)
            if decision == "reject":
                continue
            rows = rows_from_moves(
                list(game.mainline_moves()),
                game.headers.get("Site") or f"game-{read}",
                result,
                decision,
                global_counts,
                args.dedup_cap,
                elos=(
                    min(65535, max(0, _elo(game, "WhiteElo") or 0)),
                    min(65535, max(0, _elo(game, "BlackElo") or 0)),
                ),
            )
            if rows is None:
                continue
            kept += 1
            positions += len(rows)
            buffered.extend(rows)
            if len(buffered) >= 65_536:
                writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))
    process.terminate()

    print(
        json.dumps(
            {
                "read": read,
                "kept": kept,
                "positions": positions,
                "output": str(args.output),
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
