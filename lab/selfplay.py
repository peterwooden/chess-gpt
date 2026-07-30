"""Generate self-play games from a checkpoint and write them as an enriched lab shard."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq

from lab.match import make_player, random_opening
from lab.prepare import RESULTS, SCHEMA, rows_from_moves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--search", choices=("none", "value1"), default="none")
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.seed)
    player = make_player(
        args.checkpoint, temperature=args.temperature, search=args.search, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    kept = decisive = 0
    buffered: list[dict[str, object]] = []
    with pq.ParquetWriter(args.output, SCHEMA, compression="zstd") as writer:
        for number in range(args.games):
            board = chess.Board()
            for move in random_opening(rng, args.opening_plies):
                board.push(move)
            while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                board.push(player(board))
            result = RESULTS.get(board.result(claim_draw=True), 1)  # ply-cap games count as draws
            if args.winner_only and result == 1:
                continue
            decisive += int(result != 1)
            keep_color = None
            if args.winner_only:
                keep_color = chess.WHITE if result == 0 else chess.BLACK
            rows = rows_from_moves(
                list(board.move_stack), f"selfplay-{number}", result, keep_color, None, 0
            )
            if rows is None:
                continue
            rows = [r for r in rows if int(r["ply"]) >= args.opening_plies]  # never imitate the random opening
            kept += 1
            buffered.extend(rows)
            if len(buffered) >= 65_536:
                writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))

    print(
        json.dumps(
            {
                "games_played": args.games,
                "games_kept": kept,
                "decisive": decisive,
                "output": str(args.output),
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
