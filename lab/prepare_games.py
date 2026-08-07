"""Prepare compact game-record shards (one row per game) from a raw Lichess PGN archive.

Filters match the cancelled materialized-position prep (lab/prepare.py with
--both-min-elo 1600 --no-bullet): rated standard games, both Elo > 1600,
base time >= 180 s. Instead of materializing per-position rows, each game is
stored as a list of uint16 move words:

    bits 0-5 from-square, 6-11 to-square, 12-14 promotion
    (0 none, 1 knight, 2 bishop, 3 rook, 4 queen), bit 15 zero.

Square indexing matches python-chess (0=a1 .. 63=h8). Castling is the king's
actual from->to; en passant is the pawn's diagonal move; no flags for either.

result matches lab/prepare.py's pre-flip convention: 0 = white won ("1-0"),
1 = draw, 2 = black won ("0-1").
termination: 0 Normal, 1 Time forfeit, 2 Abandoned, 3 anything else.

The `verify` subcommand independently re-parses raw PGN text captured during
prep for a random sample of games, replays the SAN from scratch, and checks
the stored move words and headers against it.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import subprocess
import time
from pathlib import Path

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

RESULTS = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}  # identical to lab/prepare.py
TERMINATIONS = {"Normal": 0, "Time forfeit": 1, "Abandoned": 2}
TERMINATION_OTHER = 3
ROW_GROUP_GAMES = 100_000

SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.binary(8)),
        pa.field("white_elo", pa.uint16()),
        pa.field("black_elo", pa.uint16()),
        pa.field("result", pa.int8()),
        pa.field("time_base_s", pa.uint16()),
        pa.field("time_inc_s", pa.uint8()),
        pa.field("termination", pa.int8()),
        pa.field("ply_count", pa.uint16()),
        pa.field("moves", pa.list_(pa.uint16())),
    ]
)


class RecordingStream:
    """Wrap a text stream, keeping the raw lines read since the last reset."""

    def __init__(self, stream: io.TextIOBase) -> None:
        self._stream = stream
        self.lines: list[str] = []

    def readline(self, *args) -> str:
        line = self._stream.readline(*args)
        self.lines.append(line)
        return line

    def reset(self) -> None:
        self.lines = []

    def text(self) -> str:
        return "".join(self.lines)


def _elo(game: chess.pgn.Game, header: str) -> int | None:
    try:
        return int(game.headers[header])
    except (KeyError, ValueError):
        return None


def _time_control(game: chess.pgn.Game) -> tuple[int, int] | None:
    """(base_s, inc_s), or None if unparseable (matches prepare.py's base parse)."""
    control = game.headers.get("TimeControl", "")
    if "+" not in control:
        return None
    base_text, _, inc_text = control.partition("+")
    try:
        return int(base_text), int(inc_text)
    except ValueError:
        return None


def encode_moves(moves: list[chess.Move]) -> list[int]:
    words = []
    for move in moves:
        promotion = (move.promotion - 1) if move.promotion else 0  # KNIGHT(2)->1 .. QUEEN(5)->4
        words.append(move.from_square | (move.to_square << 6) | (promotion << 12))
    return words


def game_row(game: chess.pgn.Game, result: int) -> dict[str, object] | None:
    site = game.headers.get("Site", "")
    game_id = site.rsplit("/", 1)[-1]
    if len(game_id) != 8 or not game_id.isascii():
        return None
    moves = list(game.mainline_moves())
    if not moves or len(moves) > 65535:
        return None
    control = _time_control(game)
    assert control is not None  # the no-bullet filter already rejected these
    base_s, inc_s = control
    return {
        "game_id": game_id.encode("ascii"),
        "white_elo": min(65535, max(0, _elo(game, "WhiteElo") or 0)),
        "black_elo": min(65535, max(0, _elo(game, "BlackElo") or 0)),
        "result": result,
        "time_base_s": min(65535, base_s),
        "time_inc_s": min(255, max(0, inc_s)),
        "termination": TERMINATIONS.get(game.headers.get("Termination", ""), TERMINATION_OTHER),
        "ply_count": len(moves),
        "moves": encode_moves(moves),
    }


def accept(game: chess.pgn.Game) -> bool:
    """Rated standard, both Elo > 1600, base >= 180 s (as lab/prepare.py)."""
    control = _time_control(game)
    if control is None or control[0] < 180:
        return False
    white, black = _elo(game, "WhiteElo"), _elo(game, "BlackElo")
    return white is not None and black is not None and min(white, black) > 1600


def prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    if args.output.exists():
        try:
            existing = pq.read_metadata(args.output).num_rows
        except Exception:
            existing = -1
        if existing == args.games:
            print(json.dumps({"skipped": True, "output": str(args.output), "rows": existing}))
            return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(".tmp.parquet")
    sample_rate = min(1.0, 3.0 * args.sample_target / args.games)
    rng = random.Random(20260730)
    sample_handle = open(args.sample_file, "w") if args.sample_file else None

    process = subprocess.Popen(["zstd", "-dc", str(args.source)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    stream = RecordingStream(io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace"))

    kept = read = positions = 0
    buffered: list[dict[str, object]] = []
    with pq.ParquetWriter(tmp_output, SCHEMA, compression="zstd") as writer:
        while kept < args.games:
            stream.reset()
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            read += 1
            result = RESULTS.get(game.headers.get("Result", ""))
            if result is None or game.errors:
                continue
            if not accept(game):
                continue
            row = game_row(game, result)
            if row is None:
                continue
            kept += 1
            positions += int(row["ply_count"])  # type: ignore[arg-type]
            buffered.append(row)
            if sample_handle and rng.random() < sample_rate:
                json.dump({"game_id": row["game_id"].decode(), "pgn": stream.text()}, sample_handle)
                sample_handle.write("\n")
            if len(buffered) >= ROW_GROUP_GAMES:
                writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=SCHEMA))
    process.terminate()
    if sample_handle:
        sample_handle.close()
    if kept < args.games:
        raise SystemExit(f"archive exhausted: kept {kept} < target {args.games}")
    tmp_output.replace(args.output)

    print(
        json.dumps(
            {
                "read": read,
                "kept": kept,
                "positions": positions,
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


def decode_moves(words: list[int]) -> list[chess.Move]:
    moves = []
    for word in words:
        assert word < 0x8000, f"bit 15 set in move word {word:#06x}"
        promotion = (word >> 12) & 0x7
        moves.append(
            chess.Move(word & 0x3F, (word >> 6) & 0x3F, promotion=promotion + 1 if promotion else None)
        )
    return moves


def verify(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    metadata = pq.read_metadata(args.output)
    assert metadata.num_rows == args.games, f"row count {metadata.num_rows} != target {args.games}"

    samples = [json.loads(line) for line in open(args.sample_file)]
    rng = random.Random(args.seed)
    chosen = {s["game_id"]: s["pgn"] for s in rng.sample(samples, min(args.check_games, len(samples)))}

    wanted = {gid.encode() for gid in chosen}
    rows: dict[bytes, dict] = {}
    parquet = pq.ParquetFile(args.output)
    for batch in parquet.iter_batches(batch_size=65_536):
        ids = batch.column("game_id").to_pylist()
        hits = [i for i, gid in enumerate(ids) if gid in wanted]
        for i in hits:
            rows[ids[i]] = {name: batch.column(name)[i].as_py() for name in batch.schema.names}
    assert len(rows) == len(chosen), f"found {len(rows)} of {len(chosen)} sampled games in parquet"

    checked = 0
    for game_id, pgn in chosen.items():
        row = rows[game_id.encode()]
        game = chess.pgn.read_game(io.StringIO(pgn))
        assert game is not None and not game.errors, f"{game_id}: raw PGN failed to re-parse"

        # Independent SAN replay.
        san_board = chess.Board()
        san_moves = list(game.mainline_moves())
        for move in san_moves:
            assert move in san_board.legal_moves, f"{game_id}: SAN replay illegal move {move}"
            san_board.push(move)

        # Replay the stored u16 words.
        decoded = decode_moves(row["moves"])
        assert len(decoded) == len(san_moves) == row["ply_count"], f"{game_id}: ply count mismatch"
        u16_board = chess.Board()
        for ply, move in enumerate(decoded):
            assert move in u16_board.legal_moves, f"{game_id}: decoded move {ply} {move} illegal"
            expected = san_moves[ply]
            assert (move.from_square, move.to_square, move.promotion) == (
                expected.from_square,
                expected.to_square,
                expected.promotion,
            ), f"{game_id}: ply {ply} decoded {move} != SAN {expected}"
            u16_board.push(move)
        assert u16_board.fen() == san_board.fen(), f"{game_id}: final FEN mismatch"

        # Header spot-checks against the raw PGN.
        headers = game.headers
        assert row["white_elo"] == min(65535, max(0, int(headers["WhiteElo"]))), f"{game_id}: white_elo"
        assert row["black_elo"] == min(65535, max(0, int(headers["BlackElo"]))), f"{game_id}: black_elo"
        assert row["result"] == RESULTS[headers["Result"]], f"{game_id}: result"
        base_text, _, inc_text = headers["TimeControl"].partition("+")
        assert row["time_base_s"] == min(65535, int(base_text)), f"{game_id}: time_base_s"
        assert row["time_inc_s"] == min(255, max(0, int(inc_text))), f"{game_id}: time_inc_s"
        expected_term = TERMINATIONS.get(headers.get("Termination", ""), TERMINATION_OTHER)
        assert row["termination"] == expected_term, f"{game_id}: termination"
        assert int(base_text) >= 180 and min(row["white_elo"], row["black_elo"]) > 1600, f"{game_id}: filter"
        checked += 1

    print(
        json.dumps(
            {
                "verified": str(args.output),
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "games_checked": checked,
                "wall_seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="stream the archive into a compact game-record parquet")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--games", type=int, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--sample-file", type=Path, help="jsonl of raw PGN text for a random sample")
    prep.add_argument("--sample-target", type=int, default=1000)
    prep.set_defaults(func=prepare)

    check = sub.add_parser("verify", help="verify a shard against sampled raw PGN text")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--games", type=int, required=True)
    check.add_argument("--sample-file", type=Path, required=True)
    check.add_argument("--check-games", type=int, default=1000)
    check.add_argument("--seed", type=int, default=20260808)
    check.set_defaults(func=verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
