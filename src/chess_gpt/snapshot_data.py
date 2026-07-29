"""Prepare frozen Lichess PGNs as reusable board-snapshot Parquet shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from chess_gpt.snapshot_model import encode_board, move_index

POSITION_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.uint16()),
        pa.field("squares", pa.list_(pa.uint8(), 64)),
        pa.field("state", pa.list_(pa.uint8(), 7)),
        pa.field("phase", pa.uint8()),
        pa.field("target", pa.uint16()),
    ]
)


@dataclass(frozen=True)
class PreparationResult:
    games: int
    selected_games: int
    positions: int
    filtered_games: int
    invalid_games: int


@dataclass(frozen=True)
class FrozenFile:
    month: str
    split: str
    url: str
    sha256: str


def load_frozen_files(manifest: Path) -> list[FrozenFile]:
    """Load the authoritative download list; callers never invent dataset URLs."""
    with manifest.open("rb") as stream:
        raw = tomllib.load(stream)
    if raw["dataset"]["status"] != "frozen":
        raise ValueError("dataset manifest is not frozen")
    return [FrozenFile(**item) for item in raw["files"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_frozen_file(item: FrozenFile, cache_root: Path) -> Path:
    """Resume, verify, and atomically expose one frozen monthly archive."""
    raw_dir = cache_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / Path(item.url).name
    if destination.exists():
        actual = _sha256(destination)
        if actual != item.sha256:
            raise ValueError(f"cached {destination} has SHA-256 {actual}, expected {item.sha256}")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--continue-at", "-", "--output", str(partial), item.url],
        check=True,
    )
    actual = _sha256(partial)
    if actual != item.sha256:
        raise ValueError(f"downloaded {partial} has SHA-256 {actual}, expected {item.sha256}")
    partial.replace(destination)
    return destination


@contextmanager
def _open_pgn(path: Path) -> Iterator[TextIO]:
    if path.suffix != ".zst":
        with path.open(encoding="utf-8", errors="replace") as stream:
            yield stream
        return

    process = subprocess.Popen(
        ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.stdout is None:
        raise RuntimeError("zstd did not expose its output stream")
    stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace")
    try:
        yield stream
    finally:
        stopped_before_process_exit = process.poll() is None
        if stopped_before_process_exit:
            process.terminate()
        stream.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0 and not stopped_before_process_exit:
            raise RuntimeError(f"zstd failed with exit code {return_code}: {stderr.strip()}")


def _winner(game: chess.pgn.Game) -> chess.Color | None:
    result = game.headers.get("Result")
    if result == "1-0":
        return chess.WHITE
    if result == "0-1":
        return chess.BLACK
    return None


def _winner_elo(game: chess.pgn.Game, winner: chess.Color) -> int | None:
    header = "WhiteElo" if winner == chess.WHITE else "BlackElo"
    try:
        return int(game.headers[header])
    except (KeyError, TypeError, ValueError):
        return None


def _game_rows(
    game: chess.pgn.Game,
    game_number: int,
    *,
    target_color: chess.Color | None = None,
) -> list[dict[str, object]]:
    if game.errors:
        raise ValueError(f"PGN parser reported {len(game.errors)} error(s)")
    board = game.board()
    game_id = game.headers.get("Site") or f"game-{game_number}"
    rows: list[dict[str, object]] = []
    for ply, move in enumerate(game.mainline_moves()):
        if move not in board.legal_moves:
            raise ValueError(f"illegal move at ply {ply + 1}")
        if target_color is None or board.turn == target_color:
            snapshot = encode_board(board)
            rows.append(
                {
                    "game_id": game_id,
                    "ply": ply,
                    "squares": snapshot.squares,
                    "state": snapshot.state,
                    "phase": snapshot.phase,
                    "target": move_index(move),
                }
            )
        board.push(move)
    return rows


def prepare_pgn(
    source: Path,
    output: Path,
    *,
    split: str,
    max_games: int | None = None,
    max_selected_games: int | None = None,
    winner_only: bool = False,
    min_winner_elo: int | None = None,
    row_group_positions: int = 65_536,
    source_sha256: str | None = None,
) -> PreparationResult:
    """Stream a PGN or PGN.zst into a compressed, training-ready position shard."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    if max_selected_games is not None and max_selected_games < 1:
        raise ValueError("max_selected_games must be positive")
    if min_winner_elo is not None and min_winner_elo < 0:
        raise ValueError("min_winner_elo cannot be negative")
    if min_winner_elo is not None and not winner_only:
        raise ValueError("min_winner_elo requires winner_only")
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared_format = "board-snapshot-winner-v1" if winner_only else "board-snapshot-v1"
    metadata = {
        b"prepared_format": prepared_format.encode(),
        b"split": split.encode(),
        b"source_file": source.name.encode(),
        b"target_side": b"winner" if winner_only else b"both",
    }
    if min_winner_elo is not None:
        metadata[b"min_winner_elo"] = str(min_winner_elo).encode()
    if source_sha256 is not None:
        metadata[b"source_sha256"] = source_sha256.encode()
    schema = POSITION_SCHEMA.with_metadata(metadata)
    games = selected_games = positions = filtered_games = invalid_games = 0
    buffered: list[dict[str, object]] = []

    with _open_pgn(source) as stream, pq.ParquetWriter(
        output,
        schema,
        compression="zstd",
        use_dictionary=True,
    ) as writer:
        while (max_games is None or games < max_games) and (
            max_selected_games is None or selected_games < max_selected_games
        ):
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            winner = _winner(game)
            if winner_only and (
                winner is None
                or (
                    min_winner_elo is not None
                    and (_winner_elo(game, winner) or -1) < min_winner_elo
                )
            ):
                filtered_games += 1
                continue
            try:
                rows = _game_rows(
                    game,
                    games,
                    target_color=winner if winner_only else None,
                )
            except ValueError:
                invalid_games += 1
                continue
            selected_games += 1
            buffered.extend(rows)
            positions += len(rows)
            if len(buffered) >= row_group_positions:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))

    return PreparationResult(
        games=games,
        selected_games=selected_games,
        positions=positions,
        filtered_games=filtered_games,
        invalid_games=invalid_games,
    )


def _select_months(files: list[FrozenFile], requested: Sequence[str]) -> list[FrozenFile]:
    if requested == ["all"]:
        return files
    by_month = {item.month: item for item in files}
    missing = set(requested) - by_month.keys()
    if missing:
        raise ValueError(f"months are not frozen in the manifest: {sorted(missing)}")
    return [by_month[month] for month in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "prepare", "fetch-prepare"))
    parser.add_argument("--month", action="append", dest="months", default=[])
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset.toml"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path("data/downloads/tournament-2026")
    )
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-selected-games", type=int)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--min-winner-elo", type=int)
    parser.add_argument(
        "--discard-raw",
        action="store_true",
        help="After successful preparation, delete that verified monthly archive to save disk",
    )
    args = parser.parse_args()
    requested = args.months or ["all"]
    selected = _select_months(load_frozen_files(args.manifest), requested)
    results: list[dict[str, object]] = []
    for item in selected:
        raw_path = args.cache_root / "raw" / Path(item.url).name
        raw_verified = False
        if args.command in {"fetch", "fetch-prepare"}:
            raw_path = fetch_frozen_file(item, args.cache_root)
            raw_verified = True
        result: dict[str, object] = {
            "month": item.month,
            "split": item.split,
            "raw_path": raw_path.as_posix(),
        }
        if args.command in {"prepare", "fetch-prepare"}:
            if not raw_path.is_file():
                raise FileNotFoundError(f"fetch {item.month} before preparing it: {raw_path}")
            if not raw_verified:
                actual = _sha256(raw_path)
                if actual != item.sha256:
                    raise ValueError(
                        f"cached {raw_path} has SHA-256 {actual}, expected {item.sha256}"
                    )
            prepared_directory = (
                f"board-snapshot-winner-elo{args.min_winner_elo}-v1"
                if args.winner_only and args.min_winner_elo is not None
                else "board-snapshot-winner-v1"
                if args.winner_only
                else "board-snapshot-v1"
            )
            prepared_path = args.cache_root / "prepared" / prepared_directory / (
                f"{item.month}.parquet"
            )
            prepared = prepare_pgn(
                raw_path,
                prepared_path,
                split=item.split,
                max_games=args.max_games,
                max_selected_games=args.max_selected_games,
                winner_only=args.winner_only,
                min_winner_elo=args.min_winner_elo,
                source_sha256=item.sha256,
            )
            result.update(
                {
                    "prepared_path": prepared_path.as_posix(),
                    "prepared_sha256": _sha256(prepared_path),
                    "games": prepared.games,
                    "selected_games": prepared.selected_games,
                    "positions": prepared.positions,
                    "filtered_games": prepared.filtered_games,
                    "invalid_games": prepared.invalid_games,
                }
            )
            if args.discard_raw:
                raw_path.unlink()
                result["raw_discarded"] = True
            sidecar = prepared_path.with_suffix(".manifest.json")
            sidecar.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
