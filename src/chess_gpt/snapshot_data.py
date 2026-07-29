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
    positions: int
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


def _game_rows(game: chess.pgn.Game, game_number: int) -> list[dict[str, object]]:
    if game.errors:
        raise ValueError(f"PGN parser reported {len(game.errors)} error(s)")
    board = game.board()
    game_id = game.headers.get("Site") or f"game-{game_number}"
    rows: list[dict[str, object]] = []
    for ply, move in enumerate(game.mainline_moves()):
        if move not in board.legal_moves:
            raise ValueError(f"illegal move at ply {ply + 1}")
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
    row_group_positions: int = 65_536,
    source_sha256: str | None = None,
) -> PreparationResult:
    """Stream a PGN or PGN.zst into a compressed, training-ready position shard."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"prepared_format": b"board-snapshot-v1",
        b"split": split.encode(),
        b"source_file": source.name.encode(),
    }
    if source_sha256 is not None:
        metadata[b"source_sha256"] = source_sha256.encode()
    schema = POSITION_SCHEMA.with_metadata(metadata)
    games = positions = invalid_games = 0
    buffered: list[dict[str, object]] = []

    with _open_pgn(source) as stream, pq.ParquetWriter(
        output,
        schema,
        compression="zstd",
        use_dictionary=True,
    ) as writer:
        while max_games is None or games < max_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games += 1
            try:
                rows = _game_rows(game, games)
            except ValueError:
                invalid_games += 1
                continue
            buffered.extend(rows)
            positions += len(rows)
            if len(buffered) >= row_group_positions:
                writer.write_table(pa.Table.from_pylist(buffered, schema=schema))
                buffered.clear()
        if buffered:
            writer.write_table(pa.Table.from_pylist(buffered, schema=schema))

    return PreparationResult(games=games, positions=positions, invalid_games=invalid_games)


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
        if args.command in {"fetch", "fetch-prepare"}:
            raw_path = fetch_frozen_file(item, args.cache_root)
        result: dict[str, object] = {
            "month": item.month,
            "split": item.split,
            "raw_path": raw_path.as_posix(),
        }
        if args.command in {"prepare", "fetch-prepare"}:
            if not raw_path.is_file():
                raise FileNotFoundError(f"fetch {item.month} before preparing it: {raw_path}")
            prepared_path = (
                args.cache_root / "prepared" / "board-snapshot-v1" / f"{item.month}.parquet"
            )
            prepared = prepare_pgn(
                raw_path,
                prepared_path,
                split=item.split,
                max_games=args.max_games,
                source_sha256=item.sha256,
            )
            result.update(
                {
                    "prepared_path": prepared_path.as_posix(),
                    "prepared_sha256": _sha256(prepared_path),
                    "games": prepared.games,
                    "positions": prepared.positions,
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
