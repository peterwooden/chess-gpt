"""Load a small, game-aware slice of a prepared board-snapshot shard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Slice:
    """A slice of positions with enough structure to split honestly or dishonestly."""

    squares: np.ndarray  # (N, 64) uint8 piece codes
    state: np.ndarray  # (N, 7) uint8: turn, 4 castling rights, en-passant file, halfmove clock
    target: np.ndarray  # (N,) int64 move-class index
    game_ordinal: np.ndarray  # (N,) int32 dense id of the source game
    result: np.ndarray | None  # (N,) int64 game outcome (0 white wins, 1 draw, 2 black wins)
    extras: dict[str, np.ndarray]  # enrichment columns present in the shard, if any

    @property
    def positions(self) -> int:
        return len(self.target)

    @property
    def games(self) -> int:
        return int(self.game_ordinal.max()) + 1


def load_slice(path: Path, max_games: int) -> Slice:
    """Read the first max_games complete games from a prepared Parquet shard."""
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    has_result = "result" in names
    list_extras = [c for c in ("history_from", "history_to") if c in names]
    flat_extras = [c for c in ("repetition", "plies_remaining", "future_material") if c in names]
    columns = (
        ["game_id", "squares", "state", "target"]
        + (["result"] if has_result else [])
        + list_extras
        + flat_extras
    )
    chunks: list[dict[str, np.ndarray]] = []
    seen: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=columns):
        game_ids = batch["game_id"].to_pylist()
        ordinals = np.empty(len(game_ids), dtype=np.int32)
        for row, game_id in enumerate(game_ids):
            if game_id not in seen:
                seen[game_id] = len(seen)
            ordinals[row] = seen[game_id]
        keep = ordinals < max_games
        chunk = {
            "squares": np.stack(batch["squares"].to_numpy(zero_copy_only=False))[keep],
            "state": np.stack(batch["state"].to_numpy(zero_copy_only=False))[keep],
            "target": batch["target"].to_numpy().astype(np.int64)[keep],
            "game_ordinal": ordinals[keep],
        }
        if has_result:
            chunk["result"] = batch["result"].to_numpy().astype(np.int64)[keep]
        for name in list_extras:
            chunk[name] = np.stack(batch[name].to_numpy(zero_copy_only=False))[keep].astype(np.int64)
        for name in flat_extras:
            chunk[name] = batch[name].to_numpy().astype(np.int64)[keep]
        chunks.append(chunk)
        if len(seen) > max_games:
            break
    return Slice(
        squares=np.concatenate([c["squares"] for c in chunks]),
        state=np.concatenate([c["state"] for c in chunks]),
        target=np.concatenate([c["target"] for c in chunks]),
        game_ordinal=np.concatenate([c["game_ordinal"] for c in chunks]),
        result=np.concatenate([c["result"] for c in chunks]) if has_result else None,
        extras={
            name: np.concatenate([c[name] for c in chunks])
            for name in list_extras + flat_extras
        },
    )


def validation_mask(data: Slice, policy: str, fraction: float, seed: int) -> np.ndarray:
    """Return a boolean mask marking validation positions under the chosen policy.

    "position" shuffles individual positions, so near-duplicate neighbours from the
    same game land on both sides of the fence. "game" assigns whole games.
    """
    rng = np.random.default_rng(seed)
    if policy == "position":
        return rng.random(data.positions) < fraction
    if policy == "game":
        validation_games = rng.random(data.games) < fraction
        return validation_games[data.game_ordinal]
    raise ValueError(f"unknown split policy: {policy}")
