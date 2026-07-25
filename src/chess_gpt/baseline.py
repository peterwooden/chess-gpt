"""A deliberately small, inspectable SAN n-gram chess baseline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import pyarrow.parquet as pq

ANNOTATION_TOKENS = {"?!", "?", "??"}
CHECK_TOKENS = {"+", "#"}
PROMOTION_TOKENS = {"=Q", "=R", "=B", "=N"}
DATASET_REPO = "shazmate/lichess-chess-tokens"
DATASET_REVISION = "cb90f1bb2eab0b905e84e14f2d1d24ec5f9d1d94"
FORMAT_VERSION = 1
MAX_LEARNED_STATE_BYTES = 100_000_000

Context = tuple[str, ...]
ContextTable = dict[Context, Counter[str]]


@dataclass(frozen=True)
class Prediction:
    san: str
    source: str
    count: int


@dataclass(frozen=True)
class GameRecord:
    site: str
    moves: list[str]


def decode_token_stream(tokens_field: str) -> list[str]:
    """Reassemble ordinary SAN moves from the dataset's factorized move tokens."""
    moves: list[str] = []
    capture = False
    promotion = ""
    check = ""

    for bracketed in tokens_field.split():
        if len(bracketed) < 3 or not bracketed.startswith("[") or not bracketed.endswith("]"):
            raise ValueError(f"malformed dataset token: {bracketed!r}")
        token = bracketed[1:-1]
        if token in ANNOTATION_TOKENS:
            continue
        if token == "x":
            capture = True
            continue
        if token in PROMOTION_TOKENS:
            promotion = token
            continue
        if token in CHECK_TOKENS:
            check = token
            continue

        base = token
        if capture:
            if len(base) < 3:
                raise ValueError(f"cannot insert capture marker into SAN base: {base!r}")
            base = f"{base[:-2]}x{base[-2:]}"
        moves.append(f"{base}{promotion}{check}")
        capture = False
        promotion = ""
        check = ""

    if capture or promotion or check:
        raise ValueError("token stream ended before the final SAN base token")
    return moves


def validate_game(moves: Sequence[str]) -> bool:
    board = chess.Board()
    try:
        for san in moves:
            board.push_san(san)
    except ValueError:
        return False
    return bool(moves)


def iter_game_records(path: Path, max_games: int) -> Iterable[GameRecord]:
    seen = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=1024, columns=["site", "tokens"]):
        for row in batch.to_pylist():
            if seen >= max_games:
                return
            seen += 1
            yield GameRecord(site=str(row["site"]), moves=decode_token_stream(str(row["tokens"])))


def is_validation_game(site: str, seed: int, validation_percent: int) -> bool:
    digest = hashlib.blake2b(f"{seed}:{site}".encode(), digest_size=8).digest()
    return int.from_bytes(digest) % 100 < validation_percent


class SanNgramModel:
    """Back off from recent SAN context to side-to-move frequencies."""

    def __init__(self, order: int = 2, top_moves_per_context: int = 16) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if top_moves_per_context < 1:
            raise ValueError("top_moves_per_context must be at least 1")
        self.order = order
        self.top_moves_per_context = top_moves_per_context
        self.ngrams: dict[int, ContextTable] = {
            n: defaultdict(Counter) for n in range(1, order + 1)
        }
        self.side_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}

    def fit(self, games: Iterable[Sequence[str]]) -> None:
        for moves in games:
            for ply, move in enumerate(moves):
                self.side_counts[ply % 2][move] += 1
                for n in range(1, min(self.order, ply) + 1):
                    context = tuple(moves[ply - n : ply])
                    self.ngrams[n][context][move] += 1

    def prune(self) -> None:
        for table in self.ngrams.values():
            for context, counts in table.items():
                table[context] = Counter(dict(counts.most_common(self.top_moves_per_context)))
        for side, counts in self.side_counts.items():
            self.side_counts[side] = Counter(
                dict(counts.most_common(self.top_moves_per_context * 16))
            )

    @staticmethod
    def _best_legal(counts: Counter[str], legal: set[str]) -> tuple[str, int] | None:
        candidates = ((count, san) for san, count in counts.items() if san in legal)
        best = max(candidates, default=None)
        if best is None:
            return None
        count, san = best
        return san, count

    def predict(self, history: Sequence[str]) -> Prediction:
        board = board_from_san(history)
        legal = {board.san(move) for move in board.legal_moves}
        if not legal:
            raise ValueError("the supplied game is already over")

        for n in range(min(self.order, len(history)), 0, -1):
            context = tuple(history[-n:])
            best = self._best_legal(self.ngrams[n].get(context, Counter()), legal)
            if best is not None:
                san, count = best
                return Prediction(san=san, source=f"{n}-move context", count=count)

        best = self._best_legal(self.side_counts[len(history) % 2], legal)
        if best is not None:
            san, count = best
            return Prediction(san=san, source="side-to-move frequency", count=count)

        return Prediction(san=min(legal), source="deterministic legal fallback", count=0)

    def state_dict(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        ngrams: dict[str, dict[str, list[list[str | int]]]] = {}
        for n, table in self.ngrams.items():
            ngrams[str(n)] = {
                "\t".join(context): [[san, count] for san, count in sorted(counts.items())]
                for context, counts in sorted(table.items())
            }
        return {
            "format_version": FORMAT_VERSION,
            "model_type": "san_backoff_ngram",
            "order": self.order,
            "top_moves_per_context": self.top_moves_per_context,
            "metadata": metadata or {},
            "ngrams": ngrams,
            "side_counts": {
                str(side): [[san, count] for san, count in sorted(counts.items())]
                for side, counts in self.side_counts.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> SanNgramModel:
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported checkpoint format: {state.get('format_version')}")
        model = cls(
            order=int(state["order"]),
            top_moves_per_context=int(state["top_moves_per_context"]),
        )
        model.ngrams = {}
        for n_text, raw_table in state["ngrams"].items():
            table: ContextTable = {}
            for context_text, pairs in raw_table.items():
                context = tuple(context_text.split("\t"))
                table[context] = Counter({str(san): int(count) for san, count in pairs})
            model.ngrams[int(n_text)] = table
        model.side_counts = {
            int(side): Counter({str(san): int(count) for san, count in pairs})
            for side, pairs in state["side_counts"].items()
        }
        return model

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> dict[str, int | str]:
        canonical = json.dumps(
            self.state_dict(metadata), sort_keys=True, separators=(",", ":")
        ).encode()
        if len(canonical) > MAX_LEARNED_STATE_BYTES:
            raise ValueError(
                f"canonical model is {len(canonical):,} bytes, over the 100 MB limit"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=path, mode="wb", compresslevel=9, mtime=0) as file:
            file.write(canonical)
        return {
            "canonical_bytes": len(canonical),
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256_file(path),
        }

    @classmethod
    def load(cls, path: Path) -> SanNgramModel:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            state = json.load(file)
        return cls.from_state_dict(state)


def board_from_san(history: Sequence[str]) -> chess.Board:
    board = chess.Board()
    for ply, san in enumerate(history, start=1):
        try:
            board.push_san(san)
        except ValueError as error:
            raise ValueError(f"illegal SAN at ply {ply}: {san!r}") from error
    return board


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def evaluate(model: SanNgramModel, games: Sequence[GameRecord]) -> dict[str, float | int]:
    correct = 0
    fallback_correct = 0
    context_hits = 0
    predictions = 0
    for game in games:
        board = chess.Board()
        history: list[str] = []
        for target in game.moves:
            prediction = model.predict(history)
            fallback = min(board.san(move) for move in board.legal_moves)
            predictions += 1
            correct += int(prediction.san == target)
            fallback_correct += int(fallback == target)
            context_hits += int(prediction.source.endswith("context"))
            board.push_san(target)
            history.append(target)
    denominator = max(1, predictions)
    return {
        "validation_games": len(games),
        "validation_plies": predictions,
        "validation_top1_accuracy": correct / denominator,
        "validation_deterministic_fallback_top1_accuracy": fallback_correct
        / denominator,
        "validation_context_hit_rate": context_hits / denominator,
        "validation_legal_move_rate": 1.0,
    }


def train_from_parquet(
    data_path: Path,
    output_dir: Path,
    max_games: int,
    validation_percent: int,
    seed: int,
    order: int,
    top_moves_per_context: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    valid: list[GameRecord] = []
    invalid_games = 0
    for game in iter_game_records(data_path, max_games=max_games):
        if validate_game(game.moves):
            valid.append(game)
        else:
            invalid_games += 1

    training = [
        game
        for game in valid
        if not is_validation_game(game.site, seed, validation_percent)
    ]
    validation = [
        game for game in valid if is_validation_game(game.site, seed, validation_percent)
    ]
    if not training or not validation:
        raise ValueError("the deterministic split must contain training and validation games")

    model = SanNgramModel(order=order, top_moves_per_context=top_moves_per_context)
    model.fit(game.moves for game in training)
    model.prune()
    metrics: dict[str, Any] = {
        "experiment_id": "0001-basic-san-ngram",
        "completed_at": datetime.now(UTC).isoformat(),
        "code_revision": git_revision(),
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": data_path.as_posix(),
        "dataset_file_sha256": sha256_file(data_path),
        "seed": seed,
        "max_games": max_games,
        "valid_games": len(valid),
        "invalid_games": invalid_games,
        "training_games": len(training),
        "training_plies": sum(len(game.moves) for game in training),
        "validation_percent": validation_percent,
        "order": order,
        "top_moves_per_context": top_moves_per_context,
        **evaluate(model, validation),
    }
    checkpoint_path = output_dir / "model.json.gz"
    metrics.update(model.save(checkpoint_path, metadata=metrics))
    metrics["elapsed_seconds"] = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def self_play(model: SanNgramModel, max_plies: int) -> chess.pgn.Game:
    game = chess.pgn.Game()
    game.headers["Event"] = "Chess GPT baseline self-play smoke test"
    board = game.board()
    node = game
    history: list[str] = []
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        prediction = model.predict(history)
        move = board.parse_san(prediction.san)
        node = node.add_variation(move)
        board.push(move)
        history.append(prediction.san)
    game.headers["Result"] = board.result(claim_draw=True) if board.is_game_over() else "*"
    return game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the versioned n-gram baseline")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--max-games", type=int, default=10_000)
    train.add_argument("--validation-percent", type=int, default=10)
    train.add_argument("--seed", type=int, default=20260725)
    train.add_argument("--order", type=int, default=2)
    train.add_argument("--top-moves-per-context", type=int, default=16)

    move = subparsers.add_parser("move", help="Return one legal SAN move")
    move.add_argument("--checkpoint", type=Path, required=True)
    move.add_argument("--moves", nargs="*", default=[])

    play = subparsers.add_parser("self-play", help="Run a legal-move smoke game")
    play.add_argument("--checkpoint", type=Path, required=True)
    play.add_argument("--max-plies", type=int, default=80)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        metrics = train_from_parquet(
            data_path=args.data,
            output_dir=args.output,
            max_games=args.max_games,
            validation_percent=args.validation_percent,
            seed=args.seed,
            order=args.order,
            top_moves_per_context=args.top_moves_per_context,
        )
        json.dump(metrics, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    model = SanNgramModel.load(args.checkpoint)
    if args.command == "move":
        print(model.predict(args.moves).san)
        return
    if args.command == "self-play":
        print(self_play(model, max_plies=args.max_plies), end="\n\n")
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
