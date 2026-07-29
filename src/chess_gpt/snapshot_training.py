"""Train and evaluate reproducible board-snapshot chess policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from chess_gpt.snapshot_model import ModelConfig, SnapshotPolicy, move_index

TRAINING_FLOP_LIMIT = 10**18
TOKENS_PER_POSITION = 65


@dataclass(frozen=True)
class TrainConfig:
    experiment_id: str
    model: ModelConfig
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 20260729
    device: str = "auto"


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


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _batches(
    paths: Sequence[Path], batch_size: int, *, shuffle: bool, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    ordered_paths = list(paths)
    if shuffle:
        rng.shuffle(ordered_paths)
    for path in ordered_paths:
        parquet = pq.ParquetFile(path)
        row_groups = list(range(parquet.num_row_groups))
        if shuffle:
            rng.shuffle(row_groups)
        for row_group in row_groups:
            table = parquet.read_row_group(
                row_group, columns=["squares", "state", "phase", "target"]
            )
            rows = table.to_pylist()
            order = np.arange(len(rows))
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(rows), batch_size):
                chosen = order[start : start + batch_size]
                batch = [rows[int(index)] for index in chosen]
                yield (
                    torch.tensor([row["squares"] for row in batch], dtype=torch.long),
                    torch.tensor([row["state"] for row in batch], dtype=torch.long),
                    torch.tensor([row["phase"] for row in batch], dtype=torch.long),
                    torch.tensor([row["target"] for row in batch], dtype=torch.long),
                )


def _position_count(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def _shard_record(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    raw_metadata = parquet.schema_arrow.metadata or {}
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "rows": parquet.metadata.num_rows,
        "metadata": {
            key.decode(): value.decode() for key, value in sorted(raw_metadata.items())
        },
    }


def _board_from_tensors(squares: Sequence[int], state: Sequence[int]) -> chess.Board:
    board = chess.Board.empty()
    for square, code in enumerate(squares):
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            piece_type = code if code <= 6 else code - 6
            board.set_piece_at(square, chess.Piece(piece_type, color))
    board.turn = chess.BLACK if state[0] else chess.WHITE
    rights = 0
    for enabled, square in zip(state[1:5], (chess.H1, chess.A1, chess.H8, chess.A8), strict=True):
        if enabled:
            rights |= chess.BB_SQUARES[square]
    board.castling_rights = rights
    board.ep_square = None if state[5] == 64 else state[5]
    board.halfmove_clock = state[6]
    return board


def _legal_prediction(logits: torch.Tensor, squares: Sequence[int], state: Sequence[int]) -> int:
    board = _board_from_tensors(squares, state)
    candidates = [move_index(move) for move in board.legal_moves]
    if not candidates:
        return int(logits.argmax().item())
    candidate_tensor = torch.tensor(candidates, device=logits.device)
    return candidates[int(logits[candidate_tensor].argmax().item())]


def profiled_training_flops(config: ModelConfig, positions: int, epochs: int) -> int:
    """Conservative dense multiply-add profile: forward plus 2x backward."""
    width = config.d_model
    sequence = TOKENS_PER_POSITION
    attention_projection = 8 * sequence * width * width
    attention_scores = 4 * sequence * sequence * width
    feed_forward = 4 * sequence * width * width * config.ff_multiplier
    encoder = config.layers * (
        attention_projection + attention_scores + feed_forward
    )
    expert_count = 3 if config.architecture == "phase_moe" else 1
    experts = expert_count * 4 * width * width * config.ff_multiplier
    policy = 2 * width * 4272
    return 3 * (encoder + experts + policy) * positions * epochs


def _evaluate(
    model: SnapshotPolicy,
    paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = correct = legal_correct = positions = 0
    model.eval()
    with torch.no_grad():
        for squares, state, phase, target in _batches(
            paths, batch_size, shuffle=False, seed=0
        ):
            logits = model(squares.to(device), state.to(device), phase.to(device))
            target_device = target.to(device)
            total_loss += float(loss_function(logits, target_device).item())
            correct += int((logits.argmax(dim=1) == target_device).sum().item())
            for index in range(len(target)):
                prediction = _legal_prediction(
                    logits[index], squares[index].tolist(), state[index].tolist()
                )
                legal_correct += int(prediction == int(target[index]))
            positions += len(target)
    denominator = max(1, positions)
    return {
        "validation_positions": positions,
        "validation_loss": total_loss / denominator,
        "validation_raw_top1_accuracy": correct / denominator,
        "validation_legal_top1_accuracy": legal_correct / denominator,
        "validation_legal_move_rate": 1.0,
    }


def train_policy(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    output_dir: Path,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train one specified policy and persist its full reproducibility record."""
    if not train_paths or not validation_paths:
        raise ValueError("training and validation shards are both required")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _device(config.device)
    model = SnapshotPolicy(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    training_positions = _position_count(train_paths)
    flops = profiled_training_flops(config.model, training_positions, config.epochs)
    if flops > TRAINING_FLOP_LIMIT:
        raise ValueError(f"planned training lineage uses {flops:,} FLOPs, over the limit")

    started = time.perf_counter()
    model.train()
    updates = 0
    final_training_loss = 0.0
    for epoch in range(config.epochs):
        for squares, state, phase, target in _batches(
            train_paths,
            config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(squares.to(device), state.to(device), phase.to(device))
            loss = loss_function(logits, target.to(device))
            loss.backward()
            optimizer.step()
            final_training_loss = float(loss.item())
            updates += 1

    validation = _evaluate(model, validation_paths, config.batch_size, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = {
        "format_version": 1,
        "model_type": "board_snapshot_policy",
        "model_config": asdict(config.model),
        "train_config": {**asdict(config), "model": asdict(config.model)},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    torch.save(checkpoint, checkpoint_path)
    metrics: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "completed_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(),
        "environment_lock_sha256": _sha256(Path("uv.lock")),
        "seed": config.seed,
        "device": str(device),
        "model_config": asdict(config.model),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_positions": training_positions,
        "epochs": config.epochs,
        "updates": updates,
        "final_training_loss": final_training_loss,
        "profiled_training_flops": flops,
        "training_flop_limit": TRAINING_FLOP_LIMIT,
        "train_shards": [_shard_record(path) for path in train_paths],
        "validation_shards": [_shard_record(path) for path in validation_paths],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        **validation,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--architecture", choices=("snapshot", "phase_moe"), required=True)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = TrainConfig(
        experiment_id=args.experiment_id,
        model=ModelConfig(
            architecture=args.architecture,
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
            ff_multiplier=args.ff_multiplier,
            dropout=args.dropout,
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    metrics = train_policy(args.train, args.validation, args.output, config)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
