from __future__ import annotations

import json
import subprocess
from pathlib import Path

import chess
import onnx
import pyarrow.parquet as pq
import torch

from chess_gpt.snapshot_data import load_frozen_files, prepare_pgn
from chess_gpt.snapshot_model import (
    MOVE_VOCAB_SIZE,
    ModelConfig,
    SnapshotPolicy,
    encode_board,
    move_index,
)
from chess_gpt.snapshot_package import export_snapshot_package
from chess_gpt.snapshot_training import TrainConfig, train_policy


def test_initial_position_has_a_stable_snapshot_and_move_target() -> None:
    snapshot = encode_board(chess.Board())

    assert snapshot.squares == (
        4,
        2,
        3,
        5,
        6,
        3,
        2,
        4,
        *(1 for _ in range(8)),
        *(0 for _ in range(32)),
        *(7 for _ in range(8)),
        10,
        8,
        9,
        11,
        12,
        9,
        8,
        10,
    )
    assert snapshot.state == (0, 1, 1, 1, 1, 64, 0)
    assert snapshot.phase == 0
    assert move_index(chess.Move.from_uci("e2e4")) == 796

    board = chess.Board()
    board.push_san("e4")
    assert encode_board(board).state[5] == 64
    for san in ("a6", "e5", "d5"):
        board.push_san(san)
    assert encode_board(board).state[5] == chess.D6


def test_frozen_manifest_is_the_only_download_source() -> None:
    files = load_frozen_files(Path("data/dataset.toml"))

    assert [item.month for item in files] == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert [item.split for item in files] == ["train", "train", "train", "validation"]
    assert all(len(item.sha256) == 64 for item in files)


def test_prepare_pgn_turns_each_move_into_a_reusable_board_target_pair(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text(
        '[Event "Tiny"]\n[Site "game-1"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 *\n'
    )
    output = tmp_path / "positions.parquet"

    result = prepare_pgn(source, output, split="train", max_games=1)
    rows = pq.read_table(output).to_pylist()

    assert result.games == 1
    assert result.positions == 3
    assert [row["target"] for row in rows] == [796, 3364, 405]
    assert rows[0]["squares"] == list(encode_board(chess.Board()).squares)
    assert rows[0]["game_id"] == "game-1"
    assert rows[0]["phase"] == 0

    compressed = tmp_path / "tiny.pgn.zst"
    subprocess.run(["zstd", "-q", str(source), "-o", str(compressed)], check=True)
    capped_output = tmp_path / "capped.parquet"
    capped = prepare_pgn(compressed, capped_output, split="train", max_games=1)
    assert capped.games == 1
    assert capped.positions == 3


def test_both_policy_variants_score_the_same_stable_move_vocabulary() -> None:
    squares = torch.zeros((2, 64), dtype=torch.long)
    state = torch.zeros((2, 7), dtype=torch.long)
    phase = torch.tensor([0, 2], dtype=torch.long)

    for architecture in ("snapshot", "phase_moe"):
        model = SnapshotPolicy(
            ModelConfig(
                architecture=architecture,
                d_model=32,
                layers=1,
                heads=4,
                ff_multiplier=2,
                dropout=0.0,
            )
        )
        logits = model(squares, state, phase)

        assert logits.shape == (2, MOVE_VOCAB_SIZE)
        assert torch.isfinite(logits).all()


def test_tiny_training_run_records_checkpoint_budget_and_validation(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text(
        """[Event "One"]
[Site "game-1"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *

[Event "Two"]
[Site "game-2"]
[Result "*"]

1. d4 d5 2. c4 e6 *
"""
    )
    train_data = tmp_path / "train.parquet"
    validation_data = tmp_path / "validation.parquet"
    prepare_pgn(source, train_data, split="train", max_games=1)
    prepare_pgn(source, validation_data, split="validation", max_games=2)

    metrics = train_policy(
        [train_data],
        [validation_data],
        tmp_path / "run",
        TrainConfig(
            experiment_id="tiny-snapshot",
            model=ModelConfig(
                architecture="snapshot",
                d_model=16,
                layers=1,
                heads=4,
                ff_multiplier=2,
                dropout=0.0,
            ),
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
        ),
    )

    assert (tmp_path / "run/checkpoint.pt").is_file()
    assert (tmp_path / "run/metrics.json").is_file()
    assert metrics["training_positions"] == 4
    assert metrics["profiled_training_flops"] > 0
    assert metrics["profiled_training_flops"] < 1e18
    assert 0 <= metrics["validation_legal_top1_accuracy"] <= 1

    package_dir = tmp_path / "browser"
    manifest = export_snapshot_package(
        checkpoint=tmp_path / "run/checkpoint.pt",
        entrypoint_source=Path("adapters/board-policy/entry.source.js"),
        output=package_dir,
    )

    onnx.checker.check_model(onnx.load(package_dir / "model.onnx"))
    assert manifest["schema"] == "chess-gpt-package-v1"
    assert manifest["config"]["architecture"] == "snapshot"
    assert sum(path.stat().st_size for path in package_dir.iterdir()) < 100_000_000

    entry_url = (package_dir / "entry.js").resolve().as_uri()
    ort_url = Path("site/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs").resolve().as_uri()
    model_path = (package_dir / "model.onnx").resolve().as_posix()
    vocabulary_path = (package_dir / "vocabulary.json").resolve().as_posix()
    script = f"""
import fs from "node:fs";
import * as ort from {json.dumps(ort_url)};
import {{ loadPackage }} from {json.dumps(entry_url)};
const artifacts = new Map([
  ["model", new Uint8Array(fs.readFileSync({json.dumps(model_path)}))],
  ["vocabulary", new Uint8Array(fs.readFileSync({json.dumps(vocabulary_path)}))],
]);
const loaded = await loadPackage({{ artifacts, config: {{}}, ort }});
const game = await loaded.newGame({{ random: () => 0.5 }});
const move = await game.chooseMove({{ history: [], legalMoves: ["e4", "d4", "Nf3"] }});
if (!["e4", "d4", "Nf3"].includes(move)) throw new Error(`illegal output: ${{move}}`);
await game.dispose();
await loaded.dispose();
"""
    subprocess.run(["node", "--input-type=module", "-e", script], check=True)
