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
from chess_gpt.snapshot_monitor import request_stop
from chess_gpt.snapshot_package import export_snapshot_package
from chess_gpt.snapshot_training import TrainConfig, profiled_training_flops, train_policy


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
    assert result.selected_games == 1
    assert result.positions == 3
    assert result.filtered_games == 0
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


def test_prepare_pgn_keeps_only_moves_by_winners_at_or_above_the_elo_floor(
    tmp_path,
) -> None:
    source = tmp_path / "rated.pgn"
    source.write_text(
        """[Event "Strong winner, weak loser"]
[Site "accepted-white"]
[WhiteElo "1600"]
[BlackElo "900"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Weak winner"]
[Site "rejected-white"]
[WhiteElo "1599"]
[BlackElo "2200"]
[Result "1-0"]

1. d4 d5 2. c4 e6 1-0

[Event "Strong black winner"]
[Site "accepted-black"]
[WhiteElo "1100"]
[BlackElo "1700"]
[Result "0-1"]

1. c4 e5 2. Nc3 Nf6 0-1

[Event "Draw"]
[Site "rejected-draw"]
[WhiteElo "2000"]
[BlackElo "2000"]
[Result "1/2-1/2"]

1. Nf3 Nf6 1/2-1/2
"""
    )
    output = tmp_path / "winner-positions.parquet"

    result = prepare_pgn(
        source,
        output,
        split="train",
        winner_only=True,
        min_winner_elo=1600,
    )
    parquet = pq.ParquetFile(output)
    rows = parquet.read().to_pylist()
    metadata = parquet.schema_arrow.metadata or {}

    assert result.games == 4
    assert result.selected_games == 2
    assert result.filtered_games == 2
    assert result.positions == 4
    assert [row["game_id"] for row in rows] == [
        "accepted-white",
        "accepted-white",
        "accepted-black",
        "accepted-black",
    ]
    assert [row["ply"] for row in rows] == [0, 2, 1, 3]
    assert metadata[b"prepared_format"] == b"board-snapshot-winner-v1"
    assert metadata[b"target_side"] == b"winner"
    assert metadata[b"min_winner_elo"] == b"1600"


def test_prepare_pgn_can_stop_after_a_requested_number_of_selected_games(tmp_path) -> None:
    source = tmp_path / "rated.pgn"
    source.write_text(
        """[Site "filtered"]
[WhiteElo "1500"]
[BlackElo "1500"]
[Result "1-0"]

1. e4 e5 1-0

[Site "selected"]
[WhiteElo "1700"]
[BlackElo "1000"]
[Result "1-0"]

1. d4 d5 1-0

[Site "not-scanned"]
[WhiteElo "1800"]
[BlackElo "1800"]
[Result "1-0"]

1. c4 c5 1-0
"""
    )

    result = prepare_pgn(
        source,
        tmp_path / "positions.parquet",
        split="train",
        max_selected_games=1,
        winner_only=True,
        min_winner_elo=1600,
    )

    assert result.games == 2
    assert result.selected_games == 1
    assert result.filtered_games == 1
    assert result.positions == 1


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


def test_laptop_moe_matches_the_recorded_size_and_flop_profile() -> None:
    config = ModelConfig(
        architecture="phase_moe",
        d_model=336,
        layers=6,
        heads=8,
        ff_multiplier=4,
        dropout=0.1,
    )

    assert sum(parameter.numel() for parameter in SnapshotPolicy(config).parameters()) == 12_397_296
    assert profiled_training_flops(config, positions=1, epochs=1) == 3_297_200_256


def test_laptop_snapshot_matches_the_recorded_size_and_flop_profile() -> None:
    config = ModelConfig(
        architecture="snapshot",
        d_model=336,
        layers=6,
        heads=8,
        ff_multiplier=4,
        dropout=0.1,
    )

    parameter_count = sum(
        parameter.numel() for parameter in SnapshotPolicy(config).parameters()
    )
    assert parameter_count == 10_586_256
    assert profiled_training_flops(config, positions=1, epochs=1) == 3_286_362_240


def test_plain_snapshot_policy_does_not_consume_history_derived_phase() -> None:
    model = SnapshotPolicy(
        ModelConfig(
            architecture="snapshot",
            d_model=16,
            layers=1,
            heads=4,
            ff_multiplier=2,
            dropout=0.0,
        )
    ).eval()
    squares = torch.zeros((1, 64), dtype=torch.long)
    state = torch.zeros((1, 7), dtype=torch.long)

    opening = model(squares, state, torch.tensor([0]))
    endgame = model(squares, state, torch.tensor([2]))

    assert torch.equal(opening, endgame)


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
            enforce_frozen_data=False,
        ),
    )

    assert (tmp_path / "run/checkpoint.pt").is_file()
    assert (tmp_path / "run/metrics.json").is_file()
    assert metrics["training_positions"] == 4
    assert metrics["training_tokens"] == 4 * 65
    assert metrics["training_precision"] == "float32"
    assert metrics["hardware"]["platform"]
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


def test_training_rejects_unverified_or_wrong_split_shards_by_default(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text('[Site "game-1"]\n[Result "*"]\n\n1. e4 *\n')
    shard = tmp_path / "unverified.parquet"
    prepare_pgn(source, shard, split="validation", max_games=1)

    try:
        train_policy(
            [shard],
            [shard],
            tmp_path / "run",
            TrainConfig(
                experiment_id="must-reject",
                model=ModelConfig(d_model=16, layers=1, heads=4, ff_multiplier=2),
                device="cpu",
            ),
        )
    except ValueError as error:
        assert "frozen dataset" in str(error)
    else:
        raise AssertionError("unverified training data was accepted")


def test_training_stops_cleanly_and_logs_each_observed_loss(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text('[Site "game-1"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    train_data = tmp_path / "train.parquet"
    validation_data = tmp_path / "validation.parquet"
    prepare_pgn(source, train_data, split="train", max_games=1)
    prepare_pgn(source, validation_data, split="validation", max_games=1)

    metrics = train_policy(
        [train_data],
        [validation_data],
        tmp_path / "run",
        TrainConfig(
            experiment_id="interruptible-smoke",
            model=ModelConfig(
                architecture="phase_moe",
                d_model=16,
                layers=1,
                heads=4,
                ff_multiplier=2,
                dropout=0.0,
            ),
            epochs=2,
            batch_size=2,
            max_updates=1,
            log_every_updates=1,
            seed=9,
            device="cpu",
            enforce_frozen_data=False,
        ),
    )

    events = [json.loads(line) for line in (tmp_path / "run/losses.jsonl").read_text().splitlines()]
    assert metrics["stop_reason"] == "update_limit"
    assert metrics["updates"] == 1
    assert metrics["training_positions"] == 2
    assert metrics["actual_training_flops"] < metrics["planned_training_flops"]
    assert len(events) == 1
    assert events[0]["update"] == 1
    assert events[0]["loss"] > 0

    stopped_run = tmp_path / "stopped-run"
    request_stop(stopped_run)
    stopped_metrics = train_policy(
        [train_data],
        [validation_data],
        stopped_run,
        TrainConfig(
            experiment_id="button-stop-smoke",
            model=ModelConfig(d_model=16, layers=1, heads=4, ff_multiplier=2),
            epochs=2,
            batch_size=2,
            log_every_updates=1,
            device="cpu",
            enforce_frozen_data=False,
        ),
    )
    assert stopped_metrics["stop_reason"] == "stop_requested"
    assert stopped_metrics["updates"] == 1
    assert not (stopped_run / "STOP").exists()


def test_training_time_limit_still_produces_a_loadable_checkpoint(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text('[Site "game-1"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    train_data = tmp_path / "train.parquet"
    validation_data = tmp_path / "validation.parquet"
    prepare_pgn(source, train_data, split="train", max_games=1)
    prepare_pgn(source, validation_data, split="validation", max_games=1)

    metrics = train_policy(
        [train_data],
        [validation_data],
        tmp_path / "run",
        TrainConfig(
            experiment_id="timed-smoke",
            model=ModelConfig(d_model=16, layers=1, heads=4, ff_multiplier=2),
            epochs=20,
            batch_size=2,
            max_seconds=0,
            log_every_updates=1,
            device="cpu",
            enforce_frozen_data=False,
        ),
    )

    checkpoint = torch.load(tmp_path / "run/checkpoint.pt", weights_only=False)
    assert metrics["stop_reason"] == "time_limit"
    assert metrics["updates"] == 1
    assert checkpoint["model_type"] == "board_snapshot_policy"


def test_training_position_limit_matches_an_exact_compute_budget(tmp_path) -> None:
    source = tmp_path / "tiny.pgn"
    source.write_text('[Site "game-1"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n')
    train_data = tmp_path / "train.parquet"
    validation_data = tmp_path / "validation.parquet"
    prepare_pgn(source, train_data, split="train", max_games=1)
    prepare_pgn(source, validation_data, split="validation", max_games=1)
    model = ModelConfig(d_model=16, layers=1, heads=4, ff_multiplier=2)

    metrics = train_policy(
        [train_data],
        [validation_data],
        tmp_path / "run",
        TrainConfig(
            experiment_id="position-capped-smoke",
            model=model,
            epochs=2,
            batch_size=4,
            max_positions=3,
            log_every_updates=1,
            device="cpu",
            enforce_frozen_data=False,
        ),
    )

    assert metrics["stop_reason"] == "position_limit"
    assert metrics["training_positions"] == 3
    assert metrics["planned_training_flops"] == profiled_training_flops(model, 3, 1)
    assert metrics["actual_training_flops"] == metrics["planned_training_flops"]
