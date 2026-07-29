from __future__ import annotations

import json
from pathlib import Path

from chess_gpt.snapshot_monitor import MONITOR_HTML, load_loss_events, request_stop


def test_monitor_reads_live_losses_and_requests_a_graceful_stop(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    event = {
        "elapsed_seconds": 12.5,
        "epoch": 0,
        "loss": 7.25,
        "positions": 128,
        "smoothed_loss": 7.5,
        "update": 2,
    }
    (run_dir / "losses.jsonl").write_text(json.dumps(event) + "\n{partial")
    assert load_loss_events(run_dir) == [event]
    assert "Logarithmic training loss" in MONITOR_HTML
    assert "End training" in MONITOR_HTML
    request_stop(run_dir)
    assert (run_dir / "STOP").is_file()
