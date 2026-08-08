"""Launch the full-budget tournament run (or its preflight smoke) on rtx-pro-6000.

Usage: uv run python lab/fullbudget_launch.py smoke|run
Settled decisions: lab/PROPOSAL_FULL_BUDGET.md. 224,929 steps = 230.3M positions
= 0.97e18 profiler-FLOPs at 9,708,819 params (fwd 1,403,904,256 FLOPs/pos).
"""

import json
import subprocess
import sys
from pathlib import Path

RECIPE = {
    "id": "fullbudget-68", "optimizer": "adamw", "lr": 0.0012, "wd": 0.01,
    "betas": [0.9, 0.999], "eps": 1e-08, "clip": 0.0, "schedule": "cosine",
    "warmup": 0.05, "cosine_floor": 0.0, "cycles": 1, "batch": 1024, "epochs": 1,
    "arch": "transformer", "heads": 8, "ffn_ratio": 1, "attn_bias": True,
    "layers": 12, "d_model": 384, "dropout": 0.1, "value_weight": 1.0,
    "value_mode": "ce", "compile": True, "seed": 20260730, "save_ckpt": True,
    "history_k": 8, "flip": True, "bilinear_head": True, "qkv_tie": "kv",
    "shard_set": "fullbudget-games", "val_shard": "games:shards/games-2026-04.parquet",
    "max_steps": 224929, "ckpt_every_frac": 0.02,
}

SMOKE = {
    **RECIPE, "id": "fullbudget-smoke", "max_steps": 0, "time_budget_s": 300.0,
    "ckpt_every_frac": 0.34,
}


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("smoke", "run"):
        sys.exit("usage: fullbudget_launch.py smoke|run")
    recipe, timeout = (SMOKE, "25m") if mode == "smoke" else (RECIPE, "6h")
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    result = subprocess.run(
        [
            "hf", "jobs", "uv", "run", "--flavor", "rtx-pro-6000", "--timeout", timeout,
            "--detach", "--secrets", f"HF_TOKEN={token}",
            "--env", f"RECIPE={json.dumps(recipe)}",
            "lab/cloud_sweep.py",
        ],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
