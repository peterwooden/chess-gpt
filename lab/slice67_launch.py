"""Fire Experiment 67 data-slicing arms as detached HF jobs; print job ids.

Usage: uv run python lab/slice67_launch.py <arm> [<arm> ...]
Arms: control unfiltered elo1800 elo2000 elo2200 nobullet decisive noforfeit dedup64
"""

import json
import subprocess
import sys
from pathlib import Path

ARMS = ("control", "unfiltered", "elo1800", "elo2000", "elo2200",
        "nobullet", "decisive", "noforfeit", "dedup64")

# tie66-none's recipe verbatim (Experiment 66 control) except: per-arm id and
# shard_set, the frozen external val shard, and the 6.61e16-FLOP step count.
BASE = {
    "optimizer": "adamw", "lr": 0.0012, "wd": 0.01, "betas": [0.9, 0.999], "eps": 1e-08,
    "clip": 0.0, "schedule": "cosine", "warmup": 0.05, "cosine_floor": 0.0, "cycles": 1,
    "embed_lr_scale": 1.0, "batch": 1024, "epochs": 1, "arch": "transformer", "heads": 8,
    "ffn_ratio": 1, "attn_bias": True, "layers": 8, "d_model": 256, "dropout": 0.1,
    "value_weight": 1.0, "value_mode": "ce", "compile": True, "seed": 20260730,
    "save_ckpt": True, "history_k": 8, "flip": True, "bilinear_head": True,
    "qkv_tie": "none", "max_steps": 42000,
    "val_shard": "shards/slice67-val-april.parquet",
}


def main() -> None:
    requested = sys.argv[1:]
    unknown = [a for a in requested if a not in ARMS]
    if not requested or unknown:
        sys.exit(f"pass arm names from {ARMS}; unknown: {unknown}")
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    launched = []
    for arm in requested:
        recipe = {**BASE, "id": f"slice67-{arm}", "shard_set": f"slice67-{arm}"}
        result = subprocess.run(
            [
                "hf", "jobs", "uv", "run", "--flavor", "a100-large", "--timeout", "45m",
                "--detach", "--secrets", f"HF_TOKEN={token}",
                "--env", f"RECIPE={json.dumps(recipe)}",
                "lab/cloud_sweep.py",
            ],
            capture_output=True, text=True,
        )
        lines = (result.stdout + result.stderr).strip().splitlines()
        job = next((part for chunk in lines for part in chunk.split() if "/" in part and len(part) > 20), "?")
        status = "ok" if result.returncode == 0 else "LAUNCH-FAIL"
        launched.append({"id": arm, "job": job, "status": status})
        print(json.dumps(launched[-1]), flush=True)
    out = Path("runs/lab/slice67-jobs.json")
    existing = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(existing + launched, indent=2))
    failures = sum(1 for item in launched if item["status"] != "ok")
    print(json.dumps({"launched": len(launched) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
