"""Hardware speed trials for the full-budget run: 4 flavors x 2 compile modes.

Each job trains the final architecture (d384x12L, K=V tied) on the slice67-nobullet
shard for a 180s steady-state window after compile, printing the calibrated steps/s.
Usage: uv run python lab/hwtrial_launch.py [flavor ...]   (default: all four)
"""

import json
import subprocess
import sys
from pathlib import Path

FLAVORS = ("a100-large", "h200", "rtx-pro-6000", "l40sx1")
MODES = ("", "reduce-overhead")

# Final-run recipe (settled 2026-08-07, lab/PROPOSAL_FULL_BUDGET.md) except:
# time-boxed, subsampled for fast load, no checkpoint.
BASE = {
    "optimizer": "adamw", "lr": 0.0012, "wd": 0.01, "betas": [0.9, 0.999], "eps": 1e-08,
    "clip": 0.0, "schedule": "cosine", "warmup": 0.05, "cosine_floor": 0.0, "cycles": 1,
    "embed_lr_scale": 1.0, "batch": 1024, "epochs": 1, "arch": "transformer", "heads": 8,
    "ffn_ratio": 1, "attn_bias": True, "layers": 12, "d_model": 384, "dropout": 0.1,
    "value_weight": 1.0, "value_mode": "ce", "compile": True, "seed": 20260730,
    "save_ckpt": False, "history_k": 8, "flip": True, "bilinear_head": True,
    "qkv_tie": "kv", "time_budget_s": 180.0, "subsample": 0.25,
    "shard_set": "slice67-nobullet",
}


def main() -> None:
    requested = sys.argv[1:] or list(FLAVORS)
    unknown = [f for f in requested if f not in FLAVORS]
    if unknown:
        sys.exit(f"pass flavors from {FLAVORS}; unknown: {unknown}")
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    launched = []
    for flavor in requested:
        for mode in MODES:
            tag = f"hw-{flavor}{'-ro' if mode else ''}"
            recipe = {**BASE, "id": tag, "compile_mode": mode}
            result = subprocess.run(
                [
                    "hf", "jobs", "uv", "run", "--flavor", flavor, "--timeout", "15m",
                    "--detach", "--secrets", f"HF_TOKEN={token}",
                    "--env", f"RECIPE={json.dumps(recipe)}",
                    "lab/cloud_sweep.py",
                ],
                capture_output=True, text=True,
            )
            lines = (result.stdout + result.stderr).strip().splitlines()
            job = next((part for chunk in lines for part in chunk.split() if "/" in part and len(part) > 20), "?")
            status = "ok" if result.returncode == 0 else "LAUNCH-FAIL"
            launched.append({"id": tag, "job": job, "status": status})
            print(json.dumps(launched[-1]), flush=True)
    out = Path("runs/lab/hwtrial-jobs.json")
    existing = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(existing + launched, indent=2))
    failures = sum(1 for item in launched if item["status"] != "ok")
    print(json.dumps({"launched": len(launched) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
