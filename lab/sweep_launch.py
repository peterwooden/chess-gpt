"""Fire Fleet-2 arms as detached HF jobs; print job ids."""

import json
import subprocess
import sys
from pathlib import Path

ARMS: list[dict] = [
    {"id": "c1-control"},
    {"id": "c2-control", "seed": 1},
    {"id": "c3-control", "seed": 2},
    {"id": "o1-muon", "optimizer": "muon"},
    {"id": "o2-lion", "optimizer": "lion"},
    {"id": "o3-sfree", "optimizer": "sfree"},
    {"id": "o4-wd01", "wd": 0.1},
    {"id": "o5-wd0", "wd": 0.0},
    {"id": "o6-beta98", "betas": [0.9, 0.98]},
    {"id": "o7-lr24", "lr": 2.4e-3},
    {"id": "o8-lr06", "lr": 6e-4},
    {"id": "o9-b4096", "batch": 4096, "lr": 4.8e-3},
    {"id": "o10-clip", "clip": 1.0},
    {"id": "s1-cosine", "schedule": "cosine"},
    {"id": "s2-warmcos", "schedule": "cosine", "warmup": 0.02},
    {"id": "s3-wsd", "schedule": "wsd"},
    {"id": "s4-step", "schedule": "step"},
    {"id": "s5-ema", "ema": 0.999},
    {"id": "s6-emacos", "ema": 0.999, "schedule": "cosine"},
    {"id": "m1-residual", "residual": True},
    {"id": "m2-resnorm", "residual": True, "block_norm": True},
    {"id": "m3-gelu", "activation": "gelu"},
    {"id": "m4-relu2", "activation": "relu2"},
    {"id": "m5-deep", "layers": 3, "hidden": 896},
    {"id": "m6-wide", "layers": 1, "hidden": 2048},
    {"id": "m7-drop0", "dropout": 0.0},
    {"id": "m8-drop02", "dropout": 0.2},
    {"id": "j1-vw05", "value_weight": 0.5},
    {"id": "j2-vw2", "value_weight": 2.0},
    {"id": "j3-plyweight", "value_ply_weight": True},
    {"id": "j4-decisive", "value_decisive_only": True},
    {"id": "j5-ls005", "label_smoothing": 0.05},
    {"id": "j6-auxplies", "aux_plies": True},
    {"id": "j7-auxmat", "aux_material": True},
    {"id": "j8-distill", "distill": True},
    {"id": "d1-elite25", "elite_mix": 0.25},
    {"id": "d2-elite50", "elite_mix": 0.5},
    {"id": "d3-endgame", "endgame_oversample": True},
    {"id": "d4-shortfirst", "curriculum": "short_first"},
    {"id": "d5-longfirst", "curriculum": "long_first"},
    {"id": "v1-compile", "compile": True},
    {"id": "v2-b2048", "batch": 2048, "lr": 1.7e-3},
]


def main() -> None:
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    launched = []
    for arm in ARMS:
        result = subprocess.run(
            [
                "hf", "jobs", "uv", "run", "--flavor", "a100-large", "--timeout", "30m",
                "--detach", "--secrets", f"HF_TOKEN={token}",
                "--env", f"RECIPE={json.dumps(arm)}",
                "lab/cloud_sweep.py",
            ],
            capture_output=True, text=True,
        )
        line = (result.stdout + result.stderr).strip().splitlines()
        job = next((part for chunk in line for part in chunk.split() if "/" in part and len(part) > 20), "?")
        status = "ok" if result.returncode == 0 else "LAUNCH-FAIL"
        launched.append({"id": arm["id"], "job": job, "status": status})
        print(json.dumps(launched[-1]), flush=True)
    Path("runs/lab/sweep2-jobs.json").write_text(json.dumps(launched, indent=2))
    failures = sum(1 for item in launched if item["status"] != "ok")
    print(json.dumps({"launched": len(launched) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
