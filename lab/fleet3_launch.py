"""Fire Fleet-3 arms as detached HF jobs."""

import json
import subprocess
import sys
from pathlib import Path

ARMS: list[dict] = [
    {"id": "c1-control"},
    {"id": "c2-control", "seed": 1},
    {"id": "c3-control", "seed": 2},
    {"id": "d2-skipopen", "ply_min": 6},
    {"id": "d3-openonly", "ply_max": 20},
    {"id": "d4-endonly", "ply_min": 40},
    {"id": "d5-half2x", "subsample": 0.5, "epochs": 4},
    {"id": "d6-labelnoise", "label_noise": 0.05},
    {"id": "r1-hist0", "history_k": 0},
    {"id": "r2-hist4", "history_k": 4},
    {"id": "r3-histshuffle", "shuffle_history": True},
    {"id": "r4-nohalfmove", "zero_halfmove": True},
    {"id": "r5-nocastle", "zero_castling": True},
    {"id": "r6-rankfile", "rankfile_squares": True},
    {"id": "r7-sqdrop", "input_square_dropout": 0.05},
    {"id": "r8-d192", "d_model": 192},
    {"id": "r9-sqshuffle", "shuffle_squares": True},
    {"id": "a1-h1536", "hidden": 1536},
    {"id": "a2-h768", "hidden": 768},
    {"id": "a3-gated", "gated": True},
    {"id": "a4-twotower", "two_tower": True},
    {"id": "a5-inputnorm", "input_norm": True},
    {"id": "a6-inputres", "input_residual": True},
    {"id": "a7-lowrank", "token_rank": 64},
    {"id": "a8-untied", "untied_readout": True},
    {"id": "j1-bce", "value_mode": "bce"},
    {"id": "j2-mse", "value_mode": "mse"},
    {"id": "j3-softvalue", "value_mode": "smooth"},
    {"id": "j4-focal", "focal": True},
    {"id": "j5-entropy", "entropy_bonus": 0.01},
    {"id": "j6-nextmove", "aux_next_move": True},
    {"id": "j7-masked", "aux_masked": True},
    {"id": "j8-vw5", "value_weight": 5.0},
    {"id": "j9-distill2", "distill": True},
    {"id": "o1-warm5", "warmup": 0.05},
    {"id": "o2-warm05", "warmup": 0.005},
    {"id": "o3-floor10", "cosine_floor": 0.1},
    {"id": "o4-lr18", "lr": 1.8e-3},
    {"id": "o5-cycles2", "cycles": 2},
    {"id": "o6-embedlr", "embed_lr_scale": 0.1},
    {"id": "o7-eps", "eps": 1e-4},
    {"id": "o8-sgd", "optimizer": "sgd"},
    {"id": "o9-muon2", "optimizer": "muon2"},
    {"id": "o10-b512", "batch": 512, "lr": 8.5e-4},
    {"id": "o11-gradnoise", "grad_noise": 0.001},
    {"id": "o12-swa", "swa": True},
]

BIG_ARM = {"id": "d1-960k", "big_shard": True, "epochs": 1}


def launch(arm: dict, token: str) -> dict:
    result = subprocess.run(
        [
            "hf", "jobs", "uv", "run", "--flavor", "a100-large", "--timeout", "30m",
            "--detach", "--secrets", f"HF_TOKEN={token}",
            "--env", f"RECIPE={json.dumps(arm)}",
            "lab/cloud_sweep.py",
        ],
        capture_output=True, text=True,
    )
    return {"id": arm["id"], "status": "ok" if result.returncode == 0 else "LAUNCH-FAIL"}


def main() -> None:
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    arms = [BIG_ARM] if "--big-only" in sys.argv else ARMS
    launched = [launch(arm, token) for arm in arms]
    for item in launched:
        print(json.dumps(item), flush=True)
    failures = sum(1 for item in launched if item["status"] != "ok")
    print(json.dumps({"launched": len(launched) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
