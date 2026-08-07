"""Fire all 36 Experiment 67 round-robin pairs as parallel HF t4-small jobs."""

import itertools
import json
import subprocess
from pathlib import Path

ARMS = ("control", "unfiltered", "elo1800", "elo2000", "elo2200",
        "nobullet", "decisive", "noforfeit", "dedup64")


def main() -> None:
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
    launched = []
    for a, b in itertools.combinations(ARMS, 2):
        pair = {"a": a, "b": b, "games": 20, "seed": 20260807}
        result = subprocess.run(
            [
                "hf", "jobs", "uv", "run", "--flavor", "t4-small", "--timeout", "30m",
                "--detach", "--secrets", f"HF_TOKEN={token}",
                "--env", f"PAIR={json.dumps(pair)}",
                "lab/slice67_pair_job.py",
            ],
            capture_output=True, text=True,
        )
        status = "ok" if result.returncode == 0 else "LAUNCH-FAIL"
        launched.append({"pair": f"{a}-vs-{b}", "status": status})
        print(json.dumps(launched[-1]), flush=True)
    failures = sum(1 for item in launched if item["status"] != "ok")
    Path("runs/lab/slice67-rr-jobs.json").write_text(json.dumps(launched, indent=2))
    print(json.dumps({"launched": len(launched) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
