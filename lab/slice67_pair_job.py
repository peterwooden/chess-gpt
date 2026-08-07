# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pyarrow", "huggingface_hub", "schedulefree", "chess"]
# ///
"""Play one Experiment 67 round-robin pair on a cloud box; upload the tally.

Expects PAIR env JSON: {"a": ..., "b": ..., "games": 20, "seed": 20260807}.
Clones the public repo for lab.match, downloads both checkpoints from the shard
repo, plays the color-reversed series beam-4 both sides, uploads the result to
results3/slice67-rr/<a>-vs-<b>.json.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DATASET = "peterwooden/chess-gpt-lab-shards"
REPO = "https://github.com/peterwooden/chess-gpt"
BEAM = {"search": "beam", "depth": 4, "contempt": 0.15}

subprocess.run(["git", "clone", "--depth", "1", REPO, "/tmp/repo"], check=True)
sys.path.insert(0, "/tmp/repo/src")
sys.path.insert(0, "/tmp/repo")
from lab.match import make_player, play_series  # noqa: E402

pair = json.loads(os.environ["PAIR"])
a, b, games, seed = pair["a"], pair["b"], pair.get("games", 20), pair.get("seed", 20260807)
block = pair.get("block", 0)
checkpoints = {
    arm: Path(hf_hub_download(DATASET, f"results3/slice67-{arm}.pt", repo_type="dataset"))
    for arm in (a, b)
}
rng = random.Random(int(hashlib.sha256(f"{seed}:{a}:{b}:{block}".encode()).hexdigest()[:8], 16))
tally = play_series(make_player(checkpoints[a], **BEAM), make_player(checkpoints[b], **BEAM), games, rng)
code = subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()
result = {"a": a, "b": b, **tally, "games": games, "seed": seed, "block": block,
          "decode": BEAM, "code": code}
print(json.dumps(result), flush=True)
for attempt in range(5):
    try:
        HfApi().upload_file(
            path_or_fileobj=json.dumps(result, indent=2).encode(),
            path_in_repo=f"results3/slice67-rr/{a}-vs-{b}-b{block}.json",
            repo_id=DATASET, repo_type="dataset",
        )
        break
    except Exception as error:  # concurrent commits can conflict; back off and retry
        print(f"upload attempt {attempt}: {error}", flush=True)
        time.sleep(5 + random.random() * 20)
