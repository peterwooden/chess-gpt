"""Round-robin over the Experiment 67 data-slicing checkpoints, beam-4 both sides.

Downloads the nine slice67 checkpoints from the shard repo, plays every pair with
color-reversed seeded openings, writes the artifact incrementally after each pair,
and finishes with a joint relative-Elo fit (mean-anchored at 0).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from lab.match import make_player, play_series
from lab.roundrobin import fit_elos

DATASET = "peterwooden/chess-gpt-lab-shards"
ARMS = ("control", "unfiltered", "elo1800", "elo2000", "elo2200",
        "nobullet", "decisive", "noforfeit", "dedup64")
BEAM = {"search": "beam", "depth": 4, "contempt": 0.15}


def pair_seed(a: str, b: str, base: int) -> int:
    digest = hashlib.sha256(f"{base}:{a}:{b}".encode()).hexdigest()
    return int(digest[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, default=Path("runs/lab/slice67-roundrobin.json"))
    args = parser.parse_args()

    checkpoints = {
        arm: Path(hf_hub_download(DATASET, f"results3/slice67-{arm}.pt", repo_type="dataset"))
        for arm in ARMS
    }
    pairs = list(itertools.combinations(ARMS, 2))
    random.Random(args.seed).shuffle(pairs)  # balance partial coverage if stopped early

    done: list[dict] = []
    if args.output.exists():
        done = json.loads(args.output.read_text())["pairs"]
    finished = {(p["a"], p["b"]) for p in done}

    for a, b in pairs:
        if (a, b) in finished:
            continue
        rng = random.Random(pair_seed(a, b, args.seed))
        player_a = make_player(checkpoints[a], **BEAM)
        player_b = make_player(checkpoints[b], **BEAM)
        tally = play_series(player_a, player_b, args.games_per_pair, rng)
        done.append({"a": a, "b": b, **tally})
        print(json.dumps(done[-1]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"pairs": done}, indent=2))

    aggregates = [
        (p["a"], p["b"], p["win"] + 0.5 * p["draw"], p["win"] + p["draw"] + p["loss"])
        for p in done
    ]
    ratings = fit_elos(aggregates, anchors={})
    artifact = {"pairs": done, "elo": ratings, "decode": BEAM, "games_per_pair": args.games_per_pair}
    args.output.write_text(json.dumps(artifact, indent=2))
    for name, elo in sorted(ratings.items(), key=lambda item: -item[1]):
        print(f"{name:<11} {elo:>7.1f}")


if __name__ == "__main__":
    main()
