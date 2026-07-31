"""Round-robin among the fleet's top models, with a joint relative-Elo fit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lab.ladder import expected_score
from lab.match import make_player, play_series

FLEET = Path("runs/lab/fleet")

ENTRANTS = [
    ("28-contempt", FLEET / "28-value05.pt", {"search": "value1", "contempt": 0.15}),
    ("33-valueonly", FLEET / "33-valueonly.pt", {"search": "value1"}),
    ("06-40k", FLEET / "06-40k.pt", {}),
    ("34-epochs3", FLEET / "34-epochs3.pt", {}),
    ("39-ft-elite", FLEET / "39-ft-elite.pt", {}),
    ("40-ft-decisive", FLEET / "40-ft-decisive.pt", {}),
    ("15-mlp", FLEET / "15-mlp.pt", {}),
    ("05-20k", FLEET / "05-20k.pt", {}),
]


def fit_elos(pair_results: list[tuple[str, str, float, int]], anchors: dict[str, float]) -> dict[str, float]:
    """Coordinate-ascent MLE over pairwise aggregates, mean-anchored to ladder ratings."""
    names = sorted({name for a, b, _, _ in pair_results for name in (a, b)})
    ratings = {name: anchors.get(name, 0.0) for name in names}
    for _ in range(300):
        for name in names:
            best, best_ll = ratings[name], float("-inf")
            for candidate in range(-200, 1401, 4):
                ll = 0.0
                for a, b, points, games in pair_results:
                    if name not in (a, b):
                        continue
                    ra = candidate if a == name else ratings[a]
                    rb = candidate if b == name else ratings[b]
                    p = min(max(expected_score(ra - rb), 1e-9), 1 - 1e-9)
                    ll += points * __import__("math").log(p) + (games - points) * __import__("math").log(1 - p)
                if ll > best_ll:
                    best, best_ll = candidate, ll
            ratings[name] = best
    fitted_mean = sum(ratings.values()) / len(ratings)
    anchor_mean = sum(anchors.get(n, 0.0) for n in names) / len(names)
    return {name: round(value - fitted_mean + anchor_mean, 1) for name, value in ratings.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, default=FLEET / "roundrobin.json")
    args = parser.parse_args()

    anchors = {}
    for name, checkpoint, decode in ENTRANTS:
        tag = "search1" if name == "28-contempt" else ""
        for candidate in (
            checkpoint.with_suffix(".contempt.rating.json"),
            checkpoint.with_suffix(".rating.json"),
        ):
            if candidate.exists():
                anchors[name] = json.loads(candidate.read_text())["rating"]
                break

    players = {
        name: make_player(checkpoint, seed=args.seed, **decode)
        for name, checkpoint, decode in ENTRANTS
    }
    rng = random.Random(args.seed)
    pair_results: list[tuple[str, str, float, int]] = []
    matrix: dict[str, dict[str, str]] = {}
    for i, (name_a, _, _) in enumerate(ENTRANTS):
        for name_b, _, _ in [e for e in ENTRANTS[i + 1 :]]:
            tally = play_series(players[name_a], players[name_b], args.games_per_pair, rng)
            points = tally["win"] + 0.5 * tally["draw"]
            pair_results.append((name_a, name_b, points, args.games_per_pair))
            matrix.setdefault(name_a, {})[name_b] = f"{tally['win']}-{tally['draw']}-{tally['loss']}"
            print(json.dumps({"pair": [name_a, name_b], **tally}))

    elos = fit_elos(pair_results, anchors)
    report = {"games_per_pair": args.games_per_pair, "matrix": matrix,
              "ladder_anchors": anchors, "roundrobin_elo": elos}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"roundrobin_elo": elos}))


if __name__ == "__main__":
    main()
