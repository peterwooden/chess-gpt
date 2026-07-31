"""A self-calibrated opponent ladder that turns match play into a lab rating.

Rung 0 is the random-legal mover, defined as lab rating 0. Mixture rungs blend
random moves with Stockfish so the ladder reaches down to where tiny models live.
`calibrate` measures the rating gap between adjacent rungs by direct play;
`rate` walks a checkpoint up the ladder and fits its lab rating by maximum
likelihood over every game played.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from lab.match import Player, make_player, play_series, random_player

CALIBRATION_PATH = Path("runs/lab/ladder-calibration.json")
MAX_MEASURABLE_SCORE = 0.99  # a 99-1 series is the most a 100-game match can resolve


@dataclass(frozen=True)
class Rung:
    name: str
    engine_probability: float  # fraction of moves played by Stockfish; the rest random
    nodes: int  # Stockfish search budget per move


LADDER = [
    Rung("random", 0.0, 0),
    Rung("sf1-mix25", 0.25, 1),
    Rung("sf1-mix50", 0.50, 1),
    Rung("sf1-mix75", 0.75, 1),
    Rung("sf1", 1.0, 1),
    Rung("sf4", 1.0, 4),
    Rung("sf16", 1.0, 16),
    Rung("sf64", 1.0, 64),
]


def rung_player(rung: Rung, engine: chess.engine.SimpleEngine, rng: random.Random) -> Player:
    def move(board: chess.Board) -> chess.Move:
        if rung.engine_probability and rng.random() < rung.engine_probability:
            played = engine.play(board, chess.engine.Limit(nodes=rung.nodes)).move
            if played is not None:
                return played
        return rng.choice(list(board.legal_moves))

    return move


def expected_score(rating_difference: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def gap_from_score(score: float) -> float:
    clamped = min(max(score, 1.0 - MAX_MEASURABLE_SCORE), MAX_MEASURABLE_SCORE)
    return 400.0 * math.log10(clamped / (1.0 - clamped))


def fit_rating(observations: list[tuple[float, float, int]]) -> dict[str, float]:
    """Maximum-likelihood lab rating from (rung_rating, score_points, games) rows."""

    def log_likelihood(rating: float) -> float:
        total = 0.0
        for rung_rating, points, games in observations:
            probability = min(max(expected_score(rating - rung_rating), 1e-9), 1 - 1e-9)
            total += points * math.log(probability)
            total += (games - points) * math.log(1.0 - probability)
        return total

    grid = [x / 2 for x in range(-2000, 6001)]  # −1000 to 3000 in half-point steps
    values = [log_likelihood(rating) for rating in grid]
    best = max(range(len(grid)), key=values.__getitem__)
    threshold = values[best] - 1.92  # 95% profile-likelihood interval
    inside = [grid[i] for i in range(len(grid)) if values[i] >= threshold]
    return {"rating": grid[best], "low": min(inside), "high": max(inside)}


def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    engine.configure({"Skill Level": 0})
    return engine


def calibrate(games: int, seed: int) -> None:
    rng = random.Random(seed)
    engine = open_engine()
    ratings = {LADDER[0].name: 0.0}
    pairs = []
    try:
        for lower, upper in zip(LADDER, LADDER[1:]):
            tally = play_series(
                rung_player(upper, engine, rng), rung_player(lower, engine, rng), games, rng
            )
            score = (tally["win"] + 0.5 * tally["draw"]) / games
            gap = gap_from_score(score)
            ratings[upper.name] = ratings[lower.name] + gap
            pairs.append({"upper": upper.name, "lower": lower.name, **tally, "score": score, "gap": round(gap, 1)})
            print(json.dumps(pairs[-1]))
    finally:
        engine.quit()
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps({"ratings": {k: round(v, 1) for k, v in ratings.items()}, "pairs": pairs, "games_per_pair": games, "seed": seed}, indent=2) + "\n"
    )
    print(json.dumps({"ratings": {k: round(v, 1) for k, v in ratings.items()}}))


def rate(
    checkpoint: Path,
    probe_games: int,
    full_games: int,
    seed: int,
    temperature: float = 0.0,
    top_k: int = 0,
    search: str = "none",
    contempt: float = 0.0,
    tag: str = "",
) -> None:
    calibration = json.loads(CALIBRATION_PATH.read_text())
    ratings = calibration["ratings"]
    rng = random.Random(seed)
    engine = open_engine()
    candidate = make_player(checkpoint, temperature, top_k, search, contempt, seed)
    per_rung: dict[str, dict[str, int]] = {}
    try:
        for rung in LADDER:
            opponent = (
                random_player(rng) if rung.engine_probability == 0 else rung_player(rung, engine, rng)
            )
            tally = play_series(candidate, opponent, probe_games, rng)
            per_rung[rung.name] = tally
            score = (tally["win"] + 0.5 * tally["draw"]) / probe_games
            print(json.dumps({"rung": rung.name, "phase": "probe", **tally, "score": score}))
            if score < 0.30:
                break
        informative = [
            name
            for name, tally in per_rung.items()
            if 0.05 <= (tally["win"] + 0.5 * tally["draw"]) / sum(tally.values()) <= 0.95
        ]
        for rung in LADDER:
            if rung.name not in informative:
                continue
            opponent = (
                random_player(rng) if rung.engine_probability == 0 else rung_player(rung, engine, rng)
            )
            extra = play_series(candidate, opponent, full_games - probe_games, rng)
            for key in per_rung[rung.name]:
                per_rung[rung.name][key] += extra[key]
            print(json.dumps({"rung": rung.name, "phase": "full", **per_rung[rung.name]}))
    finally:
        engine.quit()

    observations = [
        (ratings[name], tally["win"] + 0.5 * tally["draw"], sum(tally.values()))
        for name, tally in per_rung.items()
    ]
    fitted = fit_rating(observations)
    decode = {"temperature": temperature, "top_k": top_k, "search": search, "contempt": contempt}
    report = {"checkpoint": str(checkpoint), "decode": decode, "per_rung": per_rung, **fitted, "seed": seed}
    output = checkpoint.with_suffix(f".{tag}.rating.json" if tag else ".rating.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"lab_rating": fitted["rating"], "ci95": [fitted["low"], fitted["high"]], "written": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "rate"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probe-games", type=int, default=20)
    parser.add_argument("--full-games", type=int, default=100)
    parser.add_argument("--calibration-games", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--search", choices=("none", "value1"), default="none")
    parser.add_argument("--contempt", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.calibration_games, args.seed)
    else:
        if args.checkpoint is None:
            raise SystemExit("rate requires --checkpoint")
        rate(
            args.checkpoint, args.probe_games, args.full_games, args.seed,
            args.temperature, args.top_k, args.search, args.contempt, args.tag,
        )


if __name__ == "__main__":
    main()
