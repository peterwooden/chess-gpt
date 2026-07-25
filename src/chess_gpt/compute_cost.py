"""Calculate the rental cost of a training-FLOP budget from sourced inputs."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

FLOPS_PER_TFLOP = Decimal("1e12")
SECONDS_PER_HOUR = Decimal(3600)
DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[2] / "data" / "compute-cost-snapshot.toml"
)


@dataclass(frozen=True)
class Accelerator:
    provider: str
    gpu: str
    hourly_price_usd: Decimal
    dense_bf16_tflops: Decimal


def rental_hours(
    budget_flops: Decimal, dense_bf16_tflops: Decimal, utilization: Decimal
) -> Decimal:
    """Return GPU-hours required at a stated fraction of dense BF16 peak."""
    if budget_flops <= 0 or dense_bf16_tflops <= 0:
        raise ValueError("budget and throughput must be positive")
    if not Decimal(0) < utilization <= Decimal(1):
        raise ValueError("utilization must be greater than 0 and at most 1")
    flops_per_hour = dense_bf16_tflops * FLOPS_PER_TFLOP * SECONDS_PER_HOUR
    return budget_flops / (flops_per_hour * utilization)


def rental_cost_usd(
    budget_flops: Decimal,
    dense_bf16_tflops: Decimal,
    hourly_price_usd: Decimal,
    utilization: Decimal,
) -> Decimal:
    return rental_hours(budget_flops, dense_bf16_tflops, utilization) * hourly_price_usd


def load_snapshot(path: Path) -> tuple[Decimal, list[Accelerator], str, str]:
    with path.open("rb") as file:
        raw = tomllib.load(file)
    snapshot = raw["snapshot"]
    accelerators = [
        Accelerator(
            provider=item["provider"],
            gpu=item["gpu"],
            hourly_price_usd=Decimal(item["hourly_price_usd"]),
            dense_bf16_tflops=Decimal(item["dense_bf16_tflops"]),
        )
        for item in raw["accelerators"]
    ]
    return (
        Decimal(snapshot["budget_flops"]),
        accelerators,
        snapshot["observed_at"],
        snapshot["notes"],
    )


def render_table(
    budget_flops: Decimal,
    accelerators: Sequence[Accelerator],
    utilizations: Sequence[Decimal],
) -> str:
    lines = [
        "| Provider / accelerator | Utilization | GPU-hours | Cost (USD) |",
        "|---|---:|---:|---:|",
    ]
    for accelerator in accelerators:
        for utilization in utilizations:
            hours = rental_hours(
                budget_flops, accelerator.dense_bf16_tflops, utilization
            )
            cost = hours * accelerator.hourly_price_usd
            lines.append(
                f"| {accelerator.provider} / {accelerator.gpu} "
                f"| {utilization * 100:.0f}% | {hours:.3f} | ${cost:.2f} |"
            )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument(
        "--utilization",
        nargs="+",
        type=Decimal,
        default=[Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0.1")],
        help="Fractions of dense BF16 peak to calculate (default: 1 .5 .25 .1)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    budget, accelerators, observed_at, notes = load_snapshot(args.snapshot)
    print(f"Inputs observed: {observed_at}; training budget: {budget:.0E} FLOPs")
    print(render_table(budget, accelerators, args.utilization))
    print(f"\n{notes}")


if __name__ == "__main__":
    main()
