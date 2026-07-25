from decimal import Decimal

import pytest

from chess_gpt.compute_cost import rental_cost_usd, rental_hours


def test_h100_peak_cost_for_tournament_budget() -> None:
    cost = rental_cost_usd(
        budget_flops=Decimal("1e18"),
        dense_bf16_tflops=Decimal("989.5"),
        hourly_price_usd=Decimal("2.99"),
        utilization=Decimal(1),
    )

    assert cost == pytest.approx(Decimal("0.8393"), rel=Decimal("0.001"))


def test_half_utilization_doubles_hours() -> None:
    peak_hours = rental_hours(Decimal("1e18"), Decimal("312"), Decimal(1))
    half_utilization_hours = rental_hours(
        Decimal("1e18"), Decimal("312"), Decimal("0.5")
    )

    assert half_utilization_hours == peak_hours * 2


@pytest.mark.parametrize("utilization", [Decimal(0), Decimal("1.01")])
def test_utilization_must_be_a_fraction(utilization: Decimal) -> None:
    with pytest.raises(ValueError, match="utilization"):
        rental_hours(Decimal("1e18"), Decimal("312"), utilization)
