from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FillResult:
    price_cents: int
    fees_paid: float
    slippage_cents: int


def estimate_dynamic_slippage_cents(
    base_slippage_cents: int,
    *,
    spread_cents: int | None = None,
    volume: float | None = None,
    open_interest: float | None = None,
    contracts: int = 1,
    max_extra_cents: int = 6,
) -> int:
    spread_extra = 0
    if spread_cents is not None:
        spread_extra = max(0, math.ceil(max(spread_cents - 1, 0) * 0.5))

    liquidity_extra = 0
    if volume is not None:
        if volume < max(25.0, contracts * 25):
            liquidity_extra += 2
        elif volume < max(100.0, contracts * 50):
            liquidity_extra += 1

    if open_interest is not None:
        if open_interest < max(25.0, contracts * 25):
            liquidity_extra += 1

    extra = min(max_extra_cents, spread_extra + liquidity_extra)
    return max(base_slippage_cents + extra, 0)


def apply_entry_fill(
    price_cents: int,
    slippage_cents: int,
    fee_rate_bps: float,
    *,
    spread_cents: int | None = None,
    volume: float | None = None,
    open_interest: float | None = None,
    contracts: int = 1,
) -> FillResult:
    effective_slippage = estimate_dynamic_slippage_cents(
        slippage_cents,
        spread_cents=spread_cents,
        volume=volume,
        open_interest=open_interest,
        contracts=contracts,
    )
    slipped = min(price_cents + effective_slippage, 99)
    fees = round(((slipped * contracts) / 100.0) * (fee_rate_bps / 10000.0), 4)
    return FillResult(price_cents=slipped, fees_paid=fees, slippage_cents=effective_slippage)


def apply_exit_fill(
    price_cents: int,
    slippage_cents: int,
    fee_rate_bps: float,
    *,
    spread_cents: int | None = None,
    volume: float | None = None,
    open_interest: float | None = None,
    contracts: int = 1,
) -> FillResult:
    effective_slippage = estimate_dynamic_slippage_cents(
        slippage_cents,
        spread_cents=spread_cents,
        volume=volume,
        open_interest=open_interest,
        contracts=contracts,
    )
    slipped = max(price_cents - effective_slippage, 1)
    fees = round(((slipped * contracts) / 100.0) * (fee_rate_bps / 10000.0), 4)
    return FillResult(price_cents=slipped, fees_paid=fees, slippage_cents=effective_slippage)
