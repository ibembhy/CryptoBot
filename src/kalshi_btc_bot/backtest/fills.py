from __future__ import annotations


def apply_entry_fill(price_cents: int, slippage_cents: int, fee_rate_bps: float) -> tuple[int, float]:
    slipped = min(price_cents + slippage_cents, 99)
    fees = round((slipped / 100.0) * (fee_rate_bps / 10000.0), 4)
    return slipped, fees


def apply_exit_fill(price_cents: int, slippage_cents: int, fee_rate_bps: float) -> tuple[int, float]:
    slipped = max(price_cents - slippage_cents, 1)
    fees = round((slipped / 100.0) * (fee_rate_bps / 10000.0), 4)
    return slipped, fees
