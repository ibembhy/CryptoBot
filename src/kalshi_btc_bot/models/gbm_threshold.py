from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_btc_bot.models.base import ProbabilityModel
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate
from kalshi_btc_bot.utils.math import clamp, norm_cdf


def terminal_probability_above(
    spot_price: float,
    target_price: float,
    time_to_expiry_years: float,
    volatility: float,
    drift: float = 0.0,
) -> float:
    if target_price <= 0:
        raise ValueError("target_price must be positive")
    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    if time_to_expiry_years <= 0:
        return 1.0 if spot_price >= target_price else 0.0
    sigma = max(volatility, 1e-12)
    denom = sigma * math.sqrt(time_to_expiry_years)
    numerator = math.log(spot_price / target_price) + (drift - 0.5 * sigma * sigma) * time_to_expiry_years
    return clamp(norm_cdf(numerator / denom), 0.0, 1.0)


def probability_for_snapshot(
    snapshot: MarketSnapshot,
    *,
    spot_price: float,
    volatility: float,
    drift: float,
) -> float:
    t = snapshot.time_to_expiry_years
    if snapshot.contract_type == "threshold":
        if snapshot.threshold is None:
            raise ValueError("threshold contract missing threshold")
        base = terminal_probability_above(spot_price, snapshot.threshold, t, volatility, drift)
        if snapshot.direction == "below":
            return 1.0 - base
        return base
    if snapshot.contract_type == "range":
        if snapshot.range_low is None or snapshot.range_high is None:
            raise ValueError("range contract missing bounds")
        lower = terminal_probability_above(spot_price, snapshot.range_low, t, volatility, drift)
        upper = terminal_probability_above(spot_price, snapshot.range_high, t, volatility, drift)
        return clamp(lower - upper, 0.0, 1.0)
    if snapshot.contract_type == "direction":
        reference = snapshot.threshold if snapshot.threshold is not None else spot_price
        above = terminal_probability_above(spot_price, reference, t, volatility, drift)
        if snapshot.direction == "down":
            return 1.0 - above
        return above
    raise ValueError(f"Unsupported contract type: {snapshot.contract_type}")


@dataclass
class GBMThresholdModel(ProbabilityModel):
    drift: float = 0.0
    volatility_floor: float = 0.05
    model_name: str = "gbm_threshold"

    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        sigma = max(volatility, self.volatility_floor)
        probability = self._probability_for_snapshot(snapshot, sigma)
        target = snapshot.threshold
        if snapshot.contract_type == "range":
            target = snapshot.range_high if snapshot.range_high is not None else snapshot.range_low
        elif snapshot.contract_type == "direction":
            target = snapshot.threshold if snapshot.threshold is not None else snapshot.spot_price
        return ProbabilityEstimate(
            model_name=self.model_name,
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=float(target if target is not None else snapshot.spot_price),
            volatility=sigma,
            drift=self.drift,
            probability=probability,
            inputs={
                "contract_type": snapshot.contract_type,
                "threshold": snapshot.threshold,
                "range_low": snapshot.range_low,
                "range_high": snapshot.range_high,
                "direction": snapshot.direction,
            },
        )

    def _probability_for_snapshot(self, snapshot: MarketSnapshot, volatility: float) -> float:
        return probability_for_snapshot(
            snapshot,
            spot_price=snapshot.spot_price,
            volatility=volatility,
            drift=self.drift,
        )
