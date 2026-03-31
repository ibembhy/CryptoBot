from __future__ import annotations

import math


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def probability_to_cents(probability: float) -> int:
    return int(round(clamp(probability, 0.0, 1.0) * 100.0))
