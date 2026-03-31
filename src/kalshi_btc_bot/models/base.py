from __future__ import annotations

from abc import ABC, abstractmethod

from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate


class ProbabilityModel(ABC):
    @abstractmethod
    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        raise NotImplementedError
