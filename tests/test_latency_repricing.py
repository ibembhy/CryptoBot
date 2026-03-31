from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from kalshi_btc_bot.models.latency_repricing import LatencyRepricingModel
from kalshi_btc_bot.types import MarketSnapshot


class LatencyRepricingTests(unittest.TestCase):
    def _snapshot(self) -> MarketSnapshot:
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        expiry = observed_at + timedelta(minutes=15)
        return MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=expiry,
            spot_price=67000.0,
            threshold=67100.0,
            direction="above",
            yes_bid=0.45,
            yes_ask=0.46,
            metadata={},
        )

    def test_positive_recent_move_increases_above_probability(self):
        model = LatencyRepricingModel()
        snapshot = self._snapshot()
        base = model.estimate(snapshot, volatility=0.8).probability
        moved = self._snapshot()
        moved.metadata["recent_log_return"] = 0.004
        shifted = model.estimate(moved, volatility=0.8).probability
        self.assertGreater(shifted, base)

    def test_small_move_below_threshold_keeps_probability_near_base(self):
        model = LatencyRepricingModel(min_move_bps=10.0)
        snapshot = self._snapshot()
        base = model.estimate(snapshot, volatility=0.8).probability
        moved = self._snapshot()
        moved.metadata["recent_log_return"] = 0.0003
        shifted = model.estimate(moved, volatility=0.8).probability
        self.assertAlmostEqual(base, shifted, places=8)


if __name__ == "__main__":
    unittest.main()
