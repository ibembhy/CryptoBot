from datetime import datetime, timezone
import unittest

from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel, terminal_probability_above
from kalshi_btc_bot.types import MarketSnapshot


def make_snapshot(threshold: float) -> MarketSnapshot:
    return MarketSnapshot(
        source="test",
        series_ticker="KXBTC",
        market_ticker=f"KXBTC-{threshold}",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
        expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
        spot_price=64000,
        threshold=threshold,
        direction="above",
    )


class GBMThresholdTests(unittest.TestCase):
    def test_probability_monotonicity_with_threshold(self):
        low = terminal_probability_above(64000, 64500, 1 / 365, 0.8)
        high = terminal_probability_above(64000, 66000, 1 / 365, 0.8)
        self.assertGreater(low, high)

    def test_zero_time_collapses_to_boundary(self):
        probability = terminal_probability_above(64000, 65000, 0.0, 0.5)
        self.assertEqual(probability, 0.0)

    def test_higher_vol_can_raise_otm_upside_probability(self):
        snapshot = make_snapshot(66000)
        model = GBMThresholdModel()
        low = model.estimate(snapshot, 0.3).probability
        high = model.estimate(snapshot, 0.9).probability
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
