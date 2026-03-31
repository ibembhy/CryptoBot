from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.signals.calibration import fit_probability_calibrator
from kalshi_btc_bot.signals.engine import SignalConfig, generate_signal
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate


class SignalCalibrationTests(unittest.TestCase):
    def test_fit_probability_calibrator_builds_monotonic_mapping(self):
        trades = pd.DataFrame(
            {
                "raw_model_probability": [0.2, 0.25, 0.3, 0.55, 0.6, 0.65, 0.8, 0.85, 0.9],
                "contract_won": [0, 0, 0, 1, 0, 1, 1, 1, 1],
            }
        )
        calibrator = fit_probability_calibrator(trades, min_samples=5, min_bucket_count=1, bucket_width=0.2)
        self.assertIsNotNone(calibrator)
        assert calibrator is not None
        self.assertLessEqual(calibrator.apply(0.25), calibrator.apply(0.65))

    def test_generate_signal_preserves_raw_probability_and_uses_calibrated_probability(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-CAL",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 31, 16, 20, tzinfo=timezone.utc),
            spot_price=67000.0,
            threshold=67100.0,
            direction="above",
            yes_bid=0.37,
            yes_ask=0.39,
            no_bid=0.60,
            no_ask=0.62,
            volume=1000.0,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm_threshold",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=67100.0,
            volatility=0.8,
            drift=0.0,
            probability=0.58,
            raw_probability=0.66,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(0.01, 0.0, 25, 80, uncertainty_penalty=0.0, max_spread_cents=10, max_data_age_seconds=10**9),
        )
        self.assertAlmostEqual(signal.raw_model_probability, 0.66, places=4)
        self.assertAlmostEqual(signal.model_probability, 0.58, places=4)


if __name__ == "__main__":
    unittest.main()
