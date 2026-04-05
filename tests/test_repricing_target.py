from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.models.repricing_target import RepricingTargetModel, train_repricing_profile
from kalshi_btc_bot.types import MarketSnapshot


class RepricingTargetModelTests(unittest.TestCase):
    def test_train_repricing_profile_and_estimate_future_move(self):
        observed_at = datetime(2026, 3, 31, 14, 0, tzinfo=timezone.utc)
        snapshots: list[MarketSnapshot] = []
        feature_rows = []
        feature_index = []

        for market_index in range(3):
            market = f"KXBTCD-REP-{market_index}"
            for step in range(3):
                ts = observed_at + timedelta(minutes=step)
                yes_bid = 0.39 + 0.03 * step + market_index * 0.005
                yes_ask = yes_bid + 0.02
                no_ask = 1.0 - yes_bid
                no_bid = no_ask - 0.02
                snapshots.append(
                    MarketSnapshot(
                        source="test",
                        series_ticker="KXBTCD",
                        market_ticker=market,
                        contract_type="threshold",
                        underlying_symbol="BTC-USD",
                        observed_at=ts,
                        expiry=observed_at + timedelta(minutes=20),
                        spot_price=67000.0 + step * 25.0,
                        threshold=67050.0,
                        direction="above",
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        volume=1200.0,
                        open_interest=800.0,
                    )
                )
                feature_index.append(pd.Timestamp(ts))
                feature_rows.append(
                    {
                        "close": 67000.0 + step * 25.0,
                        "log_return": 0.001 * (step + 1),
                        "log_return_3": 0.0015 * (step + 1),
                        "log_return_5": 0.002 * (step + 1),
                        "price_acceleration": 0.0004 * (step + 1),
                        "realized_volatility": 0.75,
                        "vol_of_vol": 0.12,
                    }
                )

        feature_frame = pd.DataFrame(feature_rows, index=feature_index)
        profile = train_repricing_profile(
            snapshots,
            feature_frame=feature_frame,
            horizon_minutes=1,
            min_samples=3,
            min_regime_samples=2,
            ridge_lambda=1.0,
            max_abs_target_cents=20.0,
            near_money_regime_bps=100.0,
            high_vol_regime_threshold=0.7,
            profit_cost_cents=1.0,
        )
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertGreaterEqual(profile.sample_count, 3)

        candidate = snapshots[0]
        candidate.metadata.update(
            {
                "recent_log_return": 0.002,
                "log_return_3": 0.003,
                "log_return_5": 0.004,
                "price_acceleration": 0.001,
                "realized_volatility": 0.75,
                "vol_of_vol": 0.12,
            }
        )
        current_mid = (candidate.yes_bid + candidate.yes_ask) / 2.0
        model = RepricingTargetModel(profile=profile)
        estimate = model.estimate(candidate, volatility=0.75)
        self.assertGreater(estimate.probability, current_mid)
        self.assertIn("predicted_delta_cents", estimate.inputs)
        self.assertIn("repricing_regime", estimate.inputs)
        self.assertIn("profit_probability_yes", estimate.inputs)
        self.assertIn("profit_probability_no", estimate.inputs)


if __name__ == "__main__":
    unittest.main()
