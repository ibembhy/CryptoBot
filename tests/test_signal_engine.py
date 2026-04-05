from datetime import datetime, timezone
import unittest

from kalshi_btc_bot.signals.engine import SignalConfig, generate_signal
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate


class SignalEngineTests(unittest.TestCase):
    def test_signal_engine_prefers_yes_with_positive_edge(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-1",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_ask=0.40,
            yes_bid=0.39,
            no_ask=0.62,
            no_bid=0.60,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=64000,
            target_price=65000,
            volatility=0.8,
            drift=0.0,
            probability=0.55,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(0.05, 0.0, 5, 95, uncertainty_penalty=0.02, max_spread_cents=10, max_data_age_seconds=10**9),
        )
        self.assertEqual(signal.action, "buy_yes")
        self.assertEqual(signal.side, "yes")
        self.assertAlmostEqual(signal.model_probability, 0.53, places=2)

    def test_signal_engine_can_reject_wide_spread_market(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-2",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_ask=0.40,
            yes_bid=0.20,
            no_ask=0.80,
            no_bid=0.60,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=64000,
            target_price=65000,
            volatility=0.8,
            drift=0.0,
            probability=0.60,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(0.05, 0.0, 5, 95, uncertainty_penalty=0.0, max_spread_cents=5, max_data_age_seconds=10**9),
        )
        self.assertEqual(signal.action, "no_action")

    def test_signal_engine_can_reject_far_from_spot_contract(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-3",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 15, 20, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=66000,
            direction="above",
            yes_ask=0.40,
            yes_bid=0.39,
            no_ask=0.62,
            no_bid=0.60,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=64000,
            target_price=66000,
            volatility=0.8,
            drift=0.0,
            probability=0.60,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(0.05, 0.0, 5, 95, max_near_money_bps=150.0, max_data_age_seconds=10**9),
        )
        self.assertEqual(signal.action, "no_action")
        self.assertEqual(signal.reason, "Contract is too far from spot.")

    def test_kxbtc15m_prefers_tier_a_and_marks_signal(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC15M",
            market_ticker="KXBTC15M-TEST-A",
            contract_type="direction",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 15, 12, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=64025,
            direction="above",
            yes_ask=0.50,
            yes_bid=0.48,
            no_ask=0.52,
            no_bid=0.50,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=64000,
            target_price=64025,
            volatility=0.8,
            drift=0.0,
            probability=0.62,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(
                0.05,
                0.0,
                25,
                60,
                uncertainty_penalty=0.0,
                max_spread_cents=10,
                max_data_age_seconds=10**9,
                series_tier_profiles={
                    "KXBTC15M": {
                        "allowed_sides": ["yes"],
                        "preferred_price_min_cents": 45,
                        "preferred_price_max_cents": 55,
                        "secondary_price_min_cents": 35,
                        "secondary_price_max_cents": 45,
                        "soft_price_max_cents": 60,
                        "preferred_minutes_min": 10,
                        "preferred_minutes_max": 15,
                        "soft_minutes_min": 5,
                        "disable_tier_d": True,
                    }
                },
            ),
        )
        self.assertEqual(signal.action, "buy_yes")
        self.assertEqual(signal.tier_label, "A")
        self.assertAlmostEqual(signal.size_multiplier, 1.0, places=3)
        self.assertIn("Tier A", signal.reason)

    def test_kxbtc15m_disables_experimental_tier_d(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC15M",
            market_ticker="KXBTC15M-TEST-D",
            contract_type="direction",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 15, 14, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=64025,
            direction="above",
            yes_ask=0.30,
            yes_bid=0.28,
            no_ask=0.72,
            no_bid=0.70,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=64000,
            target_price=64025,
            volatility=0.8,
            drift=0.0,
            probability=0.60,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(
                0.05,
                0.0,
                25,
                60,
                uncertainty_penalty=0.0,
                max_spread_cents=10,
                max_data_age_seconds=10**9,
                series_tier_profiles={
                    "KXBTC15M": {
                        "allowed_sides": ["yes"],
                        "preferred_price_min_cents": 45,
                        "preferred_price_max_cents": 55,
                        "secondary_price_min_cents": 35,
                        "secondary_price_max_cents": 45,
                        "soft_price_max_cents": 60,
                        "preferred_minutes_min": 10,
                        "preferred_minutes_max": 15,
                        "soft_minutes_min": 5,
                        "disable_tier_d": True,
                    }
                },
            ),
        )
        self.assertEqual(signal.action, "no_action")
        self.assertEqual(signal.tier_label, "D")
        self.assertIn("experimental", signal.reason.lower())


if __name__ == "__main__":
    unittest.main()
