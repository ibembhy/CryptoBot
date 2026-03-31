from datetime import datetime, timedelta, timezone
import unittest

from kalshi_btc_bot.trading.exits import ExitConfig, evaluate_exit
from kalshi_btc_bot.types import MarketSnapshot


class ExitRuleTests(unittest.TestCase):
    def test_take_profit_exit_trigger(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-1",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 30, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64500,
            threshold=65000,
            direction="above",
            yes_bid=0.58,
            yes_ask=0.59,
        )
        decision = evaluate_exit(
            snapshot=snapshot,
            side="yes",
            entry_price_cents=48,
            contracts=10,
            fair_value_cents=60,
            model_probability=0.60,
            config=ExitConfig(8, 10, 3, 5),
            as_of=snapshot.observed_at,
        )
        self.assertEqual(decision.trigger, "take_profit")

    def test_time_exit_trigger(self):
        observed_at = datetime(2026, 3, 30, 15, 57, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-2",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=2),
            spot_price=64500,
            threshold=65000,
            direction="above",
            yes_bid=0.49,
            yes_ask=0.50,
        )
        decision = evaluate_exit(
            snapshot=snapshot,
            side="yes",
            entry_price_cents=48,
            contracts=10,
            fair_value_cents=70,
            model_probability=0.70,
            config=ExitConfig(8, 10, 1, 5),
            as_of=observed_at,
        )
        self.assertEqual(decision.trigger, "time_exit")

    def test_value_based_hold_can_override_take_profit(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-3",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 20, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64950,
            threshold=65000,
            direction="above",
            yes_bid=0.60,
            yes_ask=0.61,
            volume=500.0,
        )
        decision = evaluate_exit(
            snapshot=snapshot,
            side="yes",
            entry_price_cents=50,
            contracts=10,
            fair_value_cents=70,
            model_probability=0.70,
            config=ExitConfig(8, 10, 2, 5, min_hold_edge=0.02, min_ev_to_hold_cents=3),
            as_of=snapshot.observed_at,
        )
        self.assertEqual(decision.action, "hold")


if __name__ == "__main__":
    unittest.main()
