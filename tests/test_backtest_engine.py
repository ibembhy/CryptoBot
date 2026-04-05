from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine, MakerSimulationConfig
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.types import MarketSnapshot


def build_snapshots():
    observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
    expiry = datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc)
    base = dict(
        source="test",
        series_ticker="KXBTC",
        market_ticker="KXBTC-65000",
        contract_type="threshold",
        underlying_symbol="BTC-USD",
        expiry=expiry,
        threshold=65000,
        direction="above",
    )
    return [
        MarketSnapshot(observed_at=observed_at, spot_price=64800, yes_bid=0.29, yes_ask=0.30, no_bid=0.69, no_ask=0.70, **base),
        MarketSnapshot(observed_at=observed_at + timedelta(minutes=20), spot_price=65150, yes_bid=0.56, yes_ask=0.57, no_bid=0.42, no_ask=0.43, **base),
        MarketSnapshot(observed_at=expiry, spot_price=64900, settlement_price=64900, yes_bid=0.01, yes_ask=0.02, no_bid=0.98, no_ask=0.99, **base),
    ]


class BacktestEngineTests(unittest.TestCase):
    def test_backtest_compare_strategies_shows_different_outcomes(self):
        features = pd.DataFrame(
            {"realized_volatility": [0.9, 0.9]},
            index=pd.to_datetime(["2026-03-30T15:00:00Z", "2026-03-30T15:20:00Z"]),
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.05, 0.0, 5, 95, uncertainty_penalty=0.0, max_spread_cents=20, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
        )
        comparison = engine.compare_strategies(build_snapshots(), features)
        early = comparison["early_exit"]
        hold = comparison["hold_to_settlement"]
        self.assertNotEqual(early.summary["pnl"], hold.summary["pnl"])
        self.assertEqual(early.trades.iloc[0]["exit_trigger"], "take_profit")
        self.assertIn("raw_edge", early.trades.columns)

    def test_binary_settlement_result_is_used_directly(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=68000,
            threshold=67500,
            direction="above",
            settlement_price=0.0,
        )
        self.assertFalse(BacktestEngine._contract_wins(snapshot, "yes"))
        self.assertTrue(BacktestEngine._contract_wins(snapshot, "no"))

    def test_maker_sim_can_fill_conservatively_when_quote_is_touched(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        expiry = datetime(2026, 3, 30, 15, 10, tzinfo=timezone.utc)
        base = dict(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            expiry=expiry,
            threshold=65000,
            direction="above",
        )
        snapshots = [
            MarketSnapshot(observed_at=observed_at, spot_price=64000, yes_bid=0.05, yes_ask=0.06, no_bid=0.94, no_ask=0.95, volume=600, open_interest=600, **base),
            MarketSnapshot(observed_at=observed_at + timedelta(seconds=10), spot_price=64020, yes_bid=0.05, yes_ask=0.06, no_bid=0.93, no_ask=0.94, volume=600, open_interest=600, **base),
            MarketSnapshot(observed_at=expiry, spot_price=64050, settlement_price=0.0, yes_bid=0.0, yes_ask=0.01, no_bid=0.99, no_ask=1.0, volume=600, open_interest=600, **base),
        ]
        features = pd.DataFrame(
            {"realized_volatility": [0.9, 0.9], "btc_micro_jump_flag": [0.0, 0.0]},
            index=pd.to_datetime([observed_at, observed_at + timedelta(seconds=10)]),
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.04, 0.0, 5, 95, uncertainty_penalty=0.0, max_spread_cents=20, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
            maker_simulation_config=MakerSimulationConfig(max_wait_seconds=30, min_fill_probability=0.55, stale_quote_age_seconds=20),
        )
        result = engine.run_strategy_with_entry_style("hold_to_settlement", snapshots, features, entry_style="maker_sim")
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["entry_style"], "maker_sim")
        self.assertGreater(float(result.trades.iloc[0]["maker_fill_probability"]), 0.55)

    def test_maker_sim_cancels_when_quote_goes_stale(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        expiry = datetime(2026, 3, 30, 15, 10, tzinfo=timezone.utc)
        base = dict(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-STALE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            expiry=expiry,
            threshold=65000,
            direction="above",
        )
        snapshots = [
            MarketSnapshot(observed_at=observed_at, spot_price=64000, yes_bid=0.05, yes_ask=0.06, no_bid=0.94, no_ask=0.95, volume=600, open_interest=600, **base),
            MarketSnapshot(observed_at=observed_at + timedelta(minutes=2), spot_price=64020, yes_bid=0.05, yes_ask=0.06, no_bid=0.94, no_ask=0.95, volume=600, open_interest=600, **base),
            MarketSnapshot(observed_at=expiry, spot_price=64050, settlement_price=0.0, yes_bid=0.0, yes_ask=0.01, no_bid=0.99, no_ask=1.0, volume=600, open_interest=600, **base),
        ]
        features = pd.DataFrame(
            {"realized_volatility": [0.9, 0.9], "btc_micro_jump_flag": [0.0, 0.0]},
            index=pd.to_datetime([observed_at, observed_at + timedelta(minutes=2)]),
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.04, 0.0, 5, 95, uncertainty_penalty=0.0, max_spread_cents=20, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
            maker_simulation_config=MakerSimulationConfig(max_wait_seconds=180, min_fill_probability=0.55, stale_quote_age_seconds=20),
        )
        result = engine.run_strategy_with_entry_style("hold_to_settlement", snapshots, features, entry_style="maker_sim")
        self.assertTrue(result.trades.empty)


if __name__ == "__main__":
    unittest.main()
