from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.dashboard import build_latest_snapshot_table, build_live_signal_table, latest_snapshots_by_market
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.risk import RiskConfig
from kalshi_btc_bot.types import MarketSnapshot


class DashboardTests(unittest.TestCase):
    def test_latest_snapshots_by_market_keeps_latest(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        first = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=30),
            spot_price=67000,
            threshold=67100,
            direction="above",
            yes_bid=0.40,
            yes_ask=0.41,
        )
        second = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at + timedelta(minutes=1),
            expiry=observed_at + timedelta(minutes=30),
            spot_price=67100,
            threshold=67100,
            direction="above",
            yes_bid=0.60,
            yes_ask=0.61,
        )
        latest = latest_snapshots_by_market([first, second])
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].spot_price, 67100)

    def test_build_live_signal_table_returns_rows(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        expiry = observed_at + timedelta(minutes=15)
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-TEST",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=observed_at,
                expiry=expiry,
                spot_price=67000,
                threshold=67050,
                direction="above",
                yes_bid=0.40,
                yes_ask=0.41,
                no_bid=0.59,
                no_ask=0.60,
            ),
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-TEST",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=observed_at + timedelta(minutes=5),
                expiry=expiry,
                spot_price=67100,
                threshold=67050,
                direction="above",
                yes_bid=0.72,
                yes_ask=0.73,
                no_bid=0.26,
                no_ask=0.27,
            ),
        ]
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.01, 0.0, 25, 80, 150.0, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 3, 5),
            backtest_config=BacktestConfig(1, 1, 1, 0.0, 1000.0),
            risk_config=RiskConfig(100.0, 1000.0, 5, 5, 100.0, 100.0),
        )
        table = build_live_signal_table(
            engine=engine,
            snapshots=snapshots,
            volatility_window=1,
            annualization_factor=365.0 * 24.0 * 60.0,
        )
        self.assertFalse(table.empty)
        self.assertIn("market_ticker", table.columns)
        self.assertIn("action", table.columns)

    def test_build_latest_snapshot_table(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-TEST",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=15),
            spot_price=67000,
            threshold=67050,
            direction="above",
            yes_bid=0.40,
            yes_ask=0.41,
        )
        table = build_latest_snapshot_table([snapshot])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["market_ticker"], "KXBTCD-TEST")


if __name__ == "__main__":
    unittest.main()
