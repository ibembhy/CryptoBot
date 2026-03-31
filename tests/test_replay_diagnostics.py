from datetime import datetime, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.types import MarketSnapshot


class ReplayDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_reports_missing_settlement(self):
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-65000",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
                expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
                spot_price=64800,
                threshold=65000,
                direction="above",
                yes_bid=0.44,
                yes_ask=0.45,
                no_bid=0.54,
                no_ask=0.55,
            )
        ]
        features = pd.DataFrame(
            {"realized_volatility": [0.9]},
            index=pd.to_datetime(["2026-03-30T15:00:00Z"]),
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.05, 0.0, 5, 95, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
        )
        diagnostics = engine.diagnose_replay(snapshots, features)
        self.assertEqual(diagnostics.summary["skipped_no_settlement"], 1)


if __name__ == "__main__":
    unittest.main()
