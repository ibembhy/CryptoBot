from datetime import datetime, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.reports.calibration import build_calibration_report
from kalshi_btc_bot.reports.sensitivity import run_volatility_sensitivity
from kalshi_btc_bot.reports.summary import build_summary_report
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.types import BacktestResult, MarketSnapshot


class ReportingTests(unittest.TestCase):
    def test_summary_report_includes_edge_distribution(self):
        trades = pd.DataFrame(
            {
                "edge": [0.06, 0.08],
                "raw_edge": [0.08, 0.11],
                "realized_pnl": [1.0, -0.5],
                "entry_notional": [0.4, 0.5],
                "hold_minutes": [10.0, 20.0],
                "exit_improved_result": [True, False],
                "clv_cents": [4.0, -1.0],
                "entry_time": pd.to_datetime(["2026-03-30T15:00:00Z", "2026-03-30T15:10:00Z"]),
                "model_probability": [0.56, 0.62],
                "contract_won": [1, 0],
                "contract_type": ["threshold", "threshold"],
                "side": ["yes", "yes"],
                "exit_trigger": ["take_profit", "time_exit"],
            }
        )
        result = BacktestResult(strategy_mode="early_exit", trades=trades, summary={"pnl": 0.5})
        report = build_summary_report(result)
        self.assertIn("edge_distribution", report)
        self.assertIn("calibration", report)

    def test_sensitivity_runs_multiple_scenarios(self):
        features = pd.DataFrame(
            {"realized_volatility": [0.9, 0.9]},
            index=pd.to_datetime(["2026-03-30T15:00:00Z", "2026-03-30T15:20:00Z"]),
        )
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTC",
                market_ticker="KXBTC-65000",
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
            ),
            MarketSnapshot(
                source="test",
                series_ticker="KXBTC",
                market_ticker="KXBTC-65000",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
                expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
                spot_price=65100,
                settlement_price=65100,
                threshold=65000,
                direction="above",
                yes_bid=0.99,
                yes_ask=0.99,
                no_bid=0.01,
                no_ask=0.01,
            ),
        ]
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.05, 0.0, 5, 95, uncertainty_penalty=0.0, max_spread_cents=20, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
        )
        report = run_volatility_sensitivity(engine, snapshots, features, [0.8, 1.0, 1.2])
        self.assertEqual(len(report), 3)

    def test_calibration_report_returns_bucket_metrics(self):
        trades = pd.DataFrame(
            {
                "model_probability": [0.52, 0.58, 0.63, 0.67],
                "contract_won": [0, 1, 1, 0],
                "contract_type": ["threshold"] * 4,
                "side": ["yes"] * 4,
                "exit_trigger": ["settlement"] * 4,
            }
        )
        report = build_calibration_report(trades)
        self.assertIn("overall", report)
        self.assertGreater(len(report["buckets"]), 0)


if __name__ == "__main__":
    unittest.main()
