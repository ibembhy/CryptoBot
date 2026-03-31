from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.models.latency_repricing import LatencyRepricingModel
from kalshi_btc_bot.reports.comparison import build_grid_search_report, build_model_comparison_report, filter_snapshots_for_focus
from kalshi_btc_bot.signals.fusion import FusionConfig
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.risk import RiskConfig
from kalshi_btc_bot.types import MarketSnapshot


class ComparisonReportTests(unittest.TestCase):
    def test_build_model_comparison_report_has_all_models(self):
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
                threshold=67100,
                direction="above",
                yes_bid=0.40,
                yes_ask=0.41,
                no_bid=0.59,
                no_ask=0.60,
                settlement_price=67200,
                metadata={"recent_log_return": 0.004},
            ),
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-TEST",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=observed_at + timedelta(minutes=5),
                expiry=expiry,
                spot_price=67200,
                threshold=67100,
                direction="above",
                yes_bid=0.75,
                yes_ask=0.76,
                no_bid=0.24,
                no_ask=0.25,
                settlement_price=67200,
                metadata={"recent_log_return": 0.002},
            ),
        ]
        feature_frame = pd.DataFrame(
            {
                "close": [67000, 67200],
                "log_return": [0.0, 0.004],
                "realized_volatility": [0.8, 0.8],
            },
            index=pd.to_datetime([observed_at, observed_at + timedelta(minutes=5)], utc=True),
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            models={
                "gbm_threshold": GBMThresholdModel(),
                "latency_repricing": LatencyRepricingModel(),
            },
            signal_config=SignalConfig(0.01, 0.0, 5, 95),
            exit_config=ExitConfig(8, 10, 3, 5),
            backtest_config=BacktestConfig(1, 1, 1, 0.0, 1000.0),
            fusion_config=FusionConfig("hybrid", "latency_repricing", "gbm_threshold", 0.6, 0.4),
            risk_config=RiskConfig(100.0, 1000.0, 5, 5, 100.0, 100.0),
        )
        report = build_model_comparison_report(
            {
                "gbm_threshold": engine,
                "latency_repricing": engine,
                "hybrid": engine,
            },
            snapshots=snapshots,
            feature_frame=feature_frame,
        )
        self.assertEqual(set(report["models"]), {"gbm_threshold", "latency_repricing", "hybrid"})

    def test_filter_snapshots_for_focus(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        keep = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-KEEP",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=67000,
            threshold=67050,
            direction="above",
            yes_bid=0.45,
            yes_ask=0.46,
        )
        drop = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-DROP",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=120),
            spot_price=67000,
            threshold=70000,
            direction="above",
            yes_bid=0.95,
            yes_ask=0.96,
        )
        filtered = filter_snapshots_for_focus(
            [keep, drop],
            near_money_bps=100.0,
            max_minutes_to_expiry=60.0,
            min_price_cents=20,
            max_price_cents=80,
        )
        self.assertEqual([snapshot.market_ticker for snapshot in filtered], ["KXBTCD-KEEP"])

    def test_build_grid_search_report_ranks_results(self):
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        expiry = observed_at + timedelta(minutes=15)
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-GRID",
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
                settlement_price=67100,
                metadata={"recent_log_return": 0.002},
            ),
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-GRID",
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
                settlement_price=67100,
                metadata={"recent_log_return": 0.001},
            ),
        ]
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            models={
                "gbm_threshold": GBMThresholdModel(),
                "latency_repricing": LatencyRepricingModel(),
            },
            signal_config=SignalConfig(0.01, 0.0, 5, 95),
            exit_config=ExitConfig(8, 10, 3, 5),
            backtest_config=BacktestConfig(1, 1, 1, 0.0, 1000.0),
            fusion_config=FusionConfig("hybrid", "latency_repricing", "gbm_threshold", 0.6, 0.4),
            risk_config=RiskConfig(100.0, 1000.0, 5, 5, 100.0, 100.0),
        )
        report = build_grid_search_report(
            {
                "gbm_threshold": engine,
                "latency_repricing": engine,
                "hybrid": engine,
            },
            snapshots=snapshots,
            volatility_window=5,
            annualization_factor=365.0 * 24.0 * 60.0,
            near_money_bps_values=[100.0],
            max_minutes_to_expiry_values=[30.0],
            min_price_cents_values=[20],
            max_price_cents_values=[80],
            top_n=5,
        )
        self.assertEqual(report["grid_size"], 3)
        self.assertEqual(len(report["top_results"]), 3)
        self.assertIn("best_mode", report["top_results"][0])


if __name__ == "__main__":
    unittest.main()
