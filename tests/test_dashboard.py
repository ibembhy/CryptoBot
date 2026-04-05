from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

import pandas as pd

from app import summarize_paper_state, summarize_signal_table
from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.dashboard import (
    active_snapshots_by_market,
    build_fills_table,
    build_latest_snapshot_table,
    build_live_signal_table,
    build_signal_reason_table,
    format_dashboard_time,
    build_positions_table,
    latest_snapshots_by_market,
    load_paper_trading_state,
)
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.risk import RiskConfig
from kalshi_btc_bot.types import MarketSnapshot


class DashboardTests(unittest.TestCase):
    def test_format_dashboard_time_uses_eastern_readable_format(self):
        formatted = format_dashboard_time("2026-04-01T02:05:00+00:00")
        self.assertEqual(formatted, "Mar 31, 2026 10:05:00 PM ET")

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
        self.assertIn("decision", table.columns)
        self.assertIn("reason_bucket", table.columns)
        self.assertIn("spread_regime", table.columns)
        self.assertIn("liquidity_regime", table.columns)
        self.assertIn("tier_label", table.columns)
        self.assertIn("size_multiplier", table.columns)

    def test_build_signal_reason_table_groups_reasons(self):
        signal_table = pd.DataFrame(
            [
                {"decision": "taken_candidate", "reason_bucket": "tradeable", "reason": "Conservative edge 0.1234 exceeds threshold in global regime."},
                {"decision": "taken_candidate", "reason_bucket": "tradeable", "reason": "Conservative edge 0.1234 exceeds threshold in global regime."},
                {"decision": "skipped", "reason_bucket": "stale_data", "reason": "Snapshot data is stale."},
            ]
        )
        reason_table = build_signal_reason_table(signal_table)
        self.assertEqual(len(reason_table), 2)
        self.assertIn("count", reason_table.columns)
        grouped = {(row["decision"], row["reason_bucket"]): int(row["count"]) for _, row in reason_table.iterrows()}
        self.assertEqual(grouped[("taken_candidate", "tradeable")], 2)
        self.assertEqual(grouped[("skipped", "stale_data")], 1)

    def test_build_latest_snapshot_table(self):
        observed_at = datetime(2026, 3, 31, 15, 0, tzinfo=timezone.utc)
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
        table = build_latest_snapshot_table([snapshot], reference_time=observed_at)
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["market_ticker"], "KXBTCD-TEST")

    def test_active_snapshots_by_market_filters_expired_markets(self):
        reference_time = datetime(2026, 3, 31, 19, 0, tzinfo=timezone.utc)
        expired = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-EXPIRED",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=reference_time - timedelta(minutes=5),
            expiry=reference_time - timedelta(minutes=1),
            spot_price=67000,
            threshold=67050,
            direction="above",
            yes_bid=0.40,
            yes_ask=0.41,
        )
        active = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-ACTIVE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=reference_time - timedelta(minutes=1),
            expiry=reference_time + timedelta(minutes=10),
            spot_price=67100,
            threshold=67150,
            direction="above",
            yes_bid=0.40,
            yes_ask=0.41,
        )
        latest = active_snapshots_by_market([expired, active], reference_time=reference_time)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].market_ticker, "KXBTCD-ACTIVE")

    def test_load_paper_trading_state_returns_defaults_when_missing(self):
        state = load_paper_trading_state(Path("test_artifacts") / "missing_paper_state.json")
        self.assertEqual(state["open_positions"], [])
        self.assertEqual(state["realized_pnl"], 0.0)

    def test_load_paper_trading_state_reconstructs_missing_closed_positions_from_fills(self):
        artifact = Path("test_artifacts") / "paper_state_reconstruct.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            pd.Series(
                {
                    "updated_at": "2026-04-01T02:03:06+00:00",
                    "open_positions": [],
                    "closed_positions": [
                        {
                            "position_id": "pos-1",
                            "market_ticker": "KXBTCD-A",
                            "side": "no",
                            "contracts": 1,
                            "entry_time": "2026-04-01T01:45:08+00:00",
                            "entry_price_cents": 55,
                            "status": "closed",
                            "exit_time": "2026-04-01T01:57:53+00:00",
                            "exit_price_cents": 98,
                            "realized_pnl": 0.43,
                        }
                    ],
                    "fills": [
                        {"market_ticker": "KXBTCD-A", "side": "buy_no", "contracts": 1, "price_cents": 55, "timestamp": "2026-04-01T01:45:08+00:00", "fees_paid": 0.0},
                        {"market_ticker": "KXBTCD-B", "side": "buy_no", "contracts": 1, "price_cents": 36, "timestamp": "2026-04-01T01:51:30+00:00", "fees_paid": 0.0},
                        {"market_ticker": "KXBTCD-B", "side": "sell_no", "contracts": 1, "price_cents": 12, "timestamp": "2026-04-01T01:53:38+00:00", "fees_paid": 0.0},
                        {"market_ticker": "KXBTCD-A", "side": "sell_no", "contracts": 1, "price_cents": 98, "timestamp": "2026-04-01T01:57:53+00:00", "fees_paid": 0.0},
                    ],
                    "realized_pnl": -0.04,
                    "session_notional": 1.29,
                }
            ).to_json(),
            encoding="utf-8",
        )
        state = load_paper_trading_state(artifact)
        self.assertEqual(len(state["closed_positions"]), 2)
        reconstructed = [row for row in state["closed_positions"] if row["market_ticker"] == "KXBTCD-B"][0]
        self.assertEqual(reconstructed["exit_trigger"], "reconstructed_from_fills")
        self.assertEqual(reconstructed["entry_price_cents"], 36)
        self.assertEqual(reconstructed["exit_price_cents"], 12)

    def test_build_positions_and_fills_tables(self):
        positions = [
            {
                "position_id": "pos-1",
                "market_ticker": "KXBTCD-TEST",
                "side": "yes",
                "contracts": 1,
                "entry_time": "2026-03-31T15:00:00+00:00",
                "entry_price_cents": 40,
                "status": "open",
                "strategy_mode": "hybrid",
            }
        ]
        fills = [
            {
                "timestamp": "2026-03-31T15:01:00+00:00",
                "market_ticker": "KXBTCD-TEST",
                "side": "buy_yes",
                "contracts": 1,
                "price_cents": 40,
                "fees_paid": 0.0,
            }
        ]
        positions_table = build_positions_table(positions)
        fills_table = build_fills_table(fills)
        self.assertFalse(positions_table.empty)
        self.assertFalse(fills_table.empty)
        self.assertIn("market_ticker", positions_table.columns)
        self.assertIn("timestamp", fills_table.columns)
        self.assertTrue(str(positions_table.iloc[0]["entry_time"]).endswith("ET"))
        self.assertTrue(str(fills_table.iloc[0]["timestamp"]).endswith("ET"))

    def test_summarize_paper_state(self):
        summary = summarize_paper_state(
            {
                "realized_pnl": 1.5,
                "session_notional": 25.0,
                "updated_at": "2026-03-31T16:00:00+00:00",
                "open_positions": [{"position_id": "open-1"}],
                "closed_positions": [
                    {"position_id": "closed-1", "realized_pnl": 1.0},
                    {"position_id": "closed-2", "realized_pnl": -0.5},
                    {"position_id": "closed-3", "realized_pnl": 1.0},
                ],
            },
            starting_capital=100.0,
        )
        self.assertEqual(summary["status"], "Winning")
        self.assertEqual(summary["open_positions"], 1)
        self.assertEqual(summary["closed_positions"], 3)
        self.assertAlmostEqual(summary["starting_capital"], 100.0)
        self.assertAlmostEqual(summary["current_equity"], 101.5)
        self.assertAlmostEqual(summary["return_pct"], 1.5)
        self.assertAlmostEqual(summary["avg_pnl_per_trade"], 0.5)
        self.assertAlmostEqual(summary["win_rate"], 66.6666666667, places=4)
        self.assertTrue(str(summary["updated_at"]).endswith("ET"))

    def test_summarize_paper_state_ignores_synthetic_closed_positions_in_headline_stats(self):
        summary = summarize_paper_state(
            {
                "realized_pnl": 1.5,
                "session_notional": 25.0,
                "updated_at": "2026-03-31T16:00:00+00:00",
                "open_positions": [],
                "closed_positions": [
                    {"position_id": "closed-1", "realized_pnl": 1.0, "strategy_mode": "hybrid", "exit_trigger": "take_profit"},
                    {"position_id": "closed-2", "realized_pnl": -0.5, "strategy_mode": "hybrid", "exit_trigger": "stop_loss"},
                    {
                        "position_id": "synthetic-1",
                        "realized_pnl": -0.25,
                        "strategy_mode": "synthetic_from_fills",
                        "exit_trigger": "reconstructed_from_fills",
                    },
                ],
            },
            starting_capital=100.0,
        )
        self.assertEqual(summary["closed_positions"], 2)
        self.assertEqual(summary["reconstructed_closed_positions"], 1)
        self.assertAlmostEqual(summary["avg_pnl_per_trade"], 0.75)
        self.assertAlmostEqual(summary["win_rate"], 50.0)

    def test_summarize_signal_table_uses_top_tradeable_candidate(self):
        signal_table = pd.DataFrame(
            [
                {
                    "market_ticker": "KXBTCD-A",
                    "action": "no_action",
                    "side": None,
                    "edge": 0.9,
                    "quality_score": 0.9,
                },
                {
                    "market_ticker": "KXBTCD-B",
                    "action": "buy_yes",
                    "side": "yes",
                    "edge": 0.2,
                    "quality_score": 0.3,
                },
                {
                    "market_ticker": "KXBTCD-C",
                    "action": "buy_no",
                    "side": "no",
                    "edge": 0.4,
                    "quality_score": 0.7,
                },
            ]
        )
        summary = summarize_signal_table(signal_table)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["tradeable_count"], 2)
        self.assertEqual(summary["best_market_ticker"], "KXBTCD-C")
        self.assertEqual(summary["best_side"], "no")
        self.assertAlmostEqual(summary["best_edge"], 0.4)
        self.assertAlmostEqual(summary["best_quality_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
