from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.models.latency_repricing import LatencyRepricingModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.signals.fusion import FusionConfig
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.live_paper import LivePaperConfig, LivePaperTrader
from kalshi_btc_bot.trading.risk import RiskConfig
from kalshi_btc_bot.types import MarketSnapshot


class _FakeCollector:
    def __init__(self, snapshots_by_call: list[list[MarketSnapshot]]) -> None:
        self.snapshots_by_call = snapshots_by_call
        self.index = 0
        self.config = type("Config", (), {"series_ticker": "KXBTCD"})()

    async def reconcile_once(self) -> list[MarketSnapshot]:
        current = self.snapshots_by_call[min(self.index, len(self.snapshots_by_call) - 1)]
        self.index += 1
        return current


class _FakeCoinbaseClient:
    def fetch_candles(self, start, end, timeframe):
        index = pd.date_range(start=start, periods=120, freq="1min", tz="UTC")
        close = pd.Series([65000 + i for i in range(len(index))], index=index)
        return pd.DataFrame(
            {
                "open": close,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": 1000.0,
            },
            index=index,
        )


class LivePaperTraderTests(unittest.TestCase):
    def _engine(self) -> BacktestEngine:
        gbm = GBMThresholdModel(drift=0.0, volatility_floor=0.05)
        latency = LatencyRepricingModel(drift=0.0, volatility_floor=0.05, persistence_factor=0.75, min_move_bps=3.0, max_move_bps=100.0)
        return BacktestEngine(
            model=gbm,
            models={gbm.model_name: gbm, latency.model_name: latency},
            signal_config=SignalConfig(
                min_edge=0.01,
                min_confidence=0.0,
                min_contract_price_cents=25,
                max_contract_price_cents=80,
                max_near_money_bps=250.0,
                min_liquidity=0.0,
                uncertainty_penalty=0.0,
                max_spread_cents=8,
                liquidity_penalty_per_100_volume=0.0,
                max_data_age_seconds=120,
            ),
            exit_config=ExitConfig(
                take_profit_cents=8,
                stop_loss_cents=10,
                fair_value_buffer_cents=3,
                time_exit_minutes_before_expiry=5,
                min_hold_edge=0.02,
                min_ev_to_hold_cents=2,
                exit_liquidity_floor=0.0,
            ),
            backtest_config=BacktestConfig(
                default_contracts=1,
                entry_slippage_cents=0,
                exit_slippage_cents=0,
                fee_rate_bps=0.0,
                starting_bankroll=1000.0,
            ),
            fusion_config=FusionConfig(
                mode="hybrid",
                primary_model="gbm_threshold",
                confirm_model="latency_repricing",
                primary_weight=0.65,
                confirm_weight=0.35,
                require_side_agreement=False,
                min_combined_edge=0.01,
                allow_primary_unconfirmed=True,
            ),
            risk_config=RiskConfig(
                max_trade_notional=25.0,
                max_session_notional=250.0,
                max_open_positions=3,
                max_positions_per_expiry=2,
                max_daily_loss=25.0,
                max_drawdown=60.0,
            ),
        )

    def test_live_paper_trader_can_open_and_close_position(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_state.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime(2026, 3, 31, 14, 0, tzinfo=timezone.utc)
        entry_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-PAPER",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=65000.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.60,
            volume=1000.0,
        )
        exit_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-PAPER",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at + timedelta(minutes=21),
            expiry=observed_at + timedelta(minutes=20),
            spot_price=64850.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.52,
            yes_ask=0.54,
            no_bid=0.44,
            no_ask=0.46,
            volume=1000.0,
            settlement_price=1.0,
        )

        trader = LivePaperTrader(
            collector=_FakeCollector([[entry_snapshot], [exit_snapshot]]),
            engine=self._engine(),
            coinbase_client=_FakeCoinbaseClient(),
            snapshot_store=SnapshotStore(db_path),
            config=LivePaperConfig(
                feature_history_hours=24,
                poll_interval_seconds=30,
                ledger_path=str(ledger_path),
                feature_timeframe="1m",
                volatility_window=20,
                annualization_factor=105120.0,
            ),
        )

        first = asyncio.run(trader.run_once())
        self.assertEqual(first["positions_opened"], 1)
        self.assertEqual(first["positions_closed"], 0)

        second = asyncio.run(trader.run_once())
        self.assertEqual(second["positions_opened"], 0)
        self.assertEqual(second["positions_closed"], 1)
        self.assertTrue(ledger_path.exists())


if __name__ == "__main__":
    unittest.main()
