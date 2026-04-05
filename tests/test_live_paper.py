from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

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
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate, TradingSignal


class _FakeCollector:
    def __init__(self, snapshots_by_call: list[list[MarketSnapshot]]) -> None:
        self.snapshots_by_call = snapshots_by_call
        self.index = 0
        self.config = type("Config", (), {"series_ticker": "KXBTCD"})()

    async def reconcile_once(self) -> list[MarketSnapshot]:
        current = self.snapshots_by_call[min(self.index, len(self.snapshots_by_call) - 1)]
        self.index += 1
        return current


class _FailingCollector:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.config = type("Config", (), {"series_ticker": "KXBTCD"})()

    async def reconcile_once(self) -> list[MarketSnapshot]:
        raise self.error


class _NoopSnapshotStore:
    def latest_snapshot_for_market(self, market_ticker: str):
        return None

    def update_market_settlement(self, market_ticker: str, settlement_price: float) -> None:
        return None


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


class _FailingCoinbaseClient:
    def fetch_candles(self, start, end, timeframe):
        raise RuntimeError("coinbase unavailable")


class _FakeSettlementEnricher:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.calls = 0

    def enrich(self) -> dict[str, object]:
        self.calls += 1
        return dict(self.result)


class _FakeSettlementKalshiClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def get_market(self, ticker: str):
        return dict(self.payload)

    def get_historical_market(self, ticker: str):
        return dict(self.payload)


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

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=30)
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
        state = json.loads(ledger_path.read_text(encoding="utf-8"))
        closed = state["closed_positions"][0]
        self.assertEqual(closed["intended_entry_price_cents"], 40)
        self.assertEqual(closed["entry_slippage_cents"], 1)
        self.assertEqual(closed["first_post_entry_mark_cents"], 52)

    def test_live_paper_preview_once_returns_ranked_candidates(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_preview.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_preview.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=20)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-PREVIEW",
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
        trader = LivePaperTrader(
            collector=_FakeCollector([[snapshot]]),
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
        preview = asyncio.run(trader.preview_once())
        self.assertEqual(preview["candidate_count"], 1)
        self.assertEqual(preview["top_candidates"][0]["market_ticker"], "KXBTCD-PREVIEW")

    def test_live_paper_trader_closes_expired_position_even_when_market_is_missing_from_current_cycle(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_expired_close.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_expired_close.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=30)
        expiry = observed_at + timedelta(minutes=5)
        entry_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-MISSING",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=expiry,
            spot_price=65000.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.60,
            volume=1000.0,
        )

        store = SnapshotStore(db_path)
        store.insert_snapshot(entry_snapshot)
        trader = LivePaperTrader(
            collector=_FakeCollector([[entry_snapshot], []]),
            engine=self._engine(),
            coinbase_client=_FakeCoinbaseClient(),
            snapshot_store=store,
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

        settled_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-MISSING",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=expiry,
            spot_price=65000.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.60,
            volume=1000.0,
            settlement_price=1.0,
        )
        store.insert_snapshot(settled_snapshot)
        expired_position = trader.position_book.positions["KXBTCD-MISSING"]
        expired_position.expiry = datetime.now(timezone.utc) - timedelta(minutes=1)

        second = asyncio.run(trader.run_once())
        self.assertEqual(second["positions_closed"], 1)

    def test_live_paper_trader_does_not_force_settlement_from_wall_clock_before_snapshot_expiry(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_clock_skew.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_clock_skew.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=30)
        expiry = observed_at + timedelta(minutes=20)
        entry_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-CLOCKSKEW",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=expiry,
            spot_price=65000.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.60,
            volume=1000.0,
        )
        pre_expiry_snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-CLOCKSKEW",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=expiry - timedelta(minutes=6),
            expiry=expiry,
            spot_price=64980.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.41,
            yes_ask=0.43,
            no_bid=0.55,
            no_ask=0.57,
            volume=1000.0,
        )

        trader = LivePaperTrader(
            collector=_FakeCollector([[entry_snapshot]]),
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
        asyncio.run(trader.run_once())
        position = next(iter(trader.position_book.positions.values()))
        skewed_now = expiry + timedelta(seconds=45)
        with patch("kalshi_btc_bot.trading.live_paper.datetime") as mocked_datetime:
            mocked_datetime.now.return_value = skewed_now
            decision = trader._maybe_close_position(position, pre_expiry_snapshot, trader._build_feature_frame())

        self.assertIsNone(decision)
        self.assertEqual(position.status, "open")

    def test_live_paper_trader_ranks_candidates_before_opening(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_rank_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_rank_state.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=30)
        weaker = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-WEAK",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=65000.0,
            threshold=65120.0,
            direction="below",
            yes_bid=0.32,
            yes_ask=0.35,
            no_bid=0.63,
            no_ask=0.66,
            volume=1500.0,
        )
        stronger = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-STRONG",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=65000.0,
            threshold=65080.0,
            direction="below",
            yes_bid=0.26,
            yes_ask=0.28,
            no_bid=0.70,
            no_ask=0.73,
            volume=1500.0,
        )

        engine = self._engine()
        engine.risk_config = replace(engine.risk_config, max_open_positions=1, max_positions_per_expiry=1)
        trader = LivePaperTrader(
            collector=_FakeCollector([[weaker, stronger]]),
            engine=engine,
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

        result = asyncio.run(trader.run_once())
        self.assertEqual(result["positions_opened"], 1)
        self.assertEqual(result["opened"][0]["market_ticker"], "KXBTCD-STRONG")

    def test_live_paper_trader_runs_periodic_settlement_enrichment_and_refreshes_calibrators(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_enrich_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_enrich_state.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime(2026, 3, 31, 14, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-IDLE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=65000.0,
            threshold=65200.0,
            direction="below",
            yes_bid=0.50,
            yes_ask=0.52,
            no_bid=0.46,
            no_ask=0.48,
            volume=500.0,
        )
        settlement_enricher = _FakeSettlementEnricher({"markets_checked": 5, "markets_updated": 2})
        refresh_calls: list[int] = []

        def _refresh():
            refresh_calls.append(1)
            return {"gbm_threshold": 40}

        trader = LivePaperTrader(
            collector=_FakeCollector([[snapshot], [snapshot]]),
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
                settlement_check_interval_seconds=300,
            ),
            settlement_enricher=settlement_enricher,
            calibrator_refresher=_refresh,
        )

        first = asyncio.run(trader.run_once())
        self.assertEqual(settlement_enricher.calls, 1)
        self.assertEqual(first["settlement_enrichment"]["markets_updated"], 2)
        self.assertEqual(first["settlement_enrichment"]["calibrators_refreshed"], {"gbm_threshold": 40})
        self.assertEqual(len(refresh_calls), 1)

        second = asyncio.run(trader.run_once())
        self.assertIsNone(second["settlement_enrichment"])
        self.assertEqual(settlement_enricher.calls, 1)

    def test_live_paper_trader_runs_periodic_settlement_enrichment_without_calibrator_refresh(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_enrich_no_refresh_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_enrich_no_refresh_state.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime(2026, 3, 31, 14, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-IDLE-NO-REFRESH",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=20),
            spot_price=65000.0,
            threshold=65200.0,
            direction="below",
            yes_bid=0.50,
            yes_ask=0.52,
            no_bid=0.46,
            no_ask=0.48,
            volume=500.0,
        )
        settlement_enricher = _FakeSettlementEnricher({"markets_checked": 3, "markets_updated": 1})

        trader = LivePaperTrader(
            collector=_FakeCollector([[snapshot]]),
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
                settlement_check_interval_seconds=300,
            ),
            settlement_enricher=settlement_enricher,
            calibrator_refresher=None,
        )

        result = asyncio.run(trader.run_once())
        self.assertEqual(settlement_enricher.calls, 1)
        self.assertEqual(result["settlement_enrichment"]["markets_updated"], 1)
        self.assertNotIn("calibrators_refreshed", result["settlement_enrichment"])

    def test_live_paper_trader_restores_peak_equity_from_ledger_on_restart(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_restart.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_restart_state.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime(2026, 4, 1, 15, 0, tzinfo=timezone.utc)
        state = {
            "updated_at": observed_at.isoformat(),
            "open_positions": [
                {
                    "position_id": "pos-2",
                    "market_ticker": "KXBTCD-OPEN",
                    "side": "yes",
                    "contracts": 1,
                    "entry_time": observed_at.isoformat(),
                    "entry_price_cents": 45,
                    "strategy_mode": "hybrid",
                    "status": "open",
                    "expiry": (observed_at + timedelta(minutes=10)).isoformat(),
                    "entry_fees_paid": 0.0,
                    "signal_time": observed_at.isoformat(),
                    "intended_entry_price_cents": 44,
                    "entry_slippage_cents": 1,
                    "first_post_entry_time": None,
                    "first_post_entry_mark_cents": None,
                    "second_post_entry_time": None,
                    "second_post_entry_mark_cents": None,
                    "exit_time": None,
                    "exit_price_cents": None,
                    "exit_trigger": None,
                    "realized_pnl": None,
                }
            ],
            "closed_positions": [
                {
                    "position_id": "pos-1",
                    "market_ticker": "KXBTCD-CLOSED",
                    "side": "yes",
                    "contracts": 1,
                    "entry_time": (observed_at - timedelta(minutes=20)).isoformat(),
                    "entry_price_cents": 40,
                    "strategy_mode": "hybrid",
                    "status": "closed",
                    "expiry": (observed_at - timedelta(minutes=5)).isoformat(),
                    "entry_fees_paid": 0.0,
                    "signal_time": (observed_at - timedelta(minutes=20)).isoformat(),
                    "intended_entry_price_cents": 39,
                    "entry_slippage_cents": 1,
                    "first_post_entry_time": None,
                    "first_post_entry_mark_cents": None,
                    "second_post_entry_time": None,
                    "second_post_entry_mark_cents": None,
                    "exit_time": (observed_at - timedelta(minutes=5)).isoformat(),
                    "exit_price_cents": 65,
                    "exit_trigger": "take_profit",
                    "realized_pnl": 0.25,
                }
            ],
            "fills": [
                {
                    "market_ticker": "KXBTCD-CLOSED",
                    "side": "buy_yes",
                    "contracts": 1,
                    "price_cents": 40,
                    "timestamp": (observed_at - timedelta(minutes=20)).isoformat(),
                    "fees_paid": 0.0,
                }
            ],
            "realized_pnl": 0.25,
            "session_notional": 0.85,
            "peak_equity": 1000.25,
        }
        ledger_path.write_text(json.dumps(state), encoding="utf-8")

        trader = LivePaperTrader(
            collector=_FakeCollector([[]]),
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

        self.assertEqual(trader.risk_manager.peak_equity, 1000.25)
        self.assertEqual(trader.risk_manager.realized_pnl, 0.25)
        self.assertEqual(trader.risk_manager.session_notional, 0.85)
        self.assertEqual(len(trader.paper_broker.fills), 1)
        self.assertIn("KXBTCD-OPEN", trader.position_book.positions)
        self.assertEqual(trader.position_book._counter, 2)
        self.assertEqual(
            trader.risk_manager.open_positions_by_expiry[(observed_at + timedelta(minutes=10)).isoformat()],
            1,
        )

    def test_live_paper_trader_scales_contracts_with_bankroll_and_tier_multiplier(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_compounding.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_compounding.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC15M",
            market_ticker="KXBTC15M-COMPOUND",
            contract_type="direction",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=12),
            spot_price=68000.0,
            threshold=68010.0,
            direction="above",
            yes_bid=0.49,
            yes_ask=0.50,
            no_bid=0.48,
            no_ask=0.49,
            volume=1000.0,
            open_interest=1000.0,
        )
        snapshot.metadata["_paper_signal"] = TradingSignal(
            market_ticker=snapshot.market_ticker,
            action="buy_yes",
            side="yes",
            raw_model_probability=0.62,
            model_probability=0.62,
            market_probability=0.50,
            edge=0.12,
            raw_edge=0.12,
            entry_price_cents=50,
            fair_value_cents=62,
            expected_value_cents=12.0,
            reason="test",
            confidence=0.62,
            quality_score=1.0,
            tier_label="A",
            size_multiplier=1.0,
            spread_cents=1,
        )
        snapshot.metadata["_paper_estimate"] = ProbabilityEstimate(
            model_name="gbm_threshold",
            observed_at=observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=snapshot.threshold or snapshot.spot_price,
            volatility=0.1,
            drift=0.0,
            probability=0.62,
        )

        trader = LivePaperTrader(
            collector=_FakeCollector([[snapshot]]),
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
                enable_bankroll_sizing=True,
                bankroll_fraction_per_trade=0.2,
                min_cash_buffer=0.0,
                max_contracts_per_trade=5,
                respect_tier_size_multiplier=True,
            ),
        )

        result = asyncio.run(trader.run_once())
        self.assertEqual(result["positions_opened"], 1)
        self.assertEqual(result["opened"][0]["contracts"], 5)
        self.assertIn("_paper_signal", snapshot.metadata)
        self.assertIn("_paper_estimate", snapshot.metadata)

    def test_live_paper_trader_raises_on_corrupt_ledger(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_corrupt.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_corrupt.json"
        ledger_path.write_text("{not valid json", encoding="utf-8")

        with self.assertRaises(RuntimeError):
            LivePaperTrader(
                collector=_FakeCollector([[]]),
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

    def test_fetch_market_settlement_price_unwraps_market_payload(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_settlement_unwrap.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_settlement_unwrap.json"
        if ledger_path.exists():
            ledger_path.unlink()

        trader = LivePaperTrader(
            collector=type(
                "Collector",
                (),
                {
                    "kalshi_client": _FakeSettlementKalshiClient(
                        {"market": {"ticker": "KXBTCD-SETTLED", "result": "yes"}}
                    )
                },
            )(),
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

        self.assertEqual(trader._fetch_market_settlement_price("KXBTCD-SETTLED"), 1.0)

    def test_live_paper_trader_records_coinbase_failure_in_state(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_coinbase_failure.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_coinbase_failure.json"
        if ledger_path.exists():
            ledger_path.unlink()

        observed_at = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-FAIL",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(minutes=15),
            spot_price=65000.0,
            threshold=65100.0,
            direction="below",
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.60,
            volume=1000.0,
        )

        trader = LivePaperTrader(
            collector=_FakeCollector([[snapshot]]),
            engine=self._engine(),
            coinbase_client=_FailingCoinbaseClient(),
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

        result = asyncio.run(trader.run_once())
        self.assertEqual(result["status"], "coinbase_data_unavailable")
        state = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(state["data_feed_status"], "degraded")
        self.assertIn("coinbase unavailable", str(state["last_coinbase_error"]))

    def test_live_paper_trader_records_market_data_failure_in_state(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "live_paper_market_data_failure.sqlite3"
        if db_path.exists():
            db_path.unlink()
        ledger_path = base_dir / "live_paper_market_data_failure.json"
        if ledger_path.exists():
            ledger_path.unlink()

        trader = LivePaperTrader(
            collector=_FailingCollector(RuntimeError("429 Too Many Requests")),
            engine=self._engine(),
            coinbase_client=_FakeCoinbaseClient(),
            snapshot_store=_NoopSnapshotStore(),
            config=LivePaperConfig(
                feature_history_hours=24,
                poll_interval_seconds=30,
                ledger_path=str(ledger_path),
                feature_timeframe="1m",
                volatility_window=20,
                annualization_factor=105120.0,
            ),
        )

        result = asyncio.run(trader.run_once())
        self.assertEqual(result["status"], "market_data_unavailable")
        state = json.loads(ledger_path.read_text(encoding="utf-8"))
        self.assertEqual(state["data_feed_status"], "degraded")
        self.assertIn("429 Too Many Requests", str(state["last_market_data_error"]))


if __name__ == "__main__":
    unittest.main()
