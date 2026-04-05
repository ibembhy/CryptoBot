from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Callable

from kalshi_btc_bot.backtest.fills import apply_entry_fill, apply_exit_fill
from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.collectors.settlements import SettlementEnricher
from kalshi_btc_bot.collectors.hybrid import HybridCollector
from kalshi_btc_bot.data.coinbase import CoinbaseClient, default_history_window
from kalshi_btc_bot.data.features import build_feature_frame
from kalshi_btc_bot.markets.normalize import _settlement_price
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.trading.exits import mark_price_cents
from kalshi_btc_bot.trading.paper_broker import PaperBroker, PaperFill
from kalshi_btc_bot.trading.positions import PositionBook
from kalshi_btc_bot.trading.risk import RiskManager
from kalshi_btc_bot.types import MarketSnapshot, Position
from kalshi_btc_bot.utils.files import read_json_strict, write_json_atomic


@dataclass(frozen=True)
class LivePaperConfig:
    feature_history_hours: int
    poll_interval_seconds: int
    ledger_path: str
    feature_timeframe: str
    volatility_window: int
    annualization_factor: float
    feature_cache_refresh_seconds: int = 15
    enable_bankroll_sizing: bool = False
    bankroll_fraction_per_trade: float = 1.0
    min_cash_buffer: float = 0.0
    max_contracts_per_trade: int = 1
    respect_tier_size_multiplier: bool = True
    settlement_check_interval_seconds: int = 300
    settlement_recent_hours: int = 72
    settlement_max_markets_per_cycle: int = 100
    snapshot_store_read_only: bool = False


class LivePaperTrader:
    def __init__(
        self,
        *,
        collector: HybridCollector,
        engine: BacktestEngine,
        coinbase_client: CoinbaseClient,
        snapshot_store: SnapshotStore,
        config: LivePaperConfig,
        settlement_enricher: SettlementEnricher | None = None,
        calibrator_refresher: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        self.collector = collector
        self.engine = engine
        self.coinbase_client = coinbase_client
        self.snapshot_store = snapshot_store
        self.config = config
        self.paper_broker = PaperBroker()
        self.position_book = PositionBook()
        self.risk_manager = RiskManager(self.engine.risk_config)
        self.risk_manager.initialize(self.engine.backtest_config.starting_bankroll)
        self.ledger_path = Path(self.config.ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self.settlement_enricher = settlement_enricher
        self.calibrator_refresher = calibrator_refresher
        self._last_settlement_enrichment_at: datetime | None = None
        self._last_coinbase_error: str | None = None
        self._last_market_data_error: str | None = None
        self._last_successful_feature_build_at: datetime | None = None
        self._cached_feature_frame = None
        self._feature_frame_built_at: datetime | None = None
        self._feature_frame_retry_after: datetime | None = None

    async def run_once(self) -> dict[str, object]:
        try:
            snapshots = await self.collector.reconcile_once()
        except Exception as exc:
            self._last_market_data_error = f"{type(exc).__name__}: {exc}"
            state = self._save_state()
            return {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": "market_data_unavailable",
                "warning": "Kalshi market data is temporarily unavailable; paper trading is paused until snapshots recover.",
                "snapshots_seen": 0,
                "positions_opened": 0,
                "positions_closed": 0,
                "open_positions": len([p for p in self.position_book.positions.values() if p.status == "open"]),
                "realized_pnl": round(self.risk_manager.realized_pnl, 2),
                "opened": [],
                "closed": [],
                "ledger_path": str(self.ledger_path),
                "state": state,
                "settlement_enrichment": None,
            }
        self._last_market_data_error = None
        settlement_result = self._maybe_run_settlement_enrichment()
        try:
            feature_frame = self._build_feature_frame()
        except Exception as exc:
            self._last_coinbase_error = f"{type(exc).__name__}: {exc}"
            state = self._save_state()
            return {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": "coinbase_data_unavailable",
                "warning": "Coinbase market data is unavailable; paper trading is paused until data returns.",
                "snapshots_seen": len(snapshots),
                "positions_opened": 0,
                "positions_closed": 0,
                "open_positions": len([p for p in self.position_book.positions.values() if p.status == "open"]),
                "realized_pnl": round(self.risk_manager.realized_pnl, 2),
                "opened": [],
                "closed": [],
                "ledger_path": str(self.ledger_path),
                "state": state,
                "settlement_enrichment": settlement_result,
            }
        self._last_coinbase_error = None
        self._last_successful_feature_build_at = datetime.now(timezone.utc)
        opened: list[dict[str, object]] = []
        closed: list[dict[str, object]] = []

        latest_by_market = {snapshot.market_ticker: snapshot for snapshot in snapshots}
        self._refresh_expired_open_position_settlements()

        for market_ticker, position in list(self.position_book.positions.items()):
            if position.status != "open":
                continue
            snapshot = latest_by_market.get(market_ticker) or self.snapshot_store.latest_snapshot_for_market(market_ticker)
            if snapshot is None:
                continue
            self._update_position_latency_diagnostics(position, snapshot)
            record = self._maybe_close_position(position, snapshot, feature_frame)
            if record is not None:
                closed.append(record)

        candidates = self._rank_candidate_snapshots(snapshots, feature_frame)
        for _, _, _, _, snapshot in candidates:
            record = self._maybe_open_position(snapshot, feature_frame)
            if record is not None:
                opened.append(record)

        state = self._save_state()
        return {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "snapshots_seen": len(snapshots),
            "positions_opened": len(opened),
            "positions_closed": len(closed),
            "open_positions": len([p for p in self.position_book.positions.values() if p.status == "open"]),
            "realized_pnl": round(self.risk_manager.realized_pnl, 2),
            "opened": opened,
            "closed": closed,
            "ledger_path": str(self.ledger_path),
            "state": state,
            "settlement_enrichment": settlement_result,
        }

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def preview_once(self) -> dict[str, object]:
        try:
            snapshots = await self.collector.reconcile_once()
        except Exception as exc:
            self._last_market_data_error = f"{type(exc).__name__}: {exc}"
            state = self._save_state()
            return {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "status": "market_data_unavailable",
                "warning": "Kalshi market data is temporarily unavailable; paper preview is paused until snapshots recover.",
                "snapshots_seen": 0,
                "candidate_count": 0,
                "top_candidates": [],
                "ledger_path": str(self.ledger_path),
                "state": state,
            }
        self._last_market_data_error = None
        feature_frame = self._build_feature_frame()
        candidates = self._rank_candidate_snapshots(snapshots, feature_frame)
        preview_rows: list[dict[str, object]] = []
        for ranking_score, repricing_score, edge, spread, snapshot in candidates[:10]:
            signal = snapshot.metadata.get("_paper_signal")
            estimate = snapshot.metadata.get("_paper_estimate")
            preview_rows.append(
                {
                    "market_ticker": snapshot.market_ticker,
                    "series_ticker": snapshot.series_ticker,
                    "observed_at": snapshot.observed_at.isoformat(),
                    "expiry": snapshot.expiry.isoformat(),
                    "side": getattr(signal, "side", None),
                    "action": getattr(signal, "action", None),
                    "entry_price_cents": getattr(signal, "entry_price_cents", None),
                    "edge": getattr(signal, "edge", None),
                    "quality_score": ranking_score,
                    "tier_label": getattr(signal, "tier_label", None),
                    "size_multiplier": getattr(signal, "size_multiplier", None),
                    "reason": getattr(signal, "reason", None),
                    "estimate_model": getattr(estimate, "model_name", None),
                    "predicted_repricing_cents": repricing_score if repricing_score != -999.0 else None,
                    "spread_sort_score": spread,
                }
            )
        return {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "snapshots_seen": len(snapshots),
            "candidate_count": len(candidates),
            "top_candidates": preview_rows,
        }

    def _build_feature_frame(self):
        now = datetime.now(timezone.utc)
        if (
            self._cached_feature_frame is not None
            and self._feature_frame_built_at is not None
            and (now - self._feature_frame_built_at).total_seconds() < self.config.feature_cache_refresh_seconds
        ):
            return self._cached_feature_frame
        if self._feature_frame_retry_after is not None and now < self._feature_frame_retry_after:
            if self._cached_feature_frame is not None:
                return self._cached_feature_frame
            raise RuntimeError("Price data unavailable; rate-limited. Retrying after cooldown.")
        try:
            start, end = default_history_window(self.config.feature_history_hours)
            candles = self.coinbase_client.fetch_candles(
                start,
                end,
                timeframe=self.config.feature_timeframe,
            )
            frame = build_feature_frame(
                candles,
                volatility_window=self.config.volatility_window,
                annualization_factor=self.config.annualization_factor,
            )
            self._cached_feature_frame = frame
            self._feature_frame_built_at = now
            self._feature_frame_retry_after = None
            return frame
        except Exception:
            self._feature_frame_retry_after = now + timedelta(seconds=30)
            raise

    def _rank_candidate_snapshots(self, snapshots: list[MarketSnapshot], feature_frame):
        candidates: list[tuple[float, float, float, float, MarketSnapshot]] = []
        for snapshot in snapshots:
            if self.position_book.has_open_position(snapshot.market_ticker):
                continue
            evaluation = self.engine.evaluate_snapshot(snapshot, feature_frame)
            if evaluation == (None, None):
                continue
            signal, estimate = evaluation
            if signal is None or estimate is None or signal.action == "no_action" or signal.side is None or signal.entry_price_cents is None:
                continue
            ranking_score = signal.quality_score
            repricing_score = signal.predicted_repricing_cents or -999.0
            edge = signal.edge or -999.0
            spread = -(signal.spread_cents or 99)
            snapshot.metadata["_paper_signal"] = signal
            snapshot.metadata["_paper_estimate"] = estimate
            candidates.append((ranking_score, repricing_score, edge, spread, snapshot))

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        return candidates

    def _maybe_run_settlement_enrichment(self) -> dict[str, object] | None:
        if self.settlement_enricher is None:
            return None
        if self.config.snapshot_store_read_only:
            return None
        now = datetime.now(timezone.utc)
        if self._last_settlement_enrichment_at is not None:
            elapsed = (now - self._last_settlement_enrichment_at).total_seconds()
            if elapsed < self.config.settlement_check_interval_seconds:
                return None

        self._last_settlement_enrichment_at = now
        result = self.settlement_enricher.enrich()
        if result.get("markets_updated", 0) > 0 and self.calibrator_refresher is not None:
            refreshed = self.calibrator_refresher()
            result = {**result, "calibrators_refreshed": refreshed}
        return result

    def _maybe_open_position(self, snapshot: MarketSnapshot, feature_frame) -> dict[str, object] | None:
        if snapshot.settlement_price is not None or snapshot.observed_at >= snapshot.expiry:
            return None
        cached_signal = snapshot.metadata.get("_paper_signal")
        cached_estimate = snapshot.metadata.get("_paper_estimate")
        if cached_signal is not None and cached_estimate is not None:
            signal, estimate = cached_signal, cached_estimate
        else:
            evaluation = self.engine.evaluate_snapshot(snapshot, feature_frame)
            if evaluation == (None, None):
                return None
            signal, estimate = evaluation
        if signal is None or estimate is None or signal.action == "no_action" or signal.side is None or signal.entry_price_cents is None:
            return None
        current_equity = self.engine.backtest_config.starting_bankroll + self.risk_manager.realized_pnl
        sized_entry = self._size_entry(signal=signal, snapshot=snapshot, current_equity=current_equity)
        if sized_entry is None:
            return None
        contracts, entry_fill, entry_notional = sized_entry
        position = self.position_book.open_position(
            market_ticker=snapshot.market_ticker,
            side=signal.side,
            contracts=contracts,
            entry_time=snapshot.observed_at,
            entry_price_cents=entry_fill.price_cents,
            strategy_mode=self.engine.fusion_config.mode if self.engine.fusion_config else "single",
            signal_time=snapshot.observed_at,
            intended_entry_price_cents=signal.entry_price_cents,
            entry_slippage_cents=entry_fill.price_cents - signal.entry_price_cents,
            expiry=snapshot.expiry,
            entry_fees_paid=entry_fill.fees_paid,
            entry_edge=float(signal.edge or 0.0),
        )
        self.risk_manager.record_entry(entry_notional=entry_notional, expiry=snapshot.expiry)
        self.paper_broker.record_fill(
            market_ticker=position.market_ticker,
            side=f"buy_{position.side}",
            contracts=position.contracts,
            price_cents=position.entry_price_cents,
            timestamp=position.entry_time,
            fees_paid=entry_fill.fees_paid,
        )
        return {
            "market_ticker": position.market_ticker,
            "side": position.side,
            "contracts": position.contracts,
            "entry_price_cents": position.entry_price_cents,
            "tier_label": signal.tier_label,
            "size_multiplier": signal.size_multiplier,
            "edge": signal.edge,
            "quality_score": signal.quality_score,
            "predicted_repricing_cents": signal.predicted_repricing_cents,
            "model_probability": signal.model_probability,
            "signal_time": snapshot.observed_at.isoformat(),
            "intended_entry_price_cents": signal.entry_price_cents,
            "entry_slippage_cents": entry_fill.price_cents - signal.entry_price_cents,
            "reason": signal.reason,
            "estimate_model": estimate.model_name,
        }

    def _size_entry(self, *, signal, snapshot: MarketSnapshot, current_equity: float):
        open_position_count = len([p for p in self.position_book.positions.values() if p.status == "open"])
        base_contracts = max(int(self.engine.backtest_config.default_contracts), 1)
        max_contracts = base_contracts
        if self.config.enable_bankroll_sizing:
            max_contracts = max(int(self.config.max_contracts_per_trade), base_contracts)
            if self.config.respect_tier_size_multiplier:
                tier_multiplier = max(float(getattr(signal, "size_multiplier", 1.0) or 0.0), 0.0)
                max_contracts = max(base_contracts, int(math.floor(max_contracts * tier_multiplier)))
        available_equity = max(float(current_equity) - float(self.config.min_cash_buffer), 0.0)
        sizing_budget = available_equity
        if self.config.enable_bankroll_sizing:
            sizing_budget = max(available_equity * float(self.config.bankroll_fraction_per_trade), 0.0)

        for contracts in range(max_contracts, 0, -1):
            entry_fill = apply_entry_fill(
                signal.entry_price_cents,
                self.engine.backtest_config.entry_slippage_cents,
                self.engine.backtest_config.fee_rate_bps,
                spread_cents=signal.spread_cents,
                volume=snapshot.volume,
                open_interest=snapshot.open_interest,
                contracts=contracts,
            )
            entry_notional = round((entry_fill.price_cents * contracts) / 100.0 + entry_fill.fees_paid, 4)
            if self.config.enable_bankroll_sizing and entry_notional > sizing_budget:
                continue
            risk_check = self.risk_manager.check_entry(
                entry_notional=entry_notional,
                expiry=snapshot.expiry,
                open_position_count=open_position_count,
                current_equity=current_equity,
            )
            if risk_check.allowed:
                return contracts, entry_fill, entry_notional
        return None

    def _maybe_close_position(self, position: Position, snapshot: MarketSnapshot, feature_frame) -> dict[str, object] | None:
        expiry = position.expiry or snapshot.expiry
        if snapshot.settlement_price is not None or snapshot.observed_at >= expiry:
            exit_price_cents = self.engine._settlement_price_cents(snapshot, position.side)
            realized_pnl = round(
                ((exit_price_cents - position.entry_price_cents) * position.contracts) / 100.0 - position.entry_fees_paid,
                2,
            )
            closed = self.position_book.close_position(
                position.market_ticker,
                exit_time=snapshot.observed_at,
                exit_price_cents=exit_price_cents,
                exit_trigger="settlement",
                realized_pnl=realized_pnl,
            )
            self.risk_manager.record_exit(
                realized_pnl=realized_pnl,
                expiry=snapshot.expiry,
                current_equity=self.engine.backtest_config.starting_bankroll + self.risk_manager.realized_pnl + realized_pnl,
            )
            self.paper_broker.record_fill(
                market_ticker=closed.market_ticker,
                side=f"sell_{closed.side}",
                contracts=closed.contracts,
                price_cents=exit_price_cents,
                timestamp=snapshot.observed_at,
                fees_paid=0.0,
            )
            return {
                "market_ticker": closed.market_ticker,
                "exit_trigger": "settlement",
                "exit_price_cents": exit_price_cents,
                "realized_pnl": realized_pnl,
            }

        decision = self.engine.evaluate_live_exit(
            snapshot=snapshot,
            side=position.side,
            entry_price_cents=position.entry_price_cents,
            contracts=position.contracts,
            feature_frame=feature_frame,
            entry_edge=position.entry_edge,
        )
        if decision is None or decision.action != "exit" or decision.exit_price_cents is None or decision.trigger is None:
            return None
        exit_fill = apply_exit_fill(
            decision.exit_price_cents,
            self.engine.backtest_config.exit_slippage_cents,
            self.engine.backtest_config.fee_rate_bps,
            spread_cents=self.engine._spread_cents(snapshot, position.side),
            volume=snapshot.volume,
            open_interest=snapshot.open_interest,
            contracts=position.contracts,
        )
        realized_pnl = round(
            ((exit_fill.price_cents - position.entry_price_cents) * position.contracts) / 100.0
            - position.entry_fees_paid
            - exit_fill.fees_paid,
            2,
        )
        closed = self.position_book.close_position(
            position.market_ticker,
            exit_time=snapshot.observed_at,
            exit_price_cents=exit_fill.price_cents,
            exit_trigger=decision.trigger,
            realized_pnl=realized_pnl,
        )
        self.risk_manager.record_exit(
            realized_pnl=realized_pnl,
            expiry=snapshot.expiry,
            current_equity=self.engine.backtest_config.starting_bankroll + self.risk_manager.realized_pnl + realized_pnl,
        )
        self.paper_broker.record_fill(
            market_ticker=closed.market_ticker,
            side=f"sell_{closed.side}",
            contracts=closed.contracts,
            price_cents=exit_fill.price_cents,
            timestamp=snapshot.observed_at,
            fees_paid=exit_fill.fees_paid,
        )
        return {
            "market_ticker": closed.market_ticker,
            "exit_trigger": decision.trigger,
            "exit_price_cents": exit_fill.price_cents,
            "realized_pnl": realized_pnl,
            "reason": decision.reason,
        }

    def _refresh_expired_open_position_settlements(self) -> None:
        if self.config.snapshot_store_read_only:
            return
        now = datetime.now(timezone.utc)
        for position in self.position_book.positions.values():
            if position.status != "open":
                continue
            expiry = position.expiry
            if expiry is None or now < expiry:
                continue
            settlement_price = self._fetch_market_settlement_price(position.market_ticker)
            if settlement_price is None:
                continue
            self.snapshot_store.update_market_settlement(position.market_ticker, settlement_price)

    def _update_position_latency_diagnostics(self, position: Position, snapshot: MarketSnapshot) -> None:
        if snapshot.observed_at <= position.entry_time:
            return
        mark_cents = mark_price_cents(snapshot, position.side)
        if mark_cents is None:
            return
        if position.first_post_entry_time is None:
            position.first_post_entry_time = snapshot.observed_at
            position.first_post_entry_mark_cents = mark_cents
            return
        if position.second_post_entry_time is None and snapshot.observed_at > position.first_post_entry_time:
            position.second_post_entry_time = snapshot.observed_at
            position.second_post_entry_mark_cents = mark_cents

    def _fetch_market_settlement_price(self, market_ticker: str) -> float | None:
        client = getattr(self.collector, "kalshi_client", None)
        if client is None:
            return None
        payload: dict[str, object] | None = None
        for fetch in (client.get_market, client.get_historical_market):
            try:
                payload = fetch(market_ticker)
            except Exception:
                continue
            if payload:
                break
        if not payload:
            return None
        market_payload = payload.get("market") if isinstance(payload, dict) else None
        if isinstance(market_payload, dict):
            return self._extract_market_settlement_price(market_payload)
        if isinstance(payload, dict):
            return self._extract_market_settlement_price(payload)
        return None

    @staticmethod
    def _extract_market_settlement_price(payload: dict[str, object]) -> float | None:
        settlement_price = _settlement_price(payload)
        if settlement_price is not None:
            return settlement_price
        status = str(payload.get("status") or "").strip().lower()
        result = str(payload.get("result") or payload.get("market_result") or "").strip().lower()
        if status == "settled":
            if result == "yes":
                return 1.0
            if result == "no":
                return 0.0
        return None

    def _save_state(self) -> dict[str, object]:
        open_positions = []
        closed_positions = []
        for position in self.position_book.positions.values():
            payload = asdict(position)
            for key in (
                "signal_time",
                "entry_time",
                "exit_time",
                "expiry",
                "first_post_entry_time",
                "second_post_entry_time",
            ):
                if payload.get(key):
                    payload[key] = payload[key].isoformat()
            if position.status == "open":
                open_positions.append(payload)
            else:
                closed_positions.append(payload)
        fills = []
        for fill in self.paper_broker.fills:
            payload = asdict(fill)
            payload["timestamp"] = payload["timestamp"].isoformat()
            fills.append(payload)
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "open_positions": open_positions,
            "closed_positions": closed_positions,
            "fills": fills,
            "realized_pnl": round(self.risk_manager.realized_pnl, 2),
            "session_notional": round(self.risk_manager.session_notional, 2),
            "peak_equity": round(self.risk_manager.peak_equity, 2),
            "data_feed_status": "degraded" if (self._last_coinbase_error or self._last_market_data_error) else "ok",
            "last_coinbase_error": self._last_coinbase_error,
            "last_market_data_error": self._last_market_data_error,
            "last_successful_feature_build_at": (
                self._last_successful_feature_build_at.isoformat()
                if self._last_successful_feature_build_at is not None
                else None
            ),
        }
        write_json_atomic(self.ledger_path, state)
        return state

    def _load_state(self) -> None:
        state = read_json_strict(
            self.ledger_path,
            missing_default=None,
            label="Paper trading ledger",
        )
        if state is None:
            return

        positions: dict[str, Position] = {}
        max_counter = 0
        for payload in state.get("open_positions", []):
            position = self._deserialize_position(payload)
            positions[position.market_ticker] = position
            max_counter = max(max_counter, self._position_counter(position.position_id))
        for payload in state.get("closed_positions", []):
            position = self._deserialize_position(payload)
            positions[position.market_ticker] = position
            max_counter = max(max_counter, self._position_counter(position.position_id))
        self.position_book.positions = positions
        self.position_book._counter = max_counter

        fills: list[PaperFill] = []
        for payload in state.get("fills", []):
            timestamp = payload.get("timestamp")
            if not timestamp:
                continue
            fills.append(
                PaperFill(
                    market_ticker=str(payload.get("market_ticker", "")),
                    side=str(payload.get("side", "")),
                    contracts=int(payload.get("contracts", 0)),
                    price_cents=int(payload.get("price_cents", 0)),
                    timestamp=datetime.fromisoformat(str(timestamp)),
                    fees_paid=float(payload.get("fees_paid", 0.0)),
                )
            )
        self.paper_broker.fills = fills

        self.risk_manager.realized_pnl = float(state.get("realized_pnl", 0.0))
        self.risk_manager.session_notional = float(state.get("session_notional", 0.0))
        self.risk_manager.open_positions_by_expiry = {}
        for position in self.position_book.positions.values():
            if position.status != "open" or position.expiry is None:
                continue
            expiry_key = position.expiry.isoformat()
            self.risk_manager.open_positions_by_expiry[expiry_key] = self.risk_manager.open_positions_by_expiry.get(expiry_key, 0) + 1

        stored_peak = state.get("peak_equity")
        if stored_peak is not None:
            self.risk_manager.peak_equity = float(stored_peak)
        else:
            self.risk_manager.peak_equity = self._rebuild_peak_equity(state.get("closed_positions", []))
        self._last_coinbase_error = str(state.get("last_coinbase_error") or "") or None
        successful_build = state.get("last_successful_feature_build_at")
        self._last_successful_feature_build_at = (
            datetime.fromisoformat(str(successful_build)) if successful_build else None
        )

    @staticmethod
    def _deserialize_position(payload: dict[str, object]) -> Position:
        datetime_fields = {
            "entry_time",
            "signal_time",
            "exit_time",
            "expiry",
            "first_post_entry_time",
            "second_post_entry_time",
        }
        normalized = dict(payload)
        for field_name in datetime_fields:
            raw = normalized.get(field_name)
            if raw:
                normalized[field_name] = datetime.fromisoformat(str(raw))
        return Position(**normalized)

    @staticmethod
    def _position_counter(position_id: str) -> int:
        if not position_id.startswith("pos-"):
            return 0
        try:
            return int(position_id.split("-", 1)[1])
        except (IndexError, ValueError):
            return 0

    def _rebuild_peak_equity(self, closed_positions: list[dict[str, object]]) -> float:
        starting_bankroll = self.engine.backtest_config.starting_bankroll
        ordered = sorted(
            (
                payload
                for payload in closed_positions
                if payload.get("exit_time") and payload.get("realized_pnl") is not None
            ),
            key=lambda payload: datetime.fromisoformat(str(payload["exit_time"])),
        )
        realized = 0.0
        peak_equity = float(starting_bankroll)
        for payload in ordered:
            realized += float(payload.get("realized_pnl", 0.0))
            peak_equity = max(peak_equity, float(starting_bankroll) + realized)
        return peak_equity
