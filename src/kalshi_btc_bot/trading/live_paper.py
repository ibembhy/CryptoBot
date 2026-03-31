from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.collectors.hybrid import HybridCollector
from kalshi_btc_bot.data.coinbase import CoinbaseClient, default_history_window
from kalshi_btc_bot.data.features import build_feature_frame
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.trading.exits import mark_price_cents
from kalshi_btc_bot.trading.paper_broker import PaperBroker
from kalshi_btc_bot.trading.positions import PositionBook
from kalshi_btc_bot.trading.risk import RiskManager
from kalshi_btc_bot.types import MarketSnapshot, Position


@dataclass(frozen=True)
class LivePaperConfig:
    feature_history_hours: int
    poll_interval_seconds: int
    ledger_path: str
    feature_timeframe: str
    volatility_window: int
    annualization_factor: float


class LivePaperTrader:
    def __init__(
        self,
        *,
        collector: HybridCollector,
        engine: BacktestEngine,
        coinbase_client: CoinbaseClient,
        snapshot_store: SnapshotStore,
        config: LivePaperConfig,
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

    async def run_once(self) -> dict[str, object]:
        snapshots = await self.collector.reconcile_once()
        feature_frame = self._build_feature_frame()
        opened: list[dict[str, object]] = []
        closed: list[dict[str, object]] = []

        latest_by_market = {snapshot.market_ticker: snapshot for snapshot in snapshots}

        for market_ticker, position in list(self.position_book.positions.items()):
            if position.status != "open":
                continue
            snapshot = latest_by_market.get(market_ticker)
            if snapshot is None:
                continue
            record = self._maybe_close_position(position, snapshot, feature_frame)
            if record is not None:
                closed.append(record)

        for snapshot in snapshots:
            if self.position_book.has_open_position(snapshot.market_ticker):
                continue
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
        }

    async def run_forever(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _build_feature_frame(self):
        start, end = default_history_window(self.config.feature_history_hours)
        candles = self.coinbase_client.fetch_candles(
            start,
            end,
            timeframe=self.config.feature_timeframe,
        )
        return build_feature_frame(
            candles,
            volatility_window=self.config.volatility_window,
            annualization_factor=self.config.annualization_factor,
        )

    def _maybe_open_position(self, snapshot: MarketSnapshot, feature_frame) -> dict[str, object] | None:
        if snapshot.settlement_price is not None or snapshot.observed_at >= snapshot.expiry:
            return None
        evaluation = self.engine.evaluate_snapshot(snapshot, feature_frame)
        if evaluation == (None, None):
            return None
        signal, estimate = evaluation
        if signal is None or estimate is None or signal.action == "no_action" or signal.side is None or signal.entry_price_cents is None:
            return None
        contracts = self.engine.backtest_config.default_contracts
        entry_notional = round((signal.entry_price_cents * contracts) / 100.0, 4)
        current_equity = self.engine.backtest_config.starting_bankroll + self.risk_manager.realized_pnl
        risk_check = self.risk_manager.check_entry(
            entry_notional=entry_notional,
            expiry=snapshot.expiry,
            open_position_count=len([p for p in self.position_book.positions.values() if p.status == "open"]),
            current_equity=current_equity,
        )
        if not risk_check.allowed:
            return None
        position = self.position_book.open_position(
            market_ticker=snapshot.market_ticker,
            side=signal.side,
            contracts=contracts,
            entry_time=snapshot.observed_at,
            entry_price_cents=signal.entry_price_cents,
            strategy_mode=self.engine.fusion_config.mode if self.engine.fusion_config else "single",
        )
        self.risk_manager.record_entry(entry_notional=entry_notional, expiry=snapshot.expiry)
        self.paper_broker.record_fill(
            market_ticker=position.market_ticker,
            side=f"buy_{position.side}",
            contracts=position.contracts,
            price_cents=position.entry_price_cents,
            timestamp=position.entry_time,
        )
        return {
            "market_ticker": position.market_ticker,
            "side": position.side,
            "contracts": position.contracts,
            "entry_price_cents": position.entry_price_cents,
            "edge": signal.edge,
            "model_probability": signal.model_probability,
            "reason": signal.reason,
            "estimate_model": estimate.model_name,
        }

    def _maybe_close_position(self, position: Position, snapshot: MarketSnapshot, feature_frame) -> dict[str, object] | None:
        if snapshot.settlement_price is not None or snapshot.observed_at >= snapshot.expiry:
            exit_price_cents = self.engine._settlement_price_cents(snapshot, position.side)
            realized_pnl = round(((exit_price_cents - position.entry_price_cents) * position.contracts) / 100.0, 2)
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
        )
        if decision is None or decision.action != "exit" or decision.exit_price_cents is None or decision.trigger is None:
            return None
        realized_pnl = round(((decision.exit_price_cents - position.entry_price_cents) * position.contracts) / 100.0, 2)
        closed = self.position_book.close_position(
            position.market_ticker,
            exit_time=snapshot.observed_at,
            exit_price_cents=decision.exit_price_cents,
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
            price_cents=decision.exit_price_cents,
            timestamp=snapshot.observed_at,
        )
        return {
            "market_ticker": closed.market_ticker,
            "exit_trigger": decision.trigger,
            "exit_price_cents": decision.exit_price_cents,
            "realized_pnl": realized_pnl,
            "reason": decision.reason,
        }

    def _save_state(self) -> dict[str, object]:
        open_positions = []
        closed_positions = []
        for position in self.position_book.positions.values():
            payload = asdict(position)
            for key in ("entry_time", "exit_time"):
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
        }
        self.ledger_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state
