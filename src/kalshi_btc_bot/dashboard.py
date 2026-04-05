from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.backtest.replay import build_replay_dataset
from kalshi_btc_bot.models.repricing_target import attach_snapshot_microstructure
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot

EASTERN_TZ = ZoneInfo("America/New_York")


def format_dashboard_time(value: object) -> str:
    if value in (None, "", "-"):
        return "-"
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        raw = str(value)
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt.astimezone(EASTERN_TZ)
    return local.strftime("%b %d, %Y %I:%M:%S %p ET")


def _signal_decision_label(action: str) -> str:
    return "taken_candidate" if action != "no_action" else "skipped"


def _signal_reason_bucket(reason: str) -> str:
    normalized = str(reason or "").strip().lower()
    if "stale" in normalized:
        return "stale_data"
    if "too far from spot" in normalized:
        return "far_from_spot"
    if "liquidity threshold" in normalized:
        return "liquidity_filter"
    if "entry band" in normalized:
        return "entry_band_filter"
    if "edge below threshold" in normalized:
        return "edge_filter"
    if "exceeds threshold" in normalized:
        return "tradeable"
    return "other"


def load_recent_snapshots(
    store: SnapshotStore,
    *,
    series_ticker: str,
    lookback_hours: int = 6,
    limit: int | None = None,
) -> list[MarketSnapshot]:
    observed_from = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    return store.load_recent_rows(
        series_ticker=series_ticker,
        observed_from=observed_from,
        limit=limit,
        descending=True,
    )


def latest_snapshots_by_market(snapshots: list[MarketSnapshot]) -> list[MarketSnapshot]:
    latest: dict[str, MarketSnapshot] = {}
    for snapshot in snapshots:
        current = latest.get(snapshot.market_ticker)
        if current is None or snapshot.observed_at >= current.observed_at:
            latest[snapshot.market_ticker] = snapshot
    return sorted(latest.values(), key=lambda snapshot: snapshot.observed_at, reverse=True)


def active_snapshots_by_market(
    snapshots: list[MarketSnapshot],
    *,
    reference_time: datetime | None = None,
) -> list[MarketSnapshot]:
    now = reference_time or datetime.now(timezone.utc)
    active = [
        snapshot
        for snapshot in snapshots
        if snapshot.settlement_price is None and snapshot.expiry > now
    ]
    return latest_snapshots_by_market(active)


def build_live_signal_table(
    *,
    engine: BacktestEngine,
    snapshots: list[MarketSnapshot],
    volatility_window: int,
    annualization_factor: float,
) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()

    latest_snapshots = latest_snapshots_by_market(snapshots)
    dataset = build_replay_dataset(
        snapshots,
        volatility_window=volatility_window,
        annualization_factor=annualization_factor,
    )
    rows: list[dict[str, object]] = []
    for snapshot in latest_snapshots:
        if snapshot.observed_at >= snapshot.expiry:
            continue
        market_history = [item for item in snapshots if item.market_ticker == snapshot.market_ticker]
        market_history = sorted(market_history, key=lambda item: item.observed_at)
        snapshot_index = next((idx for idx, item in enumerate(market_history) if item.observed_at == snapshot.observed_at), len(market_history) - 1)
        volatility = engine._lookup_volatility(dataset.feature_frame, snapshot.observed_at)
        if volatility is None:
            continue
        snapshot.metadata["volatility"] = volatility
        snapshot.metadata["recent_log_return"] = engine._lookup_recent_log_return(dataset.feature_frame, snapshot.observed_at)
        attach_snapshot_microstructure(snapshot, market_history, snapshot_index)
        signal, estimate = engine._build_signal(snapshot, volatility)
        price_cents = int(round((snapshot.yes_ask or 0.0) * 100.0)) if snapshot.yes_ask is not None else None
        minutes_to_expiry = round((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 2)
        decision_label = _signal_decision_label(signal.action)
        reason_bucket = _signal_reason_bucket(signal.reason)
        rows.append(
            {
                "market_ticker": snapshot.market_ticker,
                "observed_at": snapshot.observed_at,
                "expiry": snapshot.expiry,
                "minutes_to_expiry": minutes_to_expiry,
                "spot_price": round(snapshot.spot_price, 2),
                "threshold": snapshot.threshold,
                "direction": snapshot.direction,
                "yes_ask_cents": price_cents,
                "volume": snapshot.volume,
                "open_interest": snapshot.open_interest,
                "decision": decision_label,
                "action": signal.action,
                "side": signal.side,
                "edge": round(signal.edge, 4) if signal.edge is not None else None,
                "quality_score": round(signal.quality_score, 6),
                "tier_label": signal.tier_label,
                "size_multiplier": signal.size_multiplier,
                "model_probability": round(signal.model_probability, 4),
                "market_probability": round(signal.market_probability, 4) if signal.market_probability is not None else None,
                "fair_value_cents": signal.fair_value_cents,
                "expected_value_cents": signal.expected_value_cents,
                "predicted_repricing_cents": signal.predicted_repricing_cents,
                "profit_probability": round(signal.profit_probability, 4) if signal.profit_probability is not None else None,
                "spread_cents": signal.spread_cents,
                "liquidity_penalty": signal.liquidity_penalty,
                "spread_regime": snapshot.metadata.get("spread_regime_label"),
                "liquidity_regime": snapshot.metadata.get("liquidity_regime_label"),
                "stale_quote_flag": snapshot.metadata.get("stale_quote_flag"),
                "btc_micro_jump_flag": snapshot.metadata.get("btc_micro_jump_flag"),
                "calibration_regime": snapshot.metadata.get("calibration_regime"),
                "reason_bucket": reason_bucket,
                "reason": signal.reason,
                "model_name": estimate.model_name,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ("observed_at", "expiry"):
        if column in frame.columns:
            frame[column] = frame[column].map(format_dashboard_time)
    return frame.sort_values(
        by=["decision", "quality_score", "edge", "minutes_to_expiry"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_signal_reason_table(signal_table: pd.DataFrame) -> pd.DataFrame:
    if signal_table.empty or "reason_bucket" not in signal_table.columns:
        return pd.DataFrame()
    frame = (
        signal_table.groupby(["decision", "reason_bucket", "reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["decision", "count", "reason"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return frame


def build_latest_snapshot_table(
    snapshots: list[MarketSnapshot],
    *,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    latest = active_snapshots_by_market(snapshots, reference_time=reference_time)
    rows = []
    for snapshot in latest:
        rows.append(
            {
                "market_ticker": snapshot.market_ticker,
                "observed_at": format_dashboard_time(snapshot.observed_at),
                "expiry": format_dashboard_time(snapshot.expiry),
                "minutes_to_expiry": round((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 2),
                "spot_price": round(snapshot.spot_price, 2),
                "threshold": snapshot.threshold,
                "direction": snapshot.direction,
                "yes_bid": snapshot.yes_bid,
                "yes_ask": snapshot.yes_ask,
                "volume": snapshot.volume,
                "open_interest": snapshot.open_interest,
            }
        )
    return pd.DataFrame(rows)


def load_paper_trading_state(path: str | Path) -> dict[str, object]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {
            "updated_at": None,
            "open_positions": [],
            "closed_positions": [],
            "fills": [],
            "realized_pnl": 0.0,
            "session_notional": 0.0,
            "data_feed_status": "unknown",
            "last_coinbase_error": None,
            "last_successful_feature_build_at": None,
        }
    state = json.loads(ledger_path.read_text(encoding="utf-8"))
    return _reconcile_positions_from_fills(state)


def _reconcile_positions_from_fills(state: dict[str, object]) -> dict[str, object]:
    fills = list(state.get("fills", []) or [])
    closed_positions = list(state.get("closed_positions", []) or [])
    open_positions = list(state.get("open_positions", []) or [])
    if not fills:
        return state

    known_closed_keys = {
        (
            str(position.get("market_ticker")),
            str(position.get("side")),
            str(position.get("entry_time")),
            int(position.get("contracts", 0) or 0),
        )
        for position in closed_positions
    }
    known_open_keys = {
        (
            str(position.get("market_ticker")),
            str(position.get("side")),
            str(position.get("entry_time")),
            int(position.get("contracts", 0) or 0),
        )
        for position in open_positions
    }

    by_market: dict[tuple[str, str], list[dict[str, object]]] = {}
    for fill in fills:
        side = str(fill.get("side", ""))
        if not side.startswith(("buy_", "sell_")):
            continue
        trade_side = side.split("_", 1)[1]
        key = (str(fill.get("market_ticker")), trade_side)
        by_market.setdefault(key, []).append(fill)

    synthetic_closed: list[dict[str, object]] = []
    synthetic_counter = 1
    for (market_ticker, trade_side), market_fills in by_market.items():
        ordered = sorted(market_fills, key=lambda item: str(item.get("timestamp", "")))
        entry_fill: dict[str, object] | None = None
        for fill in ordered:
            fill_side = str(fill.get("side", ""))
            if fill_side == f"buy_{trade_side}":
                entry_fill = fill
                continue
            if fill_side != f"sell_{trade_side}" or entry_fill is None:
                continue
            contracts = int(entry_fill.get("contracts", 0) or 0)
            entry_key = (
                market_ticker,
                trade_side,
                str(entry_fill.get("timestamp")),
                contracts,
            )
            if entry_key in known_closed_keys or entry_key in known_open_keys:
                entry_fill = None
                continue
            entry_price = int(entry_fill.get("price_cents", 0) or 0)
            exit_price = int(fill.get("price_cents", 0) or 0)
            entry_fees = float(entry_fill.get("fees_paid", 0.0) or 0.0)
            exit_fees = float(fill.get("fees_paid", 0.0) or 0.0)
            realized_pnl = round(((exit_price - entry_price) * contracts) / 100.0 - entry_fees - exit_fees, 2)
            synthetic_closed.append(
                {
                    "position_id": f"synthetic-pos-{synthetic_counter}",
                    "market_ticker": market_ticker,
                    "side": trade_side,
                    "contracts": contracts,
                    "entry_time": entry_fill.get("timestamp"),
                    "entry_price_cents": entry_price,
                    "strategy_mode": "synthetic_from_fills",
                    "entry_fees_paid": entry_fees,
                    "status": "closed",
                    "exit_time": fill.get("timestamp"),
                    "exit_price_cents": exit_price,
                    "exit_trigger": "reconstructed_from_fills",
                    "realized_pnl": realized_pnl,
                }
            )
            known_closed_keys.add(entry_key)
            synthetic_counter += 1
            entry_fill = None

    if not synthetic_closed:
        return state

    merged_closed = closed_positions + synthetic_closed
    merged_closed.sort(key=lambda item: str(item.get("exit_time", "")), reverse=True)
    return {
        **state,
        "closed_positions": merged_closed,
    }


def build_positions_table(positions: list[dict[str, object]]) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()
    frame = pd.DataFrame(positions)
    if "exit_time" in frame.columns and frame["exit_time"].notna().any():
        frame = frame.sort_values(by=["exit_time", "entry_time"], ascending=[False, False], na_position="last")
    elif "entry_time" in frame.columns:
        frame = frame.sort_values(by=["entry_time"], ascending=[False], na_position="last")
    preferred = [
        "position_id",
        "market_ticker",
        "side",
        "contracts",
        "entry_time",
        "entry_price_cents",
        "exit_time",
        "exit_price_cents",
        "exit_trigger",
        "realized_pnl",
        "status",
        "strategy_mode",
    ]
    columns = [column for column in preferred if column in frame.columns]
    frame = frame[columns].copy()
    for column in ("entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = frame[column].map(format_dashboard_time)
    return frame.reset_index(drop=True)


def build_fills_table(fills: list[dict[str, object]]) -> pd.DataFrame:
    if not fills:
        return pd.DataFrame()
    frame = pd.DataFrame(fills)
    preferred = ["timestamp", "market_ticker", "side", "contracts", "price_cents", "fees_paid"]
    columns = [column for column in preferred if column in frame.columns]
    frame = frame[columns]
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
        frame["timestamp"] = frame["timestamp"].map(format_dashboard_time)
    return frame
