from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.backtest.replay import build_replay_dataset
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot


def load_recent_snapshots(
    store: SnapshotStore,
    *,
    series_ticker: str,
    lookback_hours: int = 6,
    limit: int | None = None,
) -> list[MarketSnapshot]:
    observed_from = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    return store.load_snapshots(
        series_ticker=series_ticker,
        observed_from=observed_from,
        limit=limit,
    )


def latest_snapshots_by_market(snapshots: list[MarketSnapshot]) -> list[MarketSnapshot]:
    latest: dict[str, MarketSnapshot] = {}
    for snapshot in snapshots:
        current = latest.get(snapshot.market_ticker)
        if current is None or snapshot.observed_at >= current.observed_at:
            latest[snapshot.market_ticker] = snapshot
    return sorted(latest.values(), key=lambda snapshot: snapshot.observed_at, reverse=True)


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
        volatility = engine._lookup_volatility(dataset.feature_frame, snapshot.observed_at)
        if volatility is None:
            continue
        snapshot.metadata["volatility"] = volatility
        snapshot.metadata["recent_log_return"] = engine._lookup_recent_log_return(dataset.feature_frame, snapshot.observed_at)
        signal, estimate = engine._build_signal(snapshot, volatility)
        price_cents = int(round((snapshot.yes_ask or 0.0) * 100.0)) if snapshot.yes_ask is not None else None
        minutes_to_expiry = round((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 2)
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
                "action": signal.action,
                "side": signal.side,
                "edge": round(signal.edge, 4) if signal.edge is not None else None,
                "quality_score": round(signal.quality_score, 6),
                "model_probability": round(signal.model_probability, 4),
                "market_probability": round(signal.market_probability, 4) if signal.market_probability is not None else None,
                "fair_value_cents": signal.fair_value_cents,
                "expected_value_cents": signal.expected_value_cents,
                "reason": signal.reason,
                "model_name": estimate.model_name,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["action", "quality_score", "edge", "minutes_to_expiry"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_latest_snapshot_table(snapshots: list[MarketSnapshot]) -> pd.DataFrame:
    latest = latest_snapshots_by_market(snapshots)
    rows = []
    for snapshot in latest:
        rows.append(
            {
                "market_ticker": snapshot.market_ticker,
                "observed_at": snapshot.observed_at,
                "expiry": snapshot.expiry,
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
