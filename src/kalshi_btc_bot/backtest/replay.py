from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from kalshi_btc_bot.data.features import build_feature_frame
from kalshi_btc_bot.types import MarketSnapshot


@dataclass(frozen=True)
class ReplayDataset:
    snapshots: list[MarketSnapshot]
    feature_frame: pd.DataFrame


def build_feature_frame_from_snapshots(
    snapshots: list[MarketSnapshot],
    *,
    volatility_window: int,
    annualization_factor: float,
) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "log_return", "realized_volatility"])

    price_rows: dict[pd.Timestamp, dict[str, float]] = {}
    for snapshot in snapshots:
        ts = pd.Timestamp(snapshot.observed_at)
        if ts not in price_rows:
            price_rows[ts] = {
                "open": snapshot.spot_price,
                "high": snapshot.spot_price,
                "low": snapshot.spot_price,
                "close": snapshot.spot_price,
                "volume": 0.0,
            }
        else:
            price_rows[ts]["high"] = max(price_rows[ts]["high"], snapshot.spot_price)
            price_rows[ts]["low"] = min(price_rows[ts]["low"], snapshot.spot_price)
            price_rows[ts]["close"] = snapshot.spot_price

    candles = pd.DataFrame.from_dict(price_rows, orient="index").sort_index()
    candles.index = pd.to_datetime(candles.index, utc=True)
    return build_feature_frame(
        candles,
        volatility_window=volatility_window,
        annualization_factor=annualization_factor,
    )


def build_replay_dataset(
    snapshots: list[MarketSnapshot],
    *,
    volatility_window: int,
    annualization_factor: float,
    observed_from: datetime | None = None,
    observed_to: datetime | None = None,
) -> ReplayDataset:
    filtered = []
    for snapshot in snapshots:
        if observed_from is not None and snapshot.observed_at < observed_from:
            continue
        if observed_to is not None and snapshot.observed_at > observed_to:
            continue
        filtered.append(snapshot)
    feature_frame = build_feature_frame_from_snapshots(
        filtered,
        volatility_window=volatility_window,
        annualization_factor=annualization_factor,
    )
    return ReplayDataset(snapshots=filtered, feature_frame=feature_frame)
