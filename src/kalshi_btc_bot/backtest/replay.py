from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

import pandas as pd

from kalshi_btc_bot.data.features import build_feature_frame
from kalshi_btc_bot.types import MarketSnapshot


@dataclass(frozen=True)
class ReplayDataset:
    snapshots: list[MarketSnapshot]
    feature_frame: pd.DataFrame


def _fingerprint_snapshots(
    snapshots: list[MarketSnapshot],
    *,
    volatility_window: int,
    annualization_factor: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(volatility_window).encode("utf-8"))
    digest.update(f"{annualization_factor:.12f}".encode("utf-8"))
    digest.update(str(len(snapshots)).encode("utf-8"))
    for snapshot in snapshots:
        digest.update(snapshot.market_ticker.encode("utf-8"))
        digest.update(snapshot.observed_at.isoformat().encode("utf-8"))
        digest.update(snapshot.expiry.isoformat().encode("utf-8"))
        digest.update(f"{snapshot.spot_price:.8f}".encode("utf-8"))
        digest.update(f"{snapshot.threshold or 0.0:.8f}".encode("utf-8"))
    return digest.hexdigest()


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
    cache_dir: str | None = None,
) -> ReplayDataset:
    filtered = []
    for snapshot in snapshots:
        if observed_from is not None and snapshot.observed_at < observed_from:
            continue
        if observed_to is not None and snapshot.observed_at > observed_to:
            continue
        filtered.append(snapshot)
    feature_frame: pd.DataFrame
    cache_path: Path | None = None
    if cache_dir:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        fingerprint = _fingerprint_snapshots(
            filtered,
            volatility_window=volatility_window,
            annualization_factor=annualization_factor,
        )
        cache_path = cache_root / f"{fingerprint}.pkl"
        if cache_path.exists():
            try:
                feature_frame = pd.read_pickle(cache_path)
                return ReplayDataset(snapshots=filtered, feature_frame=feature_frame)
            except Exception:
                cache_path.unlink(missing_ok=True)
    feature_frame = build_feature_frame_from_snapshots(
        filtered,
        volatility_window=volatility_window,
        annualization_factor=annualization_factor,
    )
    if cache_path is not None:
        feature_frame.to_pickle(cache_path)
    return ReplayDataset(snapshots=filtered, feature_frame=feature_frame)
