from __future__ import annotations

import pandas as pd


OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_candles(candles: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()
    resampled = (
        candles.sort_index()
        .resample(timeframe, label="right", closed="right")
        .agg(OHLC_AGG)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return resampled
