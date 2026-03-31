from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_frame(
    candles: pd.DataFrame,
    *,
    volatility_window: int,
    annualization_factor: float,
) -> pd.DataFrame:
    if candles.empty:
        return candles.copy()
    frame = candles.copy().sort_index()
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["realized_volatility"] = (
        frame["log_return"]
        .rolling(volatility_window, min_periods=volatility_window)
        .std(ddof=0)
        * np.sqrt(annualization_factor)
    )
    return frame


def attach_time_to_expiry(feature_frame: pd.DataFrame, expiry: pd.Timestamp) -> pd.DataFrame:
    frame = feature_frame.copy()
    delta = (pd.Timestamp(expiry) - frame.index).total_seconds().clip(lower=0)
    frame["time_to_expiry_years"] = delta / (365.0 * 24.0 * 60.0 * 60.0)
    return frame
