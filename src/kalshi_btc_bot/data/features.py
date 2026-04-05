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
    frame["log_return_3"] = frame["log_return"].rolling(3, min_periods=3).sum()
    frame["log_return_5"] = frame["log_return"].rolling(5, min_periods=5).sum()
    frame["price_acceleration"] = frame["log_return"] - frame["log_return"].shift(1)
    frame["realized_volatility_1"] = frame["log_return"].abs() * np.sqrt(annualization_factor)
    frame["realized_volatility_3"] = frame["log_return"].rolling(3, min_periods=3).std(ddof=0) * np.sqrt(annualization_factor)
    frame["realized_volatility_5"] = frame["log_return"].rolling(5, min_periods=5).std(ddof=0) * np.sqrt(annualization_factor)
    frame["realized_volatility"] = (
        frame["log_return"]
        .rolling(volatility_window, min_periods=volatility_window)
        .std(ddof=0)
        * np.sqrt(annualization_factor)
    )
    vol_window = max(3, volatility_window // 2)
    frame["vol_of_vol"] = frame["realized_volatility"].rolling(vol_window, min_periods=vol_window).std(ddof=0)
    frame["btc_micro_jump_flag"] = (frame["log_return"].abs() * 10_000.0 >= 10.0).astype(float)
    return frame


def attach_time_to_expiry(feature_frame: pd.DataFrame, expiry: pd.Timestamp) -> pd.DataFrame:
    frame = feature_frame.copy()
    delta = (pd.Timestamp(expiry) - frame.index).total_seconds().clip(lower=0)
    frame["time_to_expiry_years"] = delta / (365.0 * 24.0 * 60.0 * 60.0)
    return frame
