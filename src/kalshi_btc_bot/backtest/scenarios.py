from __future__ import annotations

import pandas as pd


def clone_with_volatility_multiplier(feature_frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    cloned = feature_frame.copy()
    cloned["realized_volatility"] = cloned["realized_volatility"] * multiplier
    return cloned
