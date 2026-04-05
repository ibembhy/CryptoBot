from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_TTE_BUCKETS = {
    "KXBTCD": [2, 5, 10, 15, 20, 30, 45, 60],
    "KXBTC15M": [1, 2, 4, 6, 8, 12, 14],
}

DISTANCE_BUCKETS_BPS = [0, 50, 100, 200, 350, 500]
VOLUME_BUCKETS = [-0.5, 0.5, 100, 1000, np.inf]
VOLUME_BUCKET_LABELS = ["0", "1-99", "100-999", "1000+"]
TREND_FLAT_THRESHOLD = 0.0005  # 0.05% proportional move over 5 snapshots.

SERIES_NEAR_MONEY_BPS = {
    "KXBTCD": 500.0,
    "KXBTC15M": 300.0,
}


def deduplicate_snapshots(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse repeated market snapshots with a deterministic quality ranking.

    Returns:
      cleaned_frame, duplicate_audit_frame
    """
    if frame.empty:
        return frame.copy(), pd.DataFrame()

    data = frame.copy()
    data["_row_completeness"] = data.apply(_completeness_score, axis=1)
    data["_has_crossed_book"] = (
        (data["yes_bid"].notna() & data["yes_ask"].notna() & (data["yes_bid"] > data["yes_ask"]))
        | (data["no_bid"].notna() & data["no_ask"].notna() & (data["no_bid"] > data["no_ask"]))
    )
    data["_volume_rank"] = pd.to_numeric(data.get("volume"), errors="coerce").fillna(-1.0)
    data["_open_interest_rank"] = pd.to_numeric(data.get("open_interest"), errors="coerce").fillna(-1.0)
    if "id" not in data.columns:
        data["id"] = np.arange(len(data))

    group_cols = ["market_ticker", "observed_at"]
    duplicate_mask = data.duplicated(group_cols, keep=False)
    duplicate_audit = data.loc[duplicate_mask].copy()

    # Remove byte-for-byte duplicates first, ignoring synthetic ranking fields.
    stable_compare_cols = [col for col in data.columns if not col.startswith("_") and col != "id"]
    data = data.drop_duplicates(subset=stable_compare_cols, keep="last")

    ranked = (
        data.sort_values(
            by=[
                "market_ticker",
                "observed_at",
                "_has_crossed_book",
                "_row_completeness",
                "_volume_rank",
                "_open_interest_rank",
                "id",
            ],
            ascending=[True, True, True, False, False, False, False],
        )
        .drop_duplicates(subset=group_cols, keep="first")
        .drop(columns=["_row_completeness", "_has_crossed_book", "_volume_rank", "_open_interest_rank"])
        .reset_index(drop=True)
    )
    return ranked, duplicate_audit.reset_index(drop=True)


def _completeness_score(row: pd.Series) -> int:
    score_cols = [
        "yes_bid",
        "yes_ask",
        "no_bid",
        "no_ask",
        "volume",
        "open_interest",
        "implied_probability",
        "threshold",
        "spot_price",
    ]
    return int(sum(pd.notna(row.get(col)) for col in score_cols))


def build_global_spot_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a global BTC spot feature table from deduped snapshots and join it back."""
    if frame.empty:
        return frame.copy(), pd.DataFrame()

    spot = (
        frame[["observed_at", "spot_price"]]
        .dropna(subset=["observed_at", "spot_price"])
        .sort_values("observed_at")
        .groupby("observed_at", as_index=False)
        .agg(spot_price=("spot_price", "median"))
    )
    spot["spot_return"] = spot["spot_price"].pct_change()
    spot["spot_log_return"] = np.log(spot["spot_price"] / spot["spot_price"].shift(1))
    spot["momentum_5"] = spot["spot_price"].pct_change(periods=5)
    spot["direction_5"] = np.select(
        [spot["momentum_5"] > TREND_FLAT_THRESHOLD, spot["momentum_5"] < -TREND_FLAT_THRESHOLD],
        ["up", "down"],
        default="flat",
    )
    spot["rolling_vol_20"] = spot["spot_log_return"].rolling(20, min_periods=5).std(ddof=0)
    vol_quantiles = spot["rolling_vol_20"].dropna().quantile([1 / 3, 2 / 3]).to_dict()
    low_cut = vol_quantiles.get(1 / 3, np.nan)
    high_cut = vol_quantiles.get(2 / 3, np.nan)
    spot["vol_regime"] = np.select(
        [
            spot["rolling_vol_20"].isna(),
            spot["rolling_vol_20"] <= low_cut,
            spot["rolling_vol_20"] <= high_cut,
        ],
        ["unknown", "low", "mid"],
        default="high",
    )

    enriched = frame.merge(
        spot[["observed_at", "momentum_5", "direction_5", "rolling_vol_20", "vol_regime"]],
        on="observed_at",
        how="left",
    )
    return enriched, spot


def add_market_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    data["tte_min"] = (data["expiry"] - data["observed_at"]).dt.total_seconds() / 60.0
    data["dist_bps"] = ((data["threshold"] - data["spot_price"]).abs() / data["spot_price"]) * 10000.0
    data["spread_cents_yes"] = (data["yes_ask"] - data["yes_bid"]) * 100.0
    data["spread_cents_no"] = (data["no_ask"] - data["no_bid"]) * 100.0
    data["hour_utc"] = data["observed_at"].dt.hour
    data["market_favored_side"] = np.where(data["yes_ask"] > 0.5, "yes", "no")
    data["distance_bucket_bps"] = pd.cut(
        data["dist_bps"],
        bins=DISTANCE_BUCKETS_BPS,
        labels=[f"{DISTANCE_BUCKETS_BPS[i]}-{DISTANCE_BUCKETS_BPS[i + 1]}" for i in range(len(DISTANCE_BUCKETS_BPS) - 1)],
        right=False,
        include_lowest=True,
    )
    data["volume_bucket"] = pd.cut(
        pd.to_numeric(data["volume"], errors="coerce").fillna(0),
        bins=VOLUME_BUCKETS,
        labels=VOLUME_BUCKET_LABELS,
        right=True,
        include_lowest=True,
    )
    data["trend_regime"] = data["direction_5"]
    return data


def assign_series_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    tte_bucket = []
    near_money_flag = []
    for _, row in data.iterrows():
        series = row["series_ticker"]
        tte_bucket.append(_label_bucket(row["tte_min"], SERIES_TTE_BUCKETS[series]))
        near_money_flag.append(float(row["dist_bps"]) < SERIES_NEAR_MONEY_BPS[series] if pd.notna(row["dist_bps"]) else False)
    data["tte_bucket"] = tte_bucket
    data["is_near_money"] = near_money_flag
    return data


def _label_bucket(value: float, boundaries: list[float]) -> str | None:
    if pd.isna(value):
        return None
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if lower <= value < upper:
            return f"{int(lower)}-{int(upper)}"
    if value == boundaries[-1]:
        return f"{int(boundaries[-2])}-{int(boundaries[-1])}"
    return None
