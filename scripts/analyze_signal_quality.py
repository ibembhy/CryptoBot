"""
Block 3 - Items 12 and 13: Signal quality audit + missed opportunity analysis.

Prerequisites: run pull_settled_markets.py first to generate:
  data/settled_markets_kxbtcd.csv
  data/settled_markets_kxbtc15m.csv

Key design choice:
  Entry is simulated at (close_time - entry_offset_min) — the midpoint of the
  bot's preferred entry window. For KXBTCD: 10-55 min, so simulate at -25 min.
  For KXBTC15M: 8-20 min, so simulate at -12 min. BTC spot is fetched from
  Coinbase 1-min candles at that time.

Item 12: For every settled market, compute GBM model probability at simulated
  entry vs actual settlement. Does the signal have edge?

Item 13: For every market where the model had edge (|edge| >= MIN_EDGE),
  identify which filter(s) would have blocked it.

Run from repo root:
  python scripts/analyze_signal_quality.py [--series KXBTCD] [--vol 0.60]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.models.gbm_threshold import terminal_probability_above

DATA_DIR = Path(__file__).parent.parent / "data"

# ── signal config (mirrors default.toml) ──────────────────────────────────
MIN_EDGE = 0.05
MAX_NEAR_MONEY_BPS = 200.0
PREFERRED_MINUTES = {
    "KXBTCD":   (10, 55),
    "KXBTC15M": (8, 20),
}
VOL_WINDOW_HOURS = 20     # rolling hours for realized vol
ANNUALIZATION = math.sqrt(365 * 24)  # hourly → annual


def entry_offset_min(series: str) -> int:
    lo, hi = PREFERRED_MINUTES[series]
    return (lo + hi) // 2  # midpoint of preferred window


def load_markets(series: str) -> pd.DataFrame:
    path = DATA_DIR / f"settled_markets_{series.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run pull_settled_markets.py first.")
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["floor_strike"] = pd.to_numeric(df["floor_strike"], errors="coerce")
    df["expiration_value"] = pd.to_numeric(df["expiration_value"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["actual"] = (df["result"] == "yes").astype(int)
    df["tte_minutes"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 60
    return df.dropna(subset=["floor_strike", "open_time", "close_time", "actual"])


def fetch_btc_1min(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch 1-min BTC candles in chunks to avoid Coinbase per-call limits."""
    print(f"Fetching BTC 1-min candles {start.date()} -> {end.date()}...")
    client = CoinbaseClient()
    chunk_hours = 4          # Coinbase returns ≤300 candles; 4h = 240 candles
    frames = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(hours=chunk_hours), end)
        try:
            chunk = client.fetch_candles(start=cursor, end=chunk_end, timeframe="1m")
            if not chunk.empty:
                frames.append(chunk)
        except Exception as exc:
            print(f"  Warning: candle fetch failed for {cursor} - {chunk_end}: {exc}")
        cursor = chunk_end

    if not frames:
        raise RuntimeError("No candle data returned from Coinbase")
    candles = pd.concat(frames).sort_index().drop_duplicates()
    # Ensure UTC-aware index
    if candles.index.tz is None:
        candles.index = candles.index.tz_localize("UTC")
    print(f"  {len(candles):,} 1-min candles loaded.")
    return candles


def build_vol_series(candles_1m: pd.DataFrame) -> pd.Series:
    """Compute 20-hour rolling annualized vol from 1-min close prices."""
    log_ret = (candles_1m["close"] / candles_1m["close"].shift(1)).apply(math.log)
    # Resample to hourly for the vol calc
    hourly_log_ret = log_ret.resample("1h").sum()
    rolling_vol = hourly_log_ret.rolling(VOL_WINDOW_HOURS, min_periods=5).std(ddof=0) * ANNUALIZATION
    # Forward-fill to 1-min resolution
    return rolling_vol.reindex(candles_1m.index, method="ffill")


def attach_spot_and_vol(
    df: pd.DataFrame,
    candles_1m: pd.DataFrame,
    vol_series: pd.Series,
    offset_min: int,
    fixed_vol: float | None,
) -> pd.DataFrame:
    """For each market, get BTC spot and vol at (close_time - offset_min).
    Uses merge_asof for O(n log n) instead of per-row lookups."""
    df = df.copy()
    df["entry_time"] = df["close_time"] - pd.Timedelta(minutes=offset_min)
    df["tte_at_entry_min"] = offset_min

    # Build lookup frame from candle data
    lookup = candles_1m[["close"]].rename(columns={"close": "spot_at_entry"}).copy()
    if not fixed_vol:
        lookup["vol_at_entry"] = vol_series.reindex(lookup.index).fillna(0.65)
    else:
        lookup["vol_at_entry"] = fixed_vol

    # Sort both by the join key
    df_sorted = df.sort_values("entry_time")
    lookup_sorted = lookup.sort_index()

    merged = pd.merge_asof(
        df_sorted,
        lookup_sorted.reset_index().rename(columns={"timestamp": "entry_time"}),
        on="entry_time",
        direction="backward",
    )
    # Restore original order
    merged = merged.sort_index()
    return merged.dropna(subset=["spot_at_entry", "vol_at_entry"])


def compute_model_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tte_years"] = df["tte_at_entry_min"] / (365 * 24 * 60)

    df["model_prob_yes"] = df.apply(
        lambda r: terminal_probability_above(
            spot_price=r["spot_at_entry"],
            target_price=r["floor_strike"],
            time_to_expiry_years=r["tte_years"],
            volatility=r["vol_at_entry"],
        ),
        axis=1,
    )

    # edge = model_prob_yes - 0.50 (signed: positive → buy YES, negative → buy NO)
    df["edge"] = df["model_prob_yes"] - 0.50
    df["abs_edge"] = df["edge"].abs()

    # distance from strike at entry (bps, positive = BTC above strike)
    df["distance_bps"] = (df["spot_at_entry"] - df["floor_strike"]) / df["spot_at_entry"] * 10_000
    df["near_money"] = df["distance_bps"].abs() < MAX_NEAR_MONEY_BPS

    return df


def classify_filters(df: pd.DataFrame, series: str) -> pd.DataFrame:
    lo, hi = PREFERRED_MINUTES[series]
    df = df.copy()
    # At simulated entry time, actual tte = offset_min (fixed). Filter is on
    # tte at entry vs preferred window. Since we simulate at the midpoint,
    # tte_at_entry_min is always within the window by construction. Instead,
    # check whether the market's TOTAL duration falls in scope.
    # More usefully: flag each filter dimension independently.
    df["filter_near_money"] = df["near_money"]
    df["filter_low_edge"] = df["abs_edge"] < MIN_EDGE
    # Time window filter: would the bot have been running during this market?
    # KXBTCD opens 1h before close; bot enters 10-55 min before. Always in window.
    # This filter is moot when entry is simulated at mid-window. Flag as False.
    df["filter_time_window"] = False
    df["any_filter"] = df["filter_near_money"] | df["filter_low_edge"]
    df["would_trade"] = ~df["any_filter"] & (df["volume"] > 0)
    return df


def print_calibration_table(df: pd.DataFrame, series: str, offset_min: int) -> None:
    print(f"\n{'='*65}")
    print(f"  ITEM 12: Signal Calibration — {series}")
    print(f"  Entry simulated at close_time - {offset_min} min | vol from Coinbase")
    print(f"{'='*65}")

    total = len(df)
    strong = df[df["abs_edge"] >= MIN_EDGE]
    print(f"  Total markets:            {total:,}")
    print(f"  |edge| >= {MIN_EDGE} (actionable): {len(strong):,} ({100*len(strong)/total:.1f}%)")

    if strong.empty:
        print("  No actionable markets.")
        return

    # --- Overall calibration across ALL bins ---
    strong_c = strong.copy()
    strong_c["prob_bin"] = pd.cut(
        strong_c["model_prob_yes"],
        bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
    )
    stats = strong_c.groupby("prob_bin", observed=True).agg(
        count=("actual", "count"),
        actual_win_rate=("actual", "mean"),
        avg_model_prob=("model_prob_yes", "mean"),
    ).reset_index()

    print(f"\n  Calibration table (all markets with |edge|>={MIN_EDGE}):")
    print(f"  {'Model P(YES)':<20} {'N':>7} {'Actual YES%':>12} {'Model prob':>11}")
    print(f"  {'-'*53}")
    for _, row in stats.iterrows():
        flag = " <-- uncertain zone" if 0.3 <= row["avg_model_prob"] <= 0.7 else ""
        print(f"  {str(row['prob_bin']):<20} {row['count']:>7,} "
              f"{row['actual_win_rate']*100:>11.1f}% "
              f"{row['avg_model_prob']:>11.3f}{flag}")

    # --- Directional accuracy ---
    strong_c["pred_yes"] = strong_c["model_prob_yes"] > 0.50
    correct = (strong_c["pred_yes"] == (strong_c["actual"] == 1)).mean()
    naive_no = 1 - strong_c["actual"].mean()
    print(f"\n  Directional accuracy:  {correct*100:.1f}%")
    print(f"  Naive 'always NO':     {naive_no*100:.1f}%")
    print(f"  Model lift over naive: {(correct - naive_no)*100:+.1f}pp")

    # --- Near-money sub-analysis ---
    near = strong_c[strong_c["near_money"]]
    far = strong_c[~strong_c["near_money"]]
    print(f"\n  Near-money markets (|dist| < {MAX_NEAR_MONEY_BPS:.0f}bps): {len(near):,}")
    if len(near):
        near_acc = (near["pred_yes"] == (near["actual"] == 1)).mean()
        print(f"    Accuracy: {near_acc*100:.1f}%  |  YES rate: {near['actual'].mean()*100:.1f}%")
        yes_n = near[near["pred_yes"]]
        no_n = near[~near["pred_yes"]]
        if len(yes_n):
            print(f"    YES trades: win={yes_n['actual'].mean()*100:.1f}%")
        if len(no_n):
            print(f"    NO  trades: win={(1-no_n['actual'].mean())*100:.1f}%")

    print(f"\n  Far-from-strike markets (|dist| >= {MAX_NEAR_MONEY_BPS:.0f}bps): {len(far):,}")
    if len(far):
        far_acc = (far["pred_yes"] == (far["actual"] == 1)).mean()
        print(f"    Accuracy: {far_acc*100:.1f}%  (trivially easy - market price ~1c or 99c)")


def print_missed_opportunities(df: pd.DataFrame, series: str) -> None:
    print(f"\n{'='*65}")
    print(f"  ITEM 13: Missed Opportunity Analysis — {series}")
    print(f"  Markets with strong signal that filters would block")
    print(f"{'='*65}")

    strong = df[df["abs_edge"] >= MIN_EDGE]
    blocked_nm = strong[strong["filter_near_money"]]
    blocked_le = strong[strong["filter_low_edge"]]
    passed = strong[~strong["any_filter"]]

    print(f"\n  Strong-signal markets:          {len(strong):,}")
    print(f"  Blocked by near-money filter:   {len(blocked_nm):,} ({100*len(blocked_nm)/max(len(strong),1):.1f}%)")
    print(f"  Blocked by low-edge filter:     {len(blocked_le):,} ({100*len(blocked_le)/max(len(strong),1):.1f}%)")
    print(f"  Passed all filters:             {len(passed):,}")

    # --- Near-money analysis (primary filter cost) ---
    if len(blocked_nm):
        yes_nm = blocked_nm[blocked_nm["model_prob_yes"] > 0.50]
        no_nm = blocked_nm[blocked_nm["model_prob_yes"] <= 0.50]
        print(f"\n  Near-money blocked markets breakdown:")
        print(f"    Avg |distance_bps|: {blocked_nm['distance_bps'].abs().mean():.0f}")
        print(f"    Avg |edge|:         {blocked_nm['abs_edge'].mean():.3f}")
        if len(yes_nm):
            print(f"    YES-side: {len(yes_nm):,} markets, "
                  f"actual YES rate = {yes_nm['actual'].mean()*100:.1f}%")
        if len(no_nm):
            print(f"    NO-side:  {len(no_nm):,} markets, "
                  f"actual NO  rate = {(1-no_nm['actual'].mean())*100:.1f}%")

    # --- Volume analysis ---
    has_vol = strong[strong["volume"] > 0]
    no_vol = strong[strong["volume"] == 0]
    print(f"\n  Volume breakdown (strong-signal):")
    print(f"    Had volume > 0:  {len(has_vol):,}  ({100*len(has_vol)/max(len(strong),1):.1f}%)")
    print(f"    Zero volume:     {len(no_vol):,}  ({100*len(no_vol)/max(len(strong),1):.1f}%)")
    if len(no_vol):
        print(f"    (Zero-vol markets often untradeable regardless of filter)")

    # --- Passed-filter accuracy ---
    if len(passed):
        yes_p = passed[passed["model_prob_yes"] > 0.50]
        no_p = passed[passed["model_prob_yes"] <= 0.50]
        print(f"\n  Passed-filter markets accuracy:")
        if len(yes_p):
            print(f"    YES trades: {len(yes_p):,}, win={yes_p['actual'].mean()*100:.1f}%")
        if len(no_p):
            print(f"    NO  trades: {len(no_p):,}, win={(1-no_p['actual'].mean())*100:.1f}%")


def run(series: str, fixed_vol: float | None) -> None:
    offset = entry_offset_min(series)
    print(f"\n{'#'*65}")
    print(f"  {series} | Simulated entry at close_time - {offset} min")
    print(f"{'#'*65}")

    df = load_markets(series)
    print(f"  {len(df):,} markets loaded.")

    # Fetch 1-min BTC data for the full date range
    start = (df["close_time"].min() - pd.Timedelta(hours=2)).to_pydatetime().replace(tzinfo=timezone.utc)
    end = (df["close_time"].max() + pd.Timedelta(hours=1)).to_pydatetime().replace(tzinfo=timezone.utc)
    candles_1m = fetch_btc_1min(start, end)

    vol_series = build_vol_series(candles_1m)

    df = attach_spot_and_vol(df, candles_1m, vol_series, offset_min=offset, fixed_vol=fixed_vol)
    print(f"  {len(df):,} markets matched to BTC spot.")

    df = compute_model_signals(df)
    df = classify_filters(df, series)

    print_calibration_table(df, series, offset)
    print_missed_opportunities(df, series)

    # Save enriched dataset
    out = DATA_DIR / f"analysis_{series.lower()}.csv"
    keep = [
        "ticker", "open_time", "close_time", "entry_time", "floor_strike",
        "expiration_value", "result", "actual", "volume", "tte_minutes",
        "tte_at_entry_min", "spot_at_entry", "vol_at_entry",
        "model_prob_yes", "edge", "distance_bps", "near_money",
        "filter_near_money", "filter_low_edge", "any_filter", "would_trade",
    ]
    df[keep].to_csv(out, index=False)
    print(f"\n  Enriched dataset -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal quality audit and missed opportunity analysis")
    parser.add_argument("--series", type=str, default=None, help="KXBTCD or KXBTC15M (default: both)")
    parser.add_argument("--vol", type=float, default=None,
                        help="Fixed annual vol override (e.g. 0.60). Default: realized vol from Coinbase.")
    args = parser.parse_args()

    series_list = [args.series] if args.series else ["KXBTCD", "KXBTC15M"]
    for s in series_list:
        run(s, args.vol)

    print("\nDone.")


if __name__ == "__main__":
    main()
