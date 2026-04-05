"""
Block 3 - Items 14, 15, 16: Naive baseline, NO-side analysis, time-of-day patterns.

Reads the enriched analysis CSV produced by analyze_signal_quality.py:
  data/analysis_kxbtcd.csv

Item 14: Naive baseline — buy any YES 40-60c with 10+ min to expiry.
         Compare win rate and ROI vs model.
         (We approximate market price as 50c for fair-value comparison;
          we also use model_prob as a proxy for where the market would price it.)

Item 15: NO-side analysis — does the model have edge on NO trades specifically?
         Check near-money NO markets, vol conditions, time patterns.

Item 16: Time-of-day and volatility regime — when is edge concentrated?

Run from repo root:
  python scripts/analyze_items_14_16.py [--series KXBTCD]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"

MIN_EDGE = 0.05
HIGH_VOL_THRESHOLD = 0.70   # annual vol above this = high-vol regime
LOW_VOL_THRESHOLD = 0.45    # annual vol below this = low-vol regime


def load_analysis(series: str) -> pd.DataFrame:
    path = DATA_DIR / f"analysis_{series.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run analyze_signal_quality.py first.")
    df = pd.read_csv(path, parse_dates=["open_time", "close_time", "entry_time"])
    for col in ["open_time", "close_time", "entry_time"]:
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")
    # Recompute derived columns not saved to CSV
    df["edge"] = df["model_prob_yes"] - 0.50
    df["abs_edge"] = df["edge"].abs()
    return df


# ── ITEM 14: Naive baseline ──────────────────────────────────────────────

def print_naive_baseline(df: pd.DataFrame, series: str) -> None:
    print(f"\n{'='*65}")
    print(f"  ITEM 14: Naive Baseline Comparison — {series}")
    print(f"  Strategy: buy YES whenever model_prob in [0.40, 0.60]")
    print(f"  (proxy for 40-60c market: where model is uncertain)")
    print(f"{'='*65}")

    # Naive: buy YES on any market where model_prob is 40-60%
    # These are the "coin-flip" zone markets where a naive buyer would be indifferent
    naive = df[(df["model_prob_yes"] >= 0.40) & (df["model_prob_yes"] <= 0.60)].copy()
    naive_vol = naive[naive["volume"] > 0]

    print(f"\n  40-60% model_prob markets (all):     {len(naive):,}")
    print(f"  40-60% model_prob markets (w/ vol):  {len(naive_vol):,}")

    if len(naive_vol):
        # Naive: always buy YES
        naive_yes_wr = naive_vol["actual"].mean()
        naive_yes_roi = naive_yes_wr - 0.50

        # Model: buy YES if edge > 0, buy NO if edge < 0 (within this zone)
        model_yes = naive_vol[naive_vol["edge"] > 0]
        model_no = naive_vol[naive_vol["edge"] <= 0]
        model_yes_wr = model_yes["actual"].mean() if len(model_yes) else float("nan")
        model_no_wr = (1 - model_no["actual"].mean()) if len(model_no) else float("nan")
        model_trades = len(model_yes) + len(model_no)
        model_wins = (
            (model_yes["actual"].sum() if len(model_yes) else 0) +
            ((1 - model_no["actual"]).sum() if len(model_no) else 0)
        )
        model_overall_wr = model_wins / model_trades if model_trades else float("nan")

        print(f"\n  Strategy comparison (markets with volume, at 50c):")
        print(f"  {'Strategy':<35} {'Trades':>8} {'Win%':>8} {'ROI/trade':>10}")
        print(f"  {'-'*63}")
        print(f"  {'Naive: always buy YES':<35} {len(naive_vol):>8,} "
              f"{naive_yes_wr*100:>7.1f}% {naive_yes_roi*100:>+9.1f}%")
        if len(model_yes):
            print(f"  {'Model YES (edge>0)':<35} {len(model_yes):>8,} "
                  f"{model_yes_wr*100:>7.1f}% {(model_yes_wr-0.50)*100:>+9.1f}%")
        if len(model_no):
            print(f"  {'Model NO  (edge<0)':<35} {len(model_no):>8,} "
                  f"{model_no_wr*100:>7.1f}% {(model_no_wr-0.50)*100:>+9.1f}%")
        print(f"  {'Model combined':<35} {model_trades:>8,} "
              f"{model_overall_wr*100:>7.1f}% {(model_overall_wr-0.50)*100:>+9.1f}%")

    # Wider naive: all markets with volume (buy YES always)
    with_vol = df[df["volume"] > 0]
    if len(with_vol):
        all_naive_wr = with_vol["actual"].mean()
        print(f"\n  All-market naive YES (any vol, any prob): "
              f"win={all_naive_wr*100:.1f}%, roi={100*(all_naive_wr-0.50):+.1f}%/trade")

    # Model on all-vol markets
    model_all_yes = with_vol[with_vol["edge"] > MIN_EDGE]
    model_all_no = with_vol[with_vol["edge"] < -MIN_EDGE]
    if len(model_all_yes):
        wr = model_all_yes["actual"].mean()
        print(f"  Model YES (edge>{MIN_EDGE}, any vol):          "
              f"win={wr*100:.1f}%, roi={100*(wr-0.50):+.1f}%/trade  n={len(model_all_yes):,}")
    if len(model_all_no):
        wr = 1 - model_all_no["actual"].mean()
        print(f"  Model NO  (edge<-{MIN_EDGE}, any vol):          "
              f"win={wr*100:.1f}%, roi={100*(wr-0.50):+.1f}%/trade  n={len(model_all_no):,}")


# ── ITEM 15: NO-side analysis ────────────────────────────────────────────

def print_no_side_analysis(df: pd.DataFrame, series: str) -> None:
    print(f"\n{'='*65}")
    print(f"  ITEM 15: NO-Side Analysis — {series}")
    print(f"  Does the model have edge on NO trades?")
    print(f"{'='*65}")

    with_vol = df[df["volume"] > 0]
    no_trades = with_vol[with_vol["edge"] < -MIN_EDGE]
    yes_trades = with_vol[with_vol["edge"] > MIN_EDGE]

    print(f"\n  Markets with volume:        {len(with_vol):,}")
    print(f"  YES signals (edge>{MIN_EDGE}):  {len(yes_trades):,}")
    print(f"  NO  signals (edge<-{MIN_EDGE}): {len(no_trades):,}")

    if len(no_trades):
        no_wr = 1 - no_trades["actual"].mean()
        yes_wr = yes_trades["actual"].mean() if len(yes_trades) else float("nan")
        print(f"\n  YES win rate: {yes_wr*100:.1f}%")
        print(f"  NO  win rate: {no_wr*100:.1f}%")

        # Near-money NO trades
        nm_no = no_trades[no_trades["near_money"]]
        far_no = no_trades[~no_trades["near_money"]]
        print(f"\n  NO trades by distance:")
        if len(nm_no):
            nm_no_wr = 1 - nm_no["actual"].mean()
            print(f"    Near-money (<200bps): {len(nm_no):,}  win={nm_no_wr*100:.1f}%")
        if len(far_no):
            far_no_wr = 1 - far_no["actual"].mean()
            print(f"    Far-from-strike:      {len(far_no):,}  win={far_no_wr*100:.1f}%")

        # NO trades by vol regime
        print(f"\n  NO trades by volatility regime (annual vol):")
        low_no = no_trades[no_trades["vol_at_entry"] < LOW_VOL_THRESHOLD]
        mid_no = no_trades[(no_trades["vol_at_entry"] >= LOW_VOL_THRESHOLD) &
                           (no_trades["vol_at_entry"] < HIGH_VOL_THRESHOLD)]
        hi_no = no_trades[no_trades["vol_at_entry"] >= HIGH_VOL_THRESHOLD]
        for label, subset in [
            (f"Low vol   (<{LOW_VOL_THRESHOLD})", low_no),
            (f"Mid vol   ({LOW_VOL_THRESHOLD}-{HIGH_VOL_THRESHOLD})", mid_no),
            (f"High vol  (>{HIGH_VOL_THRESHOLD})", hi_no),
        ]:
            if len(subset):
                wr = 1 - subset["actual"].mean()
                print(f"    {label}: n={len(subset):,}  win={wr*100:.1f}%")

    # Current config ignores NO on near-money — is that right?
    nm_no_all = df[df["near_money"] & (df["edge"] < -MIN_EDGE) & (df["volume"] > 0)]
    if len(nm_no_all):
        wr = 1 - nm_no_all["actual"].mean()
        print(f"\n  Near-money NO (currently blocked by filter): {len(nm_no_all):,}  win={wr*100:.1f}%")
        print(f"  -> {'WORTH UNBLOCKING' if wr >= 0.55 else 'marginal / leave blocked'}")


# ── ITEM 16: Time-of-day and volatility regime ───────────────────────────

def print_time_and_regime(df: pd.DataFrame, series: str) -> None:
    print(f"\n{'='*65}")
    print(f"  ITEM 16: Time-of-Day & Volatility Regime — {series}")
    print(f"  When is edge concentrated?")
    print(f"{'='*65}")

    with_vol = df[df["volume"] > 0].copy()
    strong = with_vol[with_vol["abs_edge"] >= MIN_EDGE]

    if strong.empty:
        print("  No strong-signal markets with volume.")
        return

    strong = strong.copy()
    strong["close_hour_utc"] = strong["close_time"].dt.hour
    strong["pred_yes"] = strong["model_prob_yes"] > 0.50

    # Build correct/won column
    strong["won"] = (
        (strong["pred_yes"] & (strong["actual"] == 1)) |
        (~strong["pred_yes"] & (strong["actual"] == 0))
    )

    # ── Time-of-day accuracy ──
    print(f"\n  Win rate by hour of close (UTC), strong-signal markets with volume:")
    print(f"  {'Hour (UTC)':<12} {'N':>7} {'Win%':>8} {'YES%':>8} {'Avg|edge|':>10}")
    print(f"  {'-'*48}")
    hourly = strong.groupby("close_hour_utc").agg(
        count=("won", "count"),
        win_rate=("won", "mean"),
        yes_rate=("actual", "mean"),
        avg_edge=("abs_edge", "mean"),
    ).reset_index()
    for _, row in hourly.iterrows():
        flag = " *" if row["win_rate"] >= 0.70 else ""
        print(f"  {int(row['close_hour_utc']):02d}:00          "
              f"{row['count']:>7,} {row['win_rate']*100:>7.1f}% "
              f"{row['yes_rate']*100:>7.1f}% {row['avg_edge']:>10.3f}{flag}")

    # ── Best and worst hours ──
    best = hourly.nlargest(3, "win_rate")
    worst = hourly.nsmallest(3, "win_rate")
    print(f"\n  Top-3 hours (UTC): " +
          ", ".join(f"{int(r['close_hour_utc']):02d}:00 ({r['win_rate']*100:.0f}%)"
                    for _, r in best.iterrows()))
    print(f"  Bot-3 hours (UTC): " +
          ", ".join(f"{int(r['close_hour_utc']):02d}:00 ({r['win_rate']*100:.0f}%)"
                    for _, r in worst.iterrows()))

    # ── Vol regime breakdown ──
    print(f"\n  Win rate by volatility regime:")
    regimes = [
        (f"Low  (<{LOW_VOL_THRESHOLD})", strong["vol_at_entry"] < LOW_VOL_THRESHOLD),
        (f"Mid  ({LOW_VOL_THRESHOLD}-{HIGH_VOL_THRESHOLD})",
         (strong["vol_at_entry"] >= LOW_VOL_THRESHOLD) &
         (strong["vol_at_entry"] < HIGH_VOL_THRESHOLD)),
        (f"High (>{HIGH_VOL_THRESHOLD})", strong["vol_at_entry"] >= HIGH_VOL_THRESHOLD),
    ]
    print(f"  {'Regime':<22} {'N':>7} {'Win%':>8} {'Avg vol':>9} {'Avg|edge|':>10}")
    print(f"  {'-'*57}")
    for label, mask in regimes:
        sub = strong[mask]
        if len(sub):
            wr = sub["won"].mean()
            avg_vol = sub["vol_at_entry"].mean()
            avg_edge = sub["abs_edge"].mean()
            print(f"  {label:<22} {len(sub):>7,} {wr*100:>7.1f}% "
                  f"{avg_vol:>8.2f}  {avg_edge:>10.3f}")

    # ── Daily pattern (day of week) ──
    strong["dow"] = strong["close_time"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = strong.groupby("dow").agg(
        count=("won", "count"),
        win_rate=("won", "mean"),
    ).reindex(dow_order).dropna()
    print(f"\n  Win rate by day of week:")
    for dow, row in dow_stats.iterrows():
        bar = "#" * int(row["win_rate"] * 20)
        print(f"  {dow:<12} {row['count']:>6,}  {row['win_rate']*100:>5.1f}%  {bar}")


def run(series: str) -> None:
    print(f"\n{'#'*65}")
    print(f"  Items 14-16: {series}")
    print(f"{'#'*65}")
    df = load_analysis(series)
    print(f"  Loaded {len(df):,} enriched markets.")
    print_naive_baseline(df, series)
    print_no_side_analysis(df, series)
    print_time_and_regime(df, series)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=str, default=None, help="KXBTCD or KXBTC15M (default: both)")
    args = parser.parse_args()
    series_list = [args.series] if args.series else ["KXBTCD", "KXBTC15M"]
    for s in series_list:
        try:
            run(s)
        except FileNotFoundError as e:
            print(f"\n  Skipping {s}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
