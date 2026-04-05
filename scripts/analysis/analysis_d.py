from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_btc_bot.models.gbm_threshold import terminal_probability_above
from scripts.analysis.cleaning import (
    add_market_derived_columns,
    assign_series_buckets,
    build_global_spot_features,
    deduplicate_snapshots,
)
from scripts.analysis.loaders import AnalysisContext, load_settlements, load_snapshots


def run_analysis_d(context: AnalysisContext) -> dict[str, Path]:
    frame = _prepare_frame(context, "KXBTCD")
    strategies = {
        "baseline_best_combo": _select_baseline_entries(frame),
        "near_expiry_only": _select_near_expiry_entries(frame),
        "momentum_agreement": _select_momentum_entries(frame),
        "market_convergence": _select_convergence_entries(frame),
    }

    rows: list[dict[str, object]] = []
    for strategy_name, entries in strategies.items():
        trades = _simulate_hold_or_tp(frame, entries, use_take_profit=(strategy_name == "baseline_best_combo"))
        rows.append(
            {
                "strategy": strategy_name,
                "trade_count": int(len(trades)),
                "win_rate": round(float((trades["realized_pnl_cents"] > 0).mean() * 100.0), 2) if not trades.empty else 0.0,
                "total_pnl_cents": round(float(trades["realized_pnl_cents"].sum()), 4) if not trades.empty else 0.0,
                "avg_pnl_cents": round(float(trades["realized_pnl_cents"].mean()), 4) if not trades.empty else 0.0,
                "avg_hold_minutes": round(float(trades["hold_minutes"].mean()), 4) if not trades.empty else 0.0,
                "yes_trade_count": int((trades["side"] == "yes").sum()) if not trades.empty else 0,
                "no_trade_count": int((trades["side"] == "no").sum()) if not trades.empty else 0,
            }
        )

    result = pd.DataFrame(rows)
    out_path = context.output_dir / "analysis_d_kxbtcd_strategy_comparison.csv"
    result.to_csv(out_path, index=False)
    return {"comparison": out_path}


def _prepare_frame(context: AnalysisContext, series: str) -> pd.DataFrame:
    snapshots = load_snapshots(
        context.db_path,
        series=series,
        columns=[
            "market_ticker",
            "series_ticker",
            "observed_at",
            "expiry",
            "spot_price",
            "threshold",
            "yes_bid",
            "yes_ask",
            "no_bid",
            "no_ask",
            "mid_price",
            "volume",
            "open_interest",
            "source",
            "metadata_json",
        ],
    )
    deduped, _ = deduplicate_snapshots(snapshots)
    deduped, _ = build_global_spot_features(deduped)
    deduped = assign_series_buckets(add_market_derived_columns(deduped))

    settlements = load_settlements(series)[["ticker", "result"]].copy()
    settlements["result"] = settlements["result"].str.lower()
    frame = deduped.merge(settlements, left_on="market_ticker", right_on="ticker", how="inner")

    drift = float(context.settings.model.get("drift", 0.0))
    vol_floor = float(context.settings.model.get("volatility_floor", 0.05))
    frame["vol_for_model"] = pd.to_numeric(frame["rolling_vol_20"], errors="coerce").fillna(vol_floor)
    frame["model_probability_yes"] = frame.apply(
        lambda row: float(
            terminal_probability_above(
                spot_price=float(row["spot_price"]),
                target_price=float(row["threshold"]),
                time_to_expiry_years=max(float(row["tte_min"]), 0.0) / (365.0 * 24.0 * 60.0),
                volatility=max(float(row["vol_for_model"]), vol_floor),
                drift=drift,
            )
        ),
        axis=1,
    )
    frame["yes_edge"] = frame["model_probability_yes"] - frame["yes_ask"]
    frame["no_edge"] = (1.0 - frame["model_probability_yes"]) - frame["no_ask"]
    frame["yes_spread_cents"] = (frame["yes_ask"] - frame["yes_bid"]) * 100.0
    frame["no_spread_cents"] = (frame["no_ask"] - frame["no_bid"]) * 100.0
    frame = frame.sort_values(["market_ticker", "observed_at"]).reset_index(drop=True)
    frame["market_row_number"] = frame.groupby("market_ticker").cumcount()
    return frame


def _pick_first(entries: pd.DataFrame, edge_col: str) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()
    ordered = entries.sort_values(["market_ticker", "observed_at", edge_col], ascending=[True, True, False])
    return ordered.drop_duplicates(subset=["market_ticker"], keep="first").reset_index(drop=True)


def _select_baseline_entries(frame: pd.DataFrame) -> pd.DataFrame:
    entries = frame[
        (frame["tte_min"] >= 30.0)
        & (frame["tte_min"] <= 45.0)
        & (frame["dist_bps"] < 200.0)
        & (frame["no_edge"] >= 0.05)
        & (frame["no_ask"] * 100.0 >= 40.0)
        & (frame["no_ask"] * 100.0 <= 60.0)
        & (frame["no_spread_cents"] <= 4.0)
    ].copy()
    entries["side"] = "no"
    entries["entry_price_cents"] = np.round(entries["no_ask"] * 100.0)
    entries["entry_edge"] = entries["no_edge"]
    return _pick_first(entries, "no_edge")


def _select_near_expiry_entries(frame: pd.DataFrame) -> pd.DataFrame:
    entries = frame[
        (frame["tte_min"] > 0.0)
        & (frame["tte_min"] <= 5.0)
        & (frame["dist_bps"] < 100.0)
    ].copy()
    entries["side"] = np.where(entries["spot_price"] >= entries["threshold"], "yes", "no")
    entries["entry_price_cents"] = np.where(
        entries["side"] == "yes",
        np.round(entries["yes_ask"] * 100.0),
        np.round(entries["no_ask"] * 100.0),
    )
    entries["entry_edge"] = np.where(entries["side"] == "yes", entries["yes_edge"], entries["no_edge"])
    entries = entries[(entries["entry_price_cents"] >= 1.0) & (entries["entry_price_cents"] <= 99.0)]
    return _pick_first(entries, "entry_edge")


def _select_momentum_entries(frame: pd.DataFrame) -> pd.DataFrame:
    entries = frame[
        (frame["tte_min"] >= 30.0)
        & (frame["tte_min"] <= 45.0)
        & (frame["dist_bps"] < 200.0)
        & (frame["trend_regime"].isin(["up", "down"]))
    ].copy()
    entries["side"] = np.where(entries["trend_regime"] == "up", "yes", "no")
    obvious_side = np.where(entries["spot_price"] >= entries["threshold"], "yes", "no")
    entries = entries[entries["side"] == obvious_side].copy()
    entries["entry_price_cents"] = np.where(
        entries["side"] == "yes",
        np.round(entries["yes_ask"] * 100.0),
        np.round(entries["no_ask"] * 100.0),
    )
    entries["entry_edge"] = np.where(entries["side"] == "yes", entries["yes_edge"], entries["no_edge"])
    entries = entries[(entries["entry_price_cents"] >= 40.0) & (entries["entry_price_cents"] <= 60.0)]
    return _pick_first(entries, "entry_edge")


def _select_convergence_entries(frame: pd.DataFrame) -> pd.DataFrame:
    entries = frame[
        (frame["tte_min"] > 0.0)
        & (frame["tte_min"] < 10.0)
        & (frame["dist_bps"] < 100.0)
    ].copy()
    obvious_side = np.where(entries["spot_price"] >= entries["threshold"], "yes", "no")
    entries["side"] = obvious_side
    entries["entry_price_cents"] = np.where(
        entries["side"] == "yes",
        np.round(entries["yes_ask"] * 100.0),
        np.round(entries["no_ask"] * 100.0),
    )
    entries["entry_edge"] = np.where(entries["side"] == "yes", entries["yes_edge"], entries["no_edge"])
    entries = entries[entries["entry_price_cents"] < 80.0]
    return _pick_first(entries, "entry_edge")


def _simulate_hold_or_tp(frame: pd.DataFrame, entries: pd.DataFrame, *, use_take_profit: bool) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame(columns=["market_ticker", "side", "realized_pnl_cents", "hold_minutes"])
    market_groups = {ticker: group.reset_index(drop=True) for ticker, group in frame.groupby("market_ticker", sort=False)}
    rows: list[dict[str, object]] = []
    for entry in entries.itertuples(index=False):
        market = market_groups[str(entry.market_ticker)]
        entry_idx = int(entry.market_row_number)
        future = market.iloc[entry_idx + 1 :].copy()
        if use_take_profit:
            bid_col = "yes_bid" if entry.side == "yes" else "no_bid"
            tp_hits = future[np.round(future[bid_col] * 100.0) - float(entry.entry_price_cents) >= 8.0]
            if not tp_hits.empty:
                exit_row = tp_hits.iloc[0]
                exit_price_cents = float(np.round(exit_row[bid_col] * 100.0))
                exit_time = exit_row["observed_at"]
            else:
                final_row = market.iloc[-1]
                result = str(final_row["result"]).lower()
                exit_price_cents = 100.0 if result == entry.side else 0.0
                exit_time = final_row["observed_at"]
        else:
            final_row = market.iloc[-1]
            result = str(final_row["result"]).lower()
            exit_price_cents = 100.0 if result == entry.side else 0.0
            exit_time = final_row["observed_at"]

        rows.append(
            {
                "market_ticker": entry.market_ticker,
                "side": entry.side,
                "realized_pnl_cents": exit_price_cents - float(entry.entry_price_cents),
                "hold_minutes": max((pd.Timestamp(exit_time) - pd.Timestamp(entry.observed_at)).total_seconds() / 60.0, 0.0),
            }
        )
    return pd.DataFrame(rows)
