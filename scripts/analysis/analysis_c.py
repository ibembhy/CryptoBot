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
from scripts.analysis.loaders import AnalysisContext, load_live_ledger, load_settlements, load_snapshots


def run_analysis_c(context: AnalysisContext) -> dict[str, Path]:
    frame = _prepare_frame(context, "KXBTCD")
    baseline_entries = _select_best_kxbtcd_entries(frame)
    reentry_penalty = _estimate_reentry_penalty_cents()

    scenarios = [
        ("base_mid_coinbase", "mid", 0, False, "coinbase"),
        ("after_ask_entry", "ask", 0, False, "coinbase"),
        ("after_delay", "ask", 1, False, "coinbase"),
        ("after_reentry_cost", "ask", 1, True, "coinbase"),
        ("after_brti_settlement", "ask", 1, True, "brti"),
    ]

    rows: list[dict[str, object]] = []
    previous_pnl: float | None = None
    base_pnl: float | None = None
    for name, price_source, delay_snapshots, apply_reentry_cost, settlement_basis in scenarios:
        trades = _simulate_degradation_scenario(
            frame,
            baseline_entries,
            entry_price_source=price_source,
            delay_snapshots=delay_snapshots,
            apply_reentry_cost=apply_reentry_cost,
            reentry_penalty_cents=reentry_penalty,
            settlement_basis=settlement_basis,
        )
        total_pnl = float(trades["realized_pnl_cents"].sum()) if not trades.empty else 0.0
        if base_pnl is None:
            base_pnl = total_pnl
        rows.append(
            {
                "scenario": name,
                "entry_price_source": price_source,
                "delay_snapshots": delay_snapshots,
                "reentry_cost_applied": apply_reentry_cost,
                "settlement_basis": settlement_basis,
                "trade_count": int(len(trades)),
                "win_rate": round(float((trades["realized_pnl_cents"] > 0).mean() * 100.0), 2) if not trades.empty else 0.0,
                "total_pnl_cents": round(total_pnl, 4),
                "avg_pnl_cents": round(float(trades["realized_pnl_cents"].mean()), 4) if not trades.empty else 0.0,
                "delta_vs_previous_cents": round(total_pnl - previous_pnl, 4) if previous_pnl is not None else 0.0,
                "delta_vs_base_cents": round(total_pnl - base_pnl, 4),
                "avg_entry_edge": round(float(trades["entry_edge"].mean()), 6) if not trades.empty else 0.0,
                "estimated_reentry_penalty_cents_per_trade": round(reentry_penalty, 4),
            }
        )
        previous_pnl = total_pnl

    result = pd.DataFrame(rows)
    out_path = context.output_dir / "analysis_c_kxbtcd_waterfall.csv"
    result.to_csv(out_path, index=False)
    return {"waterfall": out_path}


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

    settlements = load_settlements(series)[["ticker", "result", "expiration_value"]].copy()
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
    frame["coinbase_result"] = np.where(
        pd.to_numeric(frame["spot_price"], errors="coerce")
        >= pd.to_numeric(frame["threshold"], errors="coerce"),
        "yes",
        "no",
    )
    return frame


def _select_best_kxbtcd_entries(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[
        (frame["tte_min"] >= 30.0)
        & (frame["tte_min"] <= 45.0)
        & (frame["dist_bps"] < 200.0)
        & (frame["no_edge"] >= 0.05)
        & (frame["no_ask"] * 100.0 >= 40.0)
        & (frame["no_ask"] * 100.0 <= 60.0)
        & ((frame["no_ask"] - frame["no_bid"]) * 100.0 <= 4.0)
    ].copy()
    ordered = eligible.sort_values(["market_ticker", "observed_at", "no_edge"], ascending=[True, True, False])
    firsts = ordered.drop_duplicates(subset=["market_ticker"], keep="first").copy()
    firsts["entry_price_ask_cents"] = np.round(firsts["no_ask"] * 100.0)
    firsts["entry_price_mid_cents"] = np.round(((firsts["no_ask"] + firsts["no_bid"]) / 2.0) * 100.0)
    firsts["entry_spread_cents"] = (firsts["no_ask"] - firsts["no_bid"]) * 100.0
    return firsts.reset_index(drop=True)


def _estimate_reentry_penalty_cents() -> float:
    ledger = load_live_ledger("KXBTCD")
    if ledger.empty:
        return 0.0
    buys = ledger[
        (ledger["request_action"].astype(str).str.lower() == "buy")
        & ledger["request_ticker"].notna()
    ].copy()
    if buys.empty:
        return 0.0
    counts = buys.groupby("request_ticker").size()
    extra_entries_per_market = float(np.maximum(counts - 1, 0).mean()) if len(counts) else 0.0
    return extra_entries_per_market * 4.0


def _simulate_degradation_scenario(
    frame: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    entry_price_source: str,
    delay_snapshots: int,
    apply_reentry_cost: bool,
    reentry_penalty_cents: float,
    settlement_basis: str,
) -> pd.DataFrame:
    market_groups = {ticker: group.reset_index(drop=True) for ticker, group in frame.groupby("market_ticker", sort=False)}
    rows: list[dict[str, object]] = []
    for entry in entries.itertuples(index=False):
        market = market_groups[str(entry.market_ticker)]
        signal_idx = int(entry.market_row_number)
        fill_idx = signal_idx + delay_snapshots
        if fill_idx >= len(market):
            continue

        fill_row = market.iloc[fill_idx]
        entry_price_cents = float(
            np.round(((float(fill_row["no_ask"]) + float(fill_row["no_bid"])) / 2.0) * 100.0)
            if entry_price_source == "mid"
            else np.round(fill_row["no_ask"] * 100.0)
        )
        if np.isnan(entry_price_cents):
            continue

        future = market.iloc[fill_idx + 1 :].copy()
        tp_hits = future[np.round(future["no_bid"] * 100.0) - entry_price_cents >= 8.0]
        if not tp_hits.empty:
            exit_row = tp_hits.iloc[0]
            exit_price_cents = float(np.round(exit_row["no_bid"] * 100.0))
            exit_time = exit_row["observed_at"]
            exit_trigger = "take_profit"
        else:
            final_row = market.iloc[-1]
            result = str(final_row["result"] if settlement_basis == "brti" else final_row["coinbase_result"]).lower()
            exit_price_cents = 100.0 if result == "no" else 0.0
            exit_time = final_row["observed_at"]
            exit_trigger = "settlement"

        realized_pnl = exit_price_cents - entry_price_cents
        if apply_reentry_cost:
            realized_pnl -= reentry_penalty_cents

        rows.append(
            {
                "market_ticker": entry.market_ticker,
                "entry_time": fill_row["observed_at"],
                "entry_edge": float(fill_row["no_edge"]),
                "entry_price_cents": entry_price_cents,
                "exit_time": exit_time,
                "exit_price_cents": exit_price_cents,
                "exit_trigger": exit_trigger,
                "realized_pnl_cents": realized_pnl,
            }
        )
    return pd.DataFrame(rows)
