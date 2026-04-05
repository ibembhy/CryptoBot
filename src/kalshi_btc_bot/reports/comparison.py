from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import product

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.backtest.replay import build_replay_dataset
from kalshi_btc_bot.reports.summary import build_summary_report


def _bucket_price_band(trades: pd.DataFrame) -> pd.Series:
    return pd.cut(
        trades["entry_price_cents"],
        bins=[0, 10, 25, 40, 60, 75, 90, 100],
        include_lowest=True,
        right=False,
    ).astype(str)


def _bucket_hold_minutes(trades: pd.DataFrame) -> pd.Series:
    return pd.cut(
        trades["hold_minutes"],
        bins=[0, 5, 15, 30, 60, 180, 10_000],
        include_lowest=True,
        right=False,
    ).astype(str)


def _bucket_edge(trades: pd.DataFrame) -> pd.Series:
    return pd.cut(
        trades["edge"],
        bins=[-1, 0, 0.05, 0.1, 0.2, 0.4, 1.0],
        include_lowest=True,
        right=False,
    ).astype(str)


def build_bucket_breakdown(trades: pd.DataFrame) -> dict[str, list[dict]]:
    if trades.empty:
        return {}

    frame = trades.copy()
    frame["price_band"] = _bucket_price_band(frame)
    frame["hold_band"] = _bucket_hold_minutes(frame)
    frame["edge_band"] = _bucket_edge(frame)

    breakdown: dict[str, list[dict]] = {}
    for column in ("contract_type", "side", "exit_trigger", "price_band", "hold_band", "edge_band"):
        rows = []
        for key, group in frame.groupby(column):
            if group.empty:
                continue
            rows.append(
                {
                    column: key,
                    "trade_count": int(len(group)),
                    "pnl": round(float(group["realized_pnl"].sum()), 2),
                    "roi": round(
                        float(group["realized_pnl"].sum()) / max(float(group["entry_notional"].sum()), 1e-9) * 100.0,
                        2,
                    ),
                    "win_rate": round(float((group["realized_pnl"] > 0).mean() * 100.0), 2),
                    "avg_edge": round(float(group["edge"].mean()), 4),
                }
            )
        breakdown[column] = rows
    return breakdown


def build_failure_analysis(
    model_report: dict,
    *,
    strategy_mode: str = "early_exit",
    top_n: int = 3,
) -> dict:
    summary = dict(model_report.get(strategy_mode, {}))
    buckets = dict(model_report.get("bucket_breakdown", {}).get(strategy_mode, {}))
    best_buckets: dict[str, list[dict]] = {}
    worst_buckets: dict[str, list[dict]] = {}
    for bucket_name, rows in buckets.items():
        ordered = sorted(rows, key=lambda row: (float(row["pnl"]), float(row["roi"]), int(row["trade_count"])))
        worst_buckets[bucket_name] = ordered[:top_n]
        best_buckets[bucket_name] = list(reversed(ordered[-top_n:]))

    primary_leaks = []
    strongest_pockets = []
    for bucket_name in ("exit_trigger", "price_band", "edge_band", "side", "hold_band"):
        for row in worst_buckets.get(bucket_name, []):
            if float(row["pnl"]) < 0:
                primary_leaks.append({"bucket": bucket_name, **row})
        for row in best_buckets.get(bucket_name, []):
            if float(row["pnl"]) > 0:
                strongest_pockets.append({"bucket": bucket_name, **row})

    primary_leaks = sorted(
        primary_leaks,
        key=lambda row: (float(row["pnl"]), -float(row["roi"]), -int(row["trade_count"])),
    )[:top_n]
    strongest_pockets = sorted(
        strongest_pockets,
        key=lambda row: (float(row["pnl"]), float(row["roi"]), int(row["trade_count"])),
        reverse=True,
    )[:top_n]

    return {
        "summary": summary,
        "best_buckets": best_buckets,
        "worst_buckets": worst_buckets,
        "primary_leaks": primary_leaks,
        "strongest_pockets": strongest_pockets,
    }


def filter_snapshots_for_focus(
    snapshots,
    *,
    near_money_bps: float | None = None,
    max_minutes_to_expiry: float | None = None,
    min_price_cents: int | None = None,
    max_price_cents: int | None = None,
):
    filtered = []
    for snapshot in snapshots:
        if snapshot.threshold is None:
            continue
        if near_money_bps is not None and snapshot.spot_price > 0:
            distance_bps = abs(snapshot.threshold - snapshot.spot_price) / snapshot.spot_price * 10_000.0
            if distance_bps > near_money_bps:
                continue
        minutes_to_expiry = (snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0
        if max_minutes_to_expiry is not None and minutes_to_expiry > max_minutes_to_expiry:
            continue
        if snapshot.yes_ask is None:
            continue
        price_cents = int(round(snapshot.yes_ask * 100.0))
        if min_price_cents is not None and price_cents < min_price_cents:
            continue
        if max_price_cents is not None and price_cents > max_price_cents:
            continue
        filtered.append(snapshot)
    return filtered


def build_model_comparison_report(
    engines: dict[str, BacktestEngine],
    *,
    snapshots,
    feature_frame: pd.DataFrame,
) -> dict:
    report: dict[str, dict] = {"models": {}}
    for name, engine in engines.items():
        comparison = engine.compare_strategies(snapshots, feature_frame)
        hold = comparison["hold_to_settlement"]
        early = comparison["early_exit"]
        report["models"][name] = {
            "summary": comparison["summary"],
            "hold_to_settlement": build_summary_report(hold),
            "early_exit": build_summary_report(early),
            "bucket_breakdown": {
                "hold_to_settlement": build_bucket_breakdown(hold.trades),
                "early_exit": build_bucket_breakdown(early.trades),
            },
        }
    return report


def build_grid_search_report(
    engines: dict[str, BacktestEngine],
    *,
    snapshots,
    volatility_window: int,
    annualization_factor: float,
    near_money_bps_values: list[float],
    max_minutes_to_expiry_values: list[float],
    min_price_cents_values: list[int],
    max_price_cents_values: list[int],
    top_n: int = 20,
) -> dict:
    rows: list[dict] = []
    for near_money_bps, max_minutes_to_expiry, min_price_cents, max_price_cents in product(
        near_money_bps_values,
        max_minutes_to_expiry_values,
        min_price_cents_values,
        max_price_cents_values,
    ):
        if min_price_cents > max_price_cents:
            continue
        focused_snapshots = filter_snapshots_for_focus(
            snapshots,
            near_money_bps=near_money_bps,
            max_minutes_to_expiry=max_minutes_to_expiry,
            min_price_cents=min_price_cents,
            max_price_cents=max_price_cents,
        )
        dataset = build_replay_dataset(
            focused_snapshots,
            volatility_window=volatility_window,
            annualization_factor=annualization_factor,
        )
        if not dataset.snapshots:
            continue
        report = build_model_comparison_report(engines, snapshots=dataset.snapshots, feature_frame=dataset.feature_frame)
        for model_name, model_report in report["models"].items():
            hold = model_report["hold_to_settlement"]
            early = model_report["early_exit"]
            best_mode = "early_exit" if early["pnl"] >= hold["pnl"] else "hold_to_settlement"
            best = early if best_mode == "early_exit" else hold
            rows.append(
                {
                    "model": model_name,
                    "near_money_bps": near_money_bps,
                    "max_minutes_to_expiry": max_minutes_to_expiry,
                    "min_price_cents": min_price_cents,
                    "max_price_cents": max_price_cents,
                    "snapshot_count": len(dataset.snapshots),
                    "feature_rows": len(dataset.feature_frame),
                    "hold_pnl": hold["pnl"],
                    "hold_roi": hold["roi"],
                    "hold_trade_count": hold["trade_count"],
                    "early_exit_pnl": early["pnl"],
                    "early_exit_roi": early["roi"],
                    "early_exit_trade_count": early["trade_count"],
                    "best_mode": best_mode,
                    "best_pnl": best["pnl"],
                    "best_roi": best["roi"],
                    "best_trade_count": best["trade_count"],
                }
            )

    ranking = sorted(
        rows,
        key=lambda row: (
            float(row["best_pnl"]),
            float(row["best_roi"]),
            int(row["best_trade_count"]),
            int(row["snapshot_count"]),
        ),
        reverse=True,
    )
    return {
        "grid_size": len(rows),
        "top_results": ranking[:top_n],
        "all_results": ranking,
    }


def clone_engine_with_mode(engine: BacktestEngine, *, fusion_mode: str, primary_model: str) -> BacktestEngine:
    cloned = deepcopy(engine)
    if cloned.fusion_config is not None:
        cloned.fusion_config = replace(cloned.fusion_config, mode=fusion_mode, primary_model=primary_model)
    cloned.model = cloned.models[primary_model]
    return cloned
