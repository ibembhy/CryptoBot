from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.cleaning import (
    SERIES_NEAR_MONEY_BPS,
    add_market_derived_columns,
    assign_series_buckets,
    build_global_spot_features,
    deduplicate_snapshots,
)
from scripts.analysis.loaders import AnalysisContext, load_settlements, load_snapshots


def run_analysis_a(context: AnalysisContext, series_list: list[str]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    summary_lines = ["Analysis A - Does the model have real edge?", ""]

    for series in series_list:
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
                "implied_probability",
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
        settlements["actual_yes"] = (settlements["result"].str.lower() == "yes").astype(int)

        frame = deduped.merge(settlements, left_on="market_ticker", right_on="ticker", how="inner")
        frame["settlement_basis_bps"] = ((frame["expiration_value"] - frame["spot_price"]) / frame["spot_price"]) * 10000.0
        frame["has_yes_book"] = frame["yes_bid"].notna() & frame["yes_ask"].notna()
        frame["has_no_book"] = frame["no_bid"].notna() & frame["no_ask"].notna()

        filtered = frame[
            frame["tte_bucket"].notna()
            & frame["distance_bucket_bps"].notna()
            & (frame["dist_bps"] < SERIES_NEAR_MONEY_BPS[series])
            & frame["yes_ask"].notna()
            & frame["yes_bid"].notna()
        ].copy()

        base_prefix = series.lower()
        dimension_specs = {
            "tte": ["tte_bucket"],
            "distance": ["distance_bucket_bps"],
            "hour_utc": ["hour_utc"],
            "market_side": ["market_favored_side"],
            "trend": ["trend_regime"],
            "vol": ["vol_regime"],
            "volume": ["volume_bucket"],
        }

        for label, group_cols in dimension_specs.items():
            grouped = _summarize_yes_edge(filtered, group_cols)
            out_path = context.output_dir / f"{base_prefix}_analysis_a_{label}.csv"
            grouped.to_csv(out_path, index=False)
            outputs[f"{series}_{label}"] = out_path

        heatmap = _summarize_yes_edge(filtered, ["tte_bucket", "distance_bucket_bps"])
        heatmap_path = context.output_dir / f"{base_prefix}_analysis_a_tte_distance_heatmap.csv"
        heatmap.to_csv(heatmap_path, index=False)
        outputs[f"{series}_heatmap"] = heatmap_path

        basis = _basis_distribution(frame)
        basis_path = context.output_dir / f"{base_prefix}_analysis_a_brti_basis.csv"
        basis.to_csv(basis_path, index=False)
        outputs[f"{series}_basis"] = basis_path

        summary_lines.extend(
            [
                f"[{series}]",
                f"rows_after_filter={len(filtered):,}",
                f"avg_yes_ask={filtered['yes_ask'].mean():.4f}" if not filtered.empty else "avg_yes_ask=nan",
                f"actual_yes_rate={filtered['actual_yes'].mean():.4f}" if not filtered.empty else "actual_yes_rate=nan",
                f"avg_yes_edge_actual={(filtered['actual_yes'] - filtered['yes_ask']).mean():.4f}" if not filtered.empty else "avg_yes_edge_actual=nan",
                "",
            ]
        )

    summary_path = context.output_dir / "analysis_a_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs


def _summarize_yes_edge(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            sample_count=("market_ticker", "size"),
            actual_yes_win_rate=("actual_yes", "mean"),
            avg_yes_ask=("yes_ask", "mean"),
            avg_yes_bid=("yes_bid", "mean"),
            avg_spread_cents_yes=("spread_cents_yes", "mean"),
            avg_volume=("volume", "mean"),
        )
        .reset_index()
    )
    grouped["yes_edge_actual"] = grouped["actual_yes_win_rate"] - grouped["avg_yes_ask"]
    grouped["market_minus_actual"] = grouped["avg_yes_ask"] - grouped["actual_yes_win_rate"]
    return grouped.sort_values(group_cols).reset_index(drop=True)


def _basis_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.dropna(subset=["settlement_basis_bps"]).copy()
    thresholds = [50, 100, 200]
    rows = []
    total = len(valid)
    for threshold in thresholds:
        rows.append(
            {
                "basis_threshold_bps": threshold,
                "share_abs_gt_threshold": float((valid["settlement_basis_bps"].abs() > threshold).mean()) if total else np.nan,
                "share_pos_gt_threshold": float((valid["settlement_basis_bps"] > threshold).mean()) if total else np.nan,
                "share_neg_lt_threshold": float((valid["settlement_basis_bps"] < -threshold).mean()) if total else np.nan,
                "sample_count": total,
                "mean_basis_bps": float(valid["settlement_basis_bps"].mean()) if total else np.nan,
                "median_basis_bps": float(valid["settlement_basis_bps"].median()) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)
