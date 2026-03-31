from __future__ import annotations

import math

import pandas as pd


def _safe_log_loss(probability: float, outcome: float) -> float:
    p = min(max(probability, 1e-9), 1 - 1e-9)
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))


def build_calibration_report(
    trades: pd.DataFrame,
    *,
    probability_column: str = "model_probability",
    outcome_column: str = "contract_won",
    bucket_width: float = 0.05,
) -> dict:
    if trades.empty or probability_column not in trades or outcome_column not in trades:
        return {"overall": {}, "buckets": [], "slices": {}}

    frame = trades[[probability_column, outcome_column]].copy()
    frame = frame.dropna()
    if frame.empty:
        return {"overall": {}, "buckets": [], "slices": {}}

    probs = frame[probability_column].astype(float)
    outcomes = frame[outcome_column].astype(float)
    brier = float(((probs - outcomes) ** 2).mean())
    log_loss = float(sum(_safe_log_loss(p, y) for p, y in zip(probs, outcomes, strict=False)) / len(frame))

    bins = [round(i * bucket_width, 10) for i in range(int(1.0 / bucket_width) + 1)]
    if bins[-1] < 1.0:
        bins.append(1.0)
    frame["bucket"] = pd.cut(
        probs,
        bins=bins,
        include_lowest=True,
        right=False,
    )
    bucket_rows = []
    for bucket, group in frame.groupby("bucket", observed=False):
        if group.empty:
            continue
        bucket_rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(group)),
                "predicted_probability": round(float(group[probability_column].mean()), 4),
                "actual_win_rate": round(float(group[outcome_column].mean()), 4),
                "calibration_gap": round(float(group[probability_column].mean() - group[outcome_column].mean()), 4),
            }
        )

    slices: dict[str, list[dict]] = {}
    for column in ("contract_type", "side", "exit_trigger"):
        if column in trades:
            rows = []
            for key, group in trades.groupby(column):
                if group.empty or probability_column not in group or outcome_column not in group:
                    continue
                rows.append(
                    {
                        column: key,
                        "count": int(len(group)),
                        "predicted_probability": round(float(group[probability_column].mean()), 4),
                        "actual_win_rate": round(float(group[outcome_column].mean()), 4),
                    }
                )
            slices[column] = rows

    return {
        "overall": {
            "count": int(len(frame)),
            "brier_score": round(brier, 6),
            "log_loss": round(log_loss, 6),
            "mean_predicted_probability": round(float(probs.mean()), 4),
            "mean_actual_win_rate": round(float(outcomes.mean()), 4),
        },
        "buckets": bucket_rows,
        "slices": slices,
    }
