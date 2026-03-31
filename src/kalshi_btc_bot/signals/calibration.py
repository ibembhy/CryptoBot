from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pandas as pd

from kalshi_btc_bot.types import ProbabilityEstimate
from kalshi_btc_bot.utils.math import clamp

if TYPE_CHECKING:
    from kalshi_btc_bot.backtest.engine import BacktestEngine


@dataclass(frozen=True)
class ProbabilityCalibrator:
    points: tuple[tuple[float, float], ...]
    sample_count: int
    bucket_width: float

    def apply(self, probability: float) -> float:
        x = clamp(float(probability), 0.0, 1.0)
        if not self.points:
            return x
        if x <= self.points[0][0]:
            return self.points[0][1]
        if x >= self.points[-1][0]:
            return self.points[-1][1]
        for idx in range(1, len(self.points)):
            left_x, left_y = self.points[idx - 1]
            right_x, right_y = self.points[idx]
            if x <= right_x:
                if right_x == left_x:
                    return right_y
                weight = (x - left_x) / (right_x - left_x)
                return clamp(left_y + weight * (right_y - left_y), 0.0, 1.0)
        return x

    def apply_to_estimate(self, estimate: ProbabilityEstimate) -> ProbabilityEstimate:
        calibrated = self.apply(estimate.probability)
        return replace(
            estimate,
            probability=calibrated,
            raw_probability=estimate.raw_probability if estimate.raw_probability is not None else estimate.probability,
        )


def fit_probability_calibrator(
    trades: pd.DataFrame,
    *,
    probability_column: str = "raw_model_probability",
    outcome_column: str = "contract_won",
    bucket_width: float = 0.05,
    min_samples: int = 50,
    min_bucket_count: int = 3,
) -> ProbabilityCalibrator | None:
    if trades.empty or probability_column not in trades or outcome_column not in trades:
        return None
    frame = trades[[probability_column, outcome_column]].dropna().copy()
    if len(frame) < min_samples:
        return None

    bins = [round(i * bucket_width, 10) for i in range(int(1.0 / bucket_width) + 1)]
    if bins[-1] < 1.0:
        bins.append(1.0)
    frame["bucket"] = pd.cut(
        frame[probability_column].astype(float),
        bins=bins,
        include_lowest=True,
        right=False,
    )

    points: list[tuple[float, float]] = [(0.0, 0.0)]
    for _, group in frame.groupby("bucket", observed=False):
        if len(group) < min_bucket_count:
            continue
        predicted = float(group[probability_column].mean())
        actual = float(group[outcome_column].mean())
        points.append((predicted, actual))
    points.append((1.0, 1.0))
    points = sorted(points, key=lambda item: item[0])

    monotonic_points: list[tuple[float, float]] = []
    running_y = 0.0
    for x, y in points:
        running_y = max(running_y, clamp(y, 0.0, 1.0))
        monotonic_points.append((clamp(x, 0.0, 1.0), running_y))

    deduped: list[tuple[float, float]] = []
    for x, y in monotonic_points:
        if deduped and abs(deduped[-1][0] - x) < 1e-9:
            deduped[-1] = (x, max(deduped[-1][1], y))
        else:
            deduped.append((x, y))
    return ProbabilityCalibrator(points=tuple(deduped), sample_count=int(len(frame)), bucket_width=bucket_width)


def build_engine_calibrators(
    engine: "BacktestEngine",
    *,
    snapshots,
    feature_frame: pd.DataFrame,
    bucket_width: float = 0.05,
    min_samples: int = 50,
    min_bucket_count: int = 3,
) -> dict[str, ProbabilityCalibrator]:
    calibrators: dict[str, ProbabilityCalibrator] = {}
    for model_name in engine.models:
        cloned = deepcopy(engine)
        cloned.model = cloned.models[model_name]
        cloned.models = {model_name: cloned.models[model_name]}
        cloned.fusion_config = None
        trades = cloned.run_strategy("hold_to_settlement", snapshots, feature_frame).trades
        calibrator = fit_probability_calibrator(
            trades,
            probability_column="raw_model_probability",
            outcome_column="contract_won",
            bucket_width=bucket_width,
            min_samples=min_samples,
            min_bucket_count=min_bucket_count,
        )
        if calibrator is not None:
            calibrators[model_name] = calibrator
    return calibrators
