from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate
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


@dataclass(frozen=True)
class RegimeAwareProbabilityCalibrator:
    global_calibrator: ProbabilityCalibrator
    regime_calibrators: dict[str, ProbabilityCalibrator]
    sample_count: int

    def apply(self, probability: float, *, regime_key: str | None = None) -> float:
        calibrator = self.regime_calibrators.get(regime_key or "", self.global_calibrator)
        return calibrator.apply(probability)

    def apply_to_estimate(self, estimate: ProbabilityEstimate, *, regime_key: str | None = None) -> ProbabilityEstimate:
        calibrated = self.apply(estimate.probability, regime_key=regime_key)
        inputs = dict(estimate.inputs)
        if regime_key is not None:
            inputs["calibration_regime"] = regime_key
        return replace(
            estimate,
            probability=calibrated,
            inputs=inputs,
            raw_probability=estimate.raw_probability if estimate.raw_probability is not None else estimate.probability,
        )


def _serialize_probability_calibrator(calibrator: ProbabilityCalibrator) -> dict[str, object]:
    return {
        "points": [[float(x), float(y)] for x, y in calibrator.points],
        "sample_count": calibrator.sample_count,
        "bucket_width": calibrator.bucket_width,
    }


def _deserialize_probability_calibrator(payload: dict[str, object]) -> ProbabilityCalibrator | None:
    raw_points = payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        return None
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, list | tuple) or len(item) != 2:
            return None
        points.append((float(item[0]), float(item[1])))
    return ProbabilityCalibrator(
        points=tuple(points),
        sample_count=int(payload.get("sample_count", 0)),
        bucket_width=float(payload.get("bucket_width", 0.05)),
    )


def save_engine_calibrators(
    path: str | Path,
    calibrators: dict[str, RegimeAwareProbabilityCalibrator],
    *,
    metadata: dict[str, object],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "calibrators": {
            model_name: {
                "global": _serialize_probability_calibrator(calibrator.global_calibrator),
                "regimes": {
                    regime_key: _serialize_probability_calibrator(regime_calibrator)
                    for regime_key, regime_calibrator in calibrator.regime_calibrators.items()
                },
                "sample_count": calibrator.sample_count,
            }
            for model_name, calibrator in calibrators.items()
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_engine_calibrators(
    path: str | Path,
    *,
    expected_metadata: dict[str, object],
) -> dict[str, RegimeAwareProbabilityCalibrator] | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = payload.get("metadata", {})
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        return None
    calibrators: dict[str, RegimeAwareProbabilityCalibrator] = {}
    for model_name, raw_calibrator in dict(payload.get("calibrators", {})).items():
        if not isinstance(raw_calibrator, dict):
            continue
        global_calibrator = _deserialize_probability_calibrator(dict(raw_calibrator.get("global", {})))
        if global_calibrator is None:
            continue
        regime_calibrators: dict[str, ProbabilityCalibrator] = {}
        for regime_key, regime_payload in dict(raw_calibrator.get("regimes", {})).items():
            regime_calibrator = _deserialize_probability_calibrator(dict(regime_payload))
            if regime_calibrator is not None:
                regime_calibrators[str(regime_key)] = regime_calibrator
        calibrators[str(model_name)] = RegimeAwareProbabilityCalibrator(
            global_calibrator=global_calibrator,
            regime_calibrators=regime_calibrators,
            sample_count=int(raw_calibrator.get("sample_count", global_calibrator.sample_count)),
        )
    return calibrators


def calibration_regime_key(
    snapshot: MarketSnapshot,
    *,
    near_money_bps: float = 150.0,
    high_vol_threshold: float = 0.8,
    tight_spread_cents: int = 4,
) -> str:
    signed_distance_bps = float(snapshot.metadata.get("signed_distance_bps", 0.0) or 0.0)
    minutes_to_expiry = max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)
    realized_volatility = float(snapshot.metadata.get("realized_volatility", 0.0) or 0.0)
    spread_cents = float(snapshot.metadata.get("spread_cents", 0.0) or 0.0)
    money_bucket = "near" if abs(signed_distance_bps) <= near_money_bps else "otm"
    expiry_bucket = "expiry_0_15" if minutes_to_expiry <= 15.0 else "expiry_15_30"
    vol_bucket = "high_vol" if realized_volatility >= high_vol_threshold else "low_vol"
    spread_bucket = "tight_spread" if spread_cents <= tight_spread_cents else "wide_spread"
    return "|".join((money_bucket, expiry_bucket, vol_bucket, spread_bucket))


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
) -> dict[str, RegimeAwareProbabilityCalibrator]:
    calibrators: dict[str, RegimeAwareProbabilityCalibrator] = {}
    for model_name in engine.models:
        if not getattr(engine.models[model_name], "supports_settlement_calibration", True):
            continue
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
        if calibrator is None:
            continue
        regime_calibrators: dict[str, ProbabilityCalibrator] = {}
        if "calibration_regime" in trades.columns:
            for regime_key, group in trades.groupby("calibration_regime"):
                regime_calibrator = fit_probability_calibrator(
                    group,
                    probability_column="raw_model_probability",
                    outcome_column="contract_won",
                    bucket_width=bucket_width,
                    min_samples=min_samples,
                    min_bucket_count=min_bucket_count,
                )
                if regime_calibrator is not None:
                    regime_calibrators[str(regime_key)] = regime_calibrator
        calibrators[model_name] = RegimeAwareProbabilityCalibrator(
            global_calibrator=calibrator,
            regime_calibrators=regime_calibrators,
            sample_count=calibrator.sample_count,
        )
    return calibrators
