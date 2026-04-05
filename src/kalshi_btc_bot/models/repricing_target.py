from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_btc_bot.models.base import ProbabilityModel
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate
from kalshi_btc_bot.utils.math import clamp
from kalshi_btc_bot.utils.time import ensure_utc


REPRICING_FEATURE_NAMES = (
    "current_yes_probability",
    "recent_log_return_bps",
    "return_3_bps",
    "return_5_bps",
    "price_acceleration_bps",
    "realized_volatility_1",
    "realized_volatility_3",
    "realized_volatility_5",
    "realized_volatility",
    "vol_of_vol",
    "signed_distance_bps",
    "minutes_to_expiry",
    "spread_cents",
    "liquidity_log",
    "kalshi_momentum_1_cents",
    "kalshi_momentum_3_cents",
    "kalshi_reversion_gap_cents",
    "spread_regime_cents",
    "liquidity_regime",
    "quote_age_seconds",
    "stale_quote_flag",
    "btc_micro_jump_flag",
)


@dataclass(frozen=True)
class RepricingRegimeProfile:
    intercept_cents: float
    coefficients: dict[str, float]
    sample_count: int
    target_mean_cents: float
    target_std_cents: float
    yes_profit_rate: float
    no_profit_rate: float


@dataclass(frozen=True)
class RepricingModelProfile:
    horizon_minutes: int
    sample_count: int
    fallback: RepricingRegimeProfile
    regime_profiles: dict[str, RepricingRegimeProfile]
    near_money_regime_bps: float
    high_vol_regime_threshold: float


def _serialize_regime(profile: RepricingRegimeProfile) -> dict[str, object]:
    return {
        "intercept_cents": profile.intercept_cents,
        "coefficients": profile.coefficients,
        "sample_count": profile.sample_count,
        "target_mean_cents": profile.target_mean_cents,
        "target_std_cents": profile.target_std_cents,
        "yes_profit_rate": profile.yes_profit_rate,
        "no_profit_rate": profile.no_profit_rate,
    }


def _deserialize_regime(payload: dict[str, object]) -> RepricingRegimeProfile | None:
    coefficients = payload.get("coefficients", {})
    if not coefficients:
        return None
    return RepricingRegimeProfile(
        intercept_cents=float(payload.get("intercept_cents", 0.0)),
        coefficients={str(key): float(value) for key, value in coefficients.items()},
        sample_count=int(payload.get("sample_count", 0)),
        target_mean_cents=float(payload.get("target_mean_cents", 0.0)),
        target_std_cents=float(payload.get("target_std_cents", 0.0)),
        yes_profit_rate=float(payload.get("yes_profit_rate", 0.5)),
        no_profit_rate=float(payload.get("no_profit_rate", 0.5)),
    )


def save_repricing_profile(path: str | Path, profile: RepricingModelProfile, *, metadata: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "profile": {
            "horizon_minutes": profile.horizon_minutes,
            "sample_count": profile.sample_count,
            "near_money_regime_bps": profile.near_money_regime_bps,
            "high_vol_regime_threshold": profile.high_vol_regime_threshold,
            "fallback": _serialize_regime(profile.fallback),
            "regime_profiles": {name: _serialize_regime(regime) for name, regime in profile.regime_profiles.items()},
        },
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_repricing_profile(path: str | Path, *, expected_metadata: dict[str, object]) -> RepricingModelProfile | None:
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
    profile_payload = payload.get("profile", {})
    fallback_payload = profile_payload.get("fallback")
    if fallback_payload:
        fallback = _deserialize_regime(fallback_payload)
        if fallback is None:
            return None
        regime_profiles = {
            str(name): regime
            for name, raw_regime in dict(profile_payload.get("regime_profiles", {})).items()
            if (regime := _deserialize_regime(raw_regime)) is not None
        }
        return RepricingModelProfile(
            horizon_minutes=int(profile_payload.get("horizon_minutes", 5)),
            sample_count=int(profile_payload.get("sample_count", fallback.sample_count)),
            fallback=fallback,
            regime_profiles=regime_profiles,
            near_money_regime_bps=float(profile_payload.get("near_money_regime_bps", 150.0)),
            high_vol_regime_threshold=float(profile_payload.get("high_vol_regime_threshold", 0.8)),
        )
    coefficients = profile_payload.get("coefficients", {})
    if not coefficients:
        return None
    legacy = RepricingRegimeProfile(
        intercept_cents=float(profile_payload.get("intercept_cents", 0.0)),
        coefficients={str(key): float(value) for key, value in coefficients.items()},
        sample_count=int(profile_payload.get("sample_count", 0)),
        target_mean_cents=float(profile_payload.get("target_mean_cents", 0.0)),
        target_std_cents=float(profile_payload.get("target_std_cents", 0.0)),
        yes_profit_rate=0.5,
        no_profit_rate=0.5,
    )
    return RepricingModelProfile(
        horizon_minutes=int(profile_payload.get("horizon_minutes", 5)),
        sample_count=legacy.sample_count,
        fallback=legacy,
        regime_profiles={},
        near_money_regime_bps=float(profile_payload.get("near_money_regime_bps", 150.0)),
        high_vol_regime_threshold=float(profile_payload.get("high_vol_regime_threshold", 0.8)),
    )


def current_yes_probability(snapshot: MarketSnapshot) -> float | None:
    if snapshot.yes_bid is not None and snapshot.yes_ask is not None:
        return clamp((float(snapshot.yes_bid) + float(snapshot.yes_ask)) / 2.0, 0.0, 1.0)
    if snapshot.mid_price is not None:
        return clamp(float(snapshot.mid_price), 0.0, 1.0)
    if snapshot.yes_ask is not None:
        return clamp(float(snapshot.yes_ask), 0.0, 1.0)
    if snapshot.yes_bid is not None:
        return clamp(float(snapshot.yes_bid), 0.0, 1.0)
    if snapshot.no_bid is not None and snapshot.no_ask is not None:
        return clamp(1.0 - ((float(snapshot.no_bid) + float(snapshot.no_ask)) / 2.0), 0.0, 1.0)
    return snapshot.implied_probability


def snapshot_spread_cents(snapshot: MarketSnapshot) -> float:
    if snapshot.yes_bid is not None and snapshot.yes_ask is not None:
        return max((float(snapshot.yes_ask) - float(snapshot.yes_bid)) * 100.0, 0.0)
    if snapshot.no_bid is not None and snapshot.no_ask is not None:
        return max((float(snapshot.no_ask) - float(snapshot.no_bid)) * 100.0, 0.0)
    return 0.0


def snapshot_signed_distance_bps(snapshot: MarketSnapshot) -> float:
    if snapshot.threshold is None or snapshot.spot_price <= 0:
        return 0.0
    raw = (float(snapshot.spot_price) - float(snapshot.threshold)) / float(snapshot.spot_price) * 10_000.0
    if snapshot.direction == "below":
        return -raw
    return raw


def snapshot_repricing_features(snapshot: MarketSnapshot) -> dict[str, float]:
    recent_log_return = float(snapshot.metadata.get("recent_log_return", 0.0) or 0.0)
    return_3 = float(snapshot.metadata.get("log_return_3", 0.0) or 0.0)
    return_5 = float(snapshot.metadata.get("log_return_5", 0.0) or 0.0)
    price_acceleration = float(snapshot.metadata.get("price_acceleration", 0.0) or 0.0)
    realized_volatility = float(snapshot.metadata.get("realized_volatility", 0.0) or 0.0)
    realized_volatility_1 = float(snapshot.metadata.get("realized_volatility_1", 0.0) or 0.0)
    realized_volatility_3 = float(snapshot.metadata.get("realized_volatility_3", 0.0) or 0.0)
    realized_volatility_5 = float(snapshot.metadata.get("realized_volatility_5", 0.0) or 0.0)
    vol_of_vol = float(snapshot.metadata.get("vol_of_vol", 0.0) or 0.0)
    minutes_to_expiry = max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)
    liquidity = max(float(snapshot.volume or 0.0) + float(snapshot.open_interest or 0.0), 0.0)
    return {
        "current_yes_probability": float(current_yes_probability(snapshot) or 0.5),
        "recent_log_return_bps": recent_log_return * 10_000.0,
        "return_3_bps": return_3 * 10_000.0,
        "return_5_bps": return_5 * 10_000.0,
        "price_acceleration_bps": price_acceleration * 10_000.0,
        "realized_volatility_1": realized_volatility_1,
        "realized_volatility_3": realized_volatility_3,
        "realized_volatility_5": realized_volatility_5,
        "realized_volatility": realized_volatility,
        "vol_of_vol": vol_of_vol,
        "signed_distance_bps": snapshot_signed_distance_bps(snapshot),
        "minutes_to_expiry": minutes_to_expiry,
        "spread_cents": snapshot_spread_cents(snapshot),
        "liquidity_log": float(np.log1p(liquidity)),
        "kalshi_momentum_1_cents": float(snapshot.metadata.get("kalshi_momentum_1_cents", 0.0) or 0.0),
        "kalshi_momentum_3_cents": float(snapshot.metadata.get("kalshi_momentum_3_cents", 0.0) or 0.0),
        "kalshi_reversion_gap_cents": float(snapshot.metadata.get("kalshi_reversion_gap_cents", 0.0) or 0.0),
        "spread_regime_cents": float(snapshot.metadata.get("spread_regime_cents", 0.0) or 0.0),
        "liquidity_regime": float(snapshot.metadata.get("liquidity_regime", 0.0) or 0.0),
        "quote_age_seconds": float(snapshot.metadata.get("quote_age_seconds", 0.0) or 0.0),
        "stale_quote_flag": float(snapshot.metadata.get("stale_quote_flag", 0.0) or 0.0),
        "btc_micro_jump_flag": float(snapshot.metadata.get("btc_micro_jump_flag", 0.0) or 0.0),
    }


def attach_snapshot_microstructure(
    snapshot: MarketSnapshot,
    market_snapshots: list[MarketSnapshot],
    index: int,
    *,
    stale_quote_threshold_seconds: float = 90.0,
) -> None:
    current_mid = current_yes_probability(snapshot) or 0.5
    current_spread = snapshot_spread_cents(snapshot)
    current_liquidity_log = float(np.log1p(max(float(snapshot.volume or 0.0) + float(snapshot.open_interest or 0.0), 0.0)))

    previous = market_snapshots[max(0, index - 3) : index]
    previous_mids = [float(current_yes_probability(item) or 0.5) for item in previous]
    previous_spreads = [snapshot_spread_cents(item) for item in previous]
    previous_liquidities = [
        float(np.log1p(max(float(item.volume or 0.0) + float(item.open_interest or 0.0), 0.0)))
        for item in previous
    ]

    prev_mid = previous_mids[-1] if previous_mids else current_mid
    prev_mid_3 = previous_mids[0] if previous_mids else current_mid
    rolling_mid = float(np.mean(previous_mids)) if previous_mids else current_mid
    avg_spread = float(np.mean(previous_spreads)) if previous_spreads else current_spread
    avg_liquidity = float(np.mean(previous_liquidities)) if previous_liquidities else current_liquidity_log
    if previous:
        quote_age_seconds = max((snapshot.observed_at - previous[-1].observed_at).total_seconds(), 0.0)
    else:
        quote_age_seconds = 0.0

    snapshot.metadata.update(
        {
            "kalshi_momentum_1_cents": round((current_mid - prev_mid) * 100.0, 4),
            "kalshi_momentum_3_cents": round((current_mid - prev_mid_3) * 100.0, 4),
            "kalshi_reversion_gap_cents": round((rolling_mid - current_mid) * 100.0, 4),
            "spread_regime_cents": round(current_spread - avg_spread, 4),
            "liquidity_regime": round(current_liquidity_log - avg_liquidity, 6),
            "quote_age_seconds": round(float(quote_age_seconds), 4),
            "stale_quote_flag": 1.0 if quote_age_seconds > stale_quote_threshold_seconds else 0.0,
        }
    )


def repricing_regime_key(
    *,
    signed_distance_bps: float,
    realized_volatility: float,
    near_money_regime_bps: float,
    high_vol_regime_threshold: float,
) -> str:
    money_bucket = "near" if abs(signed_distance_bps) <= near_money_regime_bps else "far"
    vol_bucket = "high_vol" if realized_volatility >= high_vol_regime_threshold else "low_vol"
    return f"{money_bucket}|{vol_bucket}"


def _fit_regime_profile(
    rows: list[list[float]],
    targets: list[float],
    *,
    yes_profit_rate: float,
    no_profit_rate: float,
    ridge_lambda: float,
) -> RepricingRegimeProfile:
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-9, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std
    xtx = x_scaled.T @ x_scaled
    ridge = ridge_lambda * np.eye(x_scaled.shape[1])
    beta_scaled = np.linalg.solve(xtx + ridge, x_scaled.T @ y)
    beta = beta_scaled / x_std
    intercept = float(y.mean() - x_mean @ beta)
    coefficients = {name: float(value) for name, value in zip(REPRICING_FEATURE_NAMES, beta.tolist(), strict=True)}
    return RepricingRegimeProfile(
        intercept_cents=intercept,
        coefficients=coefficients,
        sample_count=int(len(y)),
        target_mean_cents=float(y.mean()),
        target_std_cents=float(y.std(ddof=0)),
        yes_profit_rate=float(yes_profit_rate),
        no_profit_rate=float(no_profit_rate),
    )


def train_repricing_profile(
    snapshots: list[MarketSnapshot],
    *,
    feature_frame: pd.DataFrame,
    horizon_minutes: int = 5,
    min_samples: int = 150,
    min_regime_samples: int = 75,
    ridge_lambda: float = 5.0,
    max_abs_target_cents: float = 25.0,
    near_money_regime_bps: float = 150.0,
    high_vol_regime_threshold: float = 0.8,
    profit_cost_cents: float = 2.0,
) -> RepricingModelProfile | None:
    if not snapshots or feature_frame.empty:
        return None

    rows: list[list[float]] = []
    targets: list[float] = []
    yes_profit_labels: list[float] = []
    no_profit_labels: list[float] = []
    regime_rows: dict[str, list[list[float]]] = {}
    regime_targets: dict[str, list[float]] = {}
    regime_yes_profit: dict[str, list[float]] = {}
    regime_no_profit: dict[str, list[float]] = {}
    grouped: dict[str, list[MarketSnapshot]] = {}
    for snapshot in snapshots:
        grouped.setdefault(snapshot.market_ticker, []).append(snapshot)

    horizon = timedelta(minutes=horizon_minutes)
    for market_snapshots in grouped.values():
        market_snapshots = sorted(market_snapshots, key=lambda item: item.observed_at)
        for index, snapshot in enumerate(market_snapshots[:-1]):
            current_mid = current_yes_probability(snapshot)
            if current_mid is None or snapshot.observed_at >= snapshot.expiry:
                continue
            target_time = ensure_utc(snapshot.observed_at) + horizon
            future_snapshot = None
            for candidate in market_snapshots[index + 1 :]:
                if ensure_utc(candidate.observed_at) >= target_time and candidate.observed_at < candidate.expiry:
                    future_snapshot = candidate
                    break
            if future_snapshot is None:
                continue
            future_mid = current_yes_probability(future_snapshot)
            if future_mid is None:
                continue

            feature_row = _feature_row_for(feature_frame, snapshot.observed_at)
            if feature_row is None:
                continue
            snapshot.metadata.update(feature_row)
            attach_snapshot_microstructure(snapshot, market_snapshots, index)
            feature_values = snapshot_repricing_features(snapshot)
            feature_vector = [feature_values[name] for name in REPRICING_FEATURE_NAMES]
            rows.append(feature_vector)
            target_cents = clamp((future_mid - current_mid) * 100.0, -max_abs_target_cents, max_abs_target_cents)
            targets.append(float(target_cents))
            yes_entry = float(snapshot.yes_ask) * 100.0 if snapshot.yes_ask is not None else None
            yes_exit = float(future_snapshot.yes_bid) * 100.0 if future_snapshot.yes_bid is not None else None
            no_entry = float(snapshot.no_ask) * 100.0 if snapshot.no_ask is not None else None
            no_exit = float(future_snapshot.no_bid) * 100.0 if future_snapshot.no_bid is not None else None
            yes_profitable = 1.0 if yes_entry is not None and yes_exit is not None and yes_exit - yes_entry > profit_cost_cents else 0.0
            no_profitable = 1.0 if no_entry is not None and no_exit is not None and no_exit - no_entry > profit_cost_cents else 0.0
            yes_profit_labels.append(yes_profitable)
            no_profit_labels.append(no_profitable)
            regime = repricing_regime_key(
                signed_distance_bps=feature_values["signed_distance_bps"],
                realized_volatility=feature_values["realized_volatility"],
                near_money_regime_bps=near_money_regime_bps,
                high_vol_regime_threshold=high_vol_regime_threshold,
            )
            regime_rows.setdefault(regime, []).append(feature_vector)
            regime_targets.setdefault(regime, []).append(float(target_cents))
            regime_yes_profit.setdefault(regime, []).append(yes_profitable)
            regime_no_profit.setdefault(regime, []).append(no_profitable)

    if len(rows) < min_samples:
        return None

    fallback = _fit_regime_profile(
        rows,
        targets,
        yes_profit_rate=float(np.mean(yes_profit_labels)) if yes_profit_labels else 0.5,
        no_profit_rate=float(np.mean(no_profit_labels)) if no_profit_labels else 0.5,
        ridge_lambda=ridge_lambda,
    )
    regime_profiles: dict[str, RepricingRegimeProfile] = {}
    for regime, regime_target_values in regime_targets.items():
        if len(regime_target_values) < min_regime_samples:
            continue
        regime_profiles[regime] = _fit_regime_profile(
            regime_rows[regime],
            regime_target_values,
            yes_profit_rate=float(np.mean(regime_yes_profit[regime])) if regime_yes_profit[regime] else fallback.yes_profit_rate,
            no_profit_rate=float(np.mean(regime_no_profit[regime])) if regime_no_profit[regime] else fallback.no_profit_rate,
            ridge_lambda=ridge_lambda,
        )
    return RepricingModelProfile(
        horizon_minutes=horizon_minutes,
        sample_count=int(len(targets)),
        fallback=fallback,
        regime_profiles=regime_profiles,
        near_money_regime_bps=near_money_regime_bps,
        high_vol_regime_threshold=high_vol_regime_threshold,
    )


def _feature_row_for(feature_frame: pd.DataFrame, observed_at) -> dict[str, float] | None:
    eligible = feature_frame.loc[feature_frame.index <= ensure_utc(observed_at)]
    if eligible.empty:
        return None
    latest = eligible.iloc[-1]
    return {
        "recent_log_return": 0.0 if pd.isna(latest.get("log_return")) else float(latest.get("log_return")),
        "log_return_3": 0.0 if pd.isna(latest.get("log_return_3")) else float(latest.get("log_return_3")),
        "log_return_5": 0.0 if pd.isna(latest.get("log_return_5")) else float(latest.get("log_return_5")),
        "price_acceleration": 0.0 if pd.isna(latest.get("price_acceleration")) else float(latest.get("price_acceleration")),
        "realized_volatility_1": 0.0 if pd.isna(latest.get("realized_volatility_1")) else float(latest.get("realized_volatility_1")),
        "realized_volatility_3": 0.0 if pd.isna(latest.get("realized_volatility_3")) else float(latest.get("realized_volatility_3")),
        "realized_volatility_5": 0.0 if pd.isna(latest.get("realized_volatility_5")) else float(latest.get("realized_volatility_5")),
        "realized_volatility": 0.0 if pd.isna(latest.get("realized_volatility")) else float(latest.get("realized_volatility")),
        "vol_of_vol": 0.0 if pd.isna(latest.get("vol_of_vol")) else float(latest.get("vol_of_vol")),
        "btc_micro_jump_flag": 0.0 if pd.isna(latest.get("btc_micro_jump_flag")) else float(latest.get("btc_micro_jump_flag")),
    }


@dataclass
class RepricingTargetModel(ProbabilityModel):
    profile: RepricingModelProfile
    volatility_floor: float = 0.05
    max_probability_shift: float = 0.2
    model_name: str = "repricing_target"
    supports_settlement_calibration: bool = False

    def estimate(self, snapshot: MarketSnapshot, volatility: float) -> ProbabilityEstimate:
        sigma = max(volatility, self.volatility_floor)
        feature_values = snapshot_repricing_features(snapshot)
        regime = repricing_regime_key(
            signed_distance_bps=feature_values["signed_distance_bps"],
            realized_volatility=feature_values["realized_volatility"],
            near_money_regime_bps=self.profile.near_money_regime_bps,
            high_vol_regime_threshold=self.profile.high_vol_regime_threshold,
        )
        regime_profile = self.profile.regime_profiles.get(regime, self.profile.fallback)
        predicted_delta_cents = regime_profile.intercept_cents
        for name, coefficient in regime_profile.coefficients.items():
            predicted_delta_cents += coefficient * feature_values[name]
        current_probability = feature_values["current_yes_probability"]
        normalized_shift = math.tanh(predicted_delta_cents / 10.0) * self.max_probability_shift
        repriced_probability = clamp(current_probability + normalized_shift, 0.0, 1.0)
        target = snapshot.threshold if snapshot.threshold is not None else snapshot.spot_price
        return ProbabilityEstimate(
            model_name=self.model_name,
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=float(target),
            volatility=sigma,
            drift=0.0,
            probability=repriced_probability,
            inputs={
                **feature_values,
                "repricing_regime": regime,
                "predicted_delta_cents": round(predicted_delta_cents, 4),
                "normalized_probability_shift": round(normalized_shift, 6),
                "horizon_minutes": self.profile.horizon_minutes,
                "training_samples": regime_profile.sample_count,
                "global_training_samples": self.profile.sample_count,
                "target_mean_cents": round(regime_profile.target_mean_cents, 4),
                "target_std_cents": round(regime_profile.target_std_cents, 4),
                "profit_probability_yes": round(regime_profile.yes_profit_rate, 6),
                "profit_probability_no": round(regime_profile.no_profit_rate, 6),
            },
            notes="Empirical short-horizon repricing estimate trained on future Kalshi price changes.",
        )
