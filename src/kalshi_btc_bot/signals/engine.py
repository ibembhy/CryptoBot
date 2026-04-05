from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate, TradingSignal
from kalshi_btc_bot.utils.math import clamp, probability_to_cents


@dataclass(frozen=True)
class SignalConfig:
    min_edge: float
    min_confidence: float
    min_contract_price_cents: int
    max_contract_price_cents: int
    max_near_money_bps: float | None = None
    min_liquidity: float = 0.0
    uncertainty_penalty: float = 0.0
    max_spread_cents: int = 99
    liquidity_penalty_per_100_volume: float = 0.0
    max_data_age_seconds: int = 120
    series_tier_profiles: dict[str, dict[str, object]] = field(default_factory=dict)


def _series_tier(
    snapshot: MarketSnapshot,
    side: str,
    entry_price_cents: int,
    config: SignalConfig,
) -> tuple[str | None, float, float, bool, str | None]:
    profile = config.series_tier_profiles.get(snapshot.series_ticker)
    if not profile:
        return None, 1.0, 0.0, True, None

    allowed_sides = profile.get("allowed_sides")
    if isinstance(allowed_sides, list) and allowed_sides and side not in {str(item) for item in allowed_sides}:
        return None, 0.0, -999.0, False, "Series profile excludes this side."

    minutes_to_expiry = max((snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0, 0.0)
    preferred_price_min = int(profile.get("preferred_price_min_cents", config.min_contract_price_cents))
    preferred_price_max = int(profile.get("preferred_price_max_cents", config.max_contract_price_cents))
    secondary_price_min = int(profile.get("secondary_price_min_cents", preferred_price_min))
    secondary_price_max = int(profile.get("secondary_price_max_cents", preferred_price_max))
    soft_price_max = int(profile.get("soft_price_max_cents", config.max_contract_price_cents))
    preferred_minutes_min = float(profile.get("preferred_minutes_min", 0.0))
    preferred_minutes_max = float(profile.get("preferred_minutes_max", 10_000.0))
    soft_minutes_min = float(profile.get("soft_minutes_min", 0.0))
    tier_d_enabled = not bool(profile.get("disable_tier_d", False))

    tier_label = "D"
    size_multiplier = float(profile.get("tier_d_size_multiplier", 0.1))
    quality_bonus = float(profile.get("tier_d_quality_bonus", -0.75))

    if preferred_price_min <= entry_price_cents <= preferred_price_max and preferred_minutes_min <= minutes_to_expiry <= preferred_minutes_max:
        tier_label = "A"
        size_multiplier = float(profile.get("tier_a_size_multiplier", 1.0))
        quality_bonus = float(profile.get("tier_a_quality_bonus", 0.75))
    elif secondary_price_min <= entry_price_cents < secondary_price_max and preferred_minutes_min <= minutes_to_expiry <= preferred_minutes_max:
        tier_label = "B"
        size_multiplier = float(profile.get("tier_b_size_multiplier", 0.7))
        quality_bonus = float(profile.get("tier_b_quality_bonus", 0.35))
    elif ((preferred_price_max < entry_price_cents <= soft_price_max) or (soft_minutes_min <= minutes_to_expiry < preferred_minutes_min)):
        tier_label = "C"
        size_multiplier = float(profile.get("tier_c_size_multiplier", 0.4))
        quality_bonus = float(profile.get("tier_c_quality_bonus", -0.1))

    if tier_label == "D" and not tier_d_enabled:
        return tier_label, size_multiplier, quality_bonus, False, "Series quality tier is experimental and disabled."
    return tier_label, size_multiplier, quality_bonus, True, None


def _entry_price_and_probability(snapshot: MarketSnapshot, side: str) -> tuple[int | None, float | None]:
    if side == "yes":
        if snapshot.yes_ask is None:
            return None, None
        return int(round(snapshot.yes_ask * 100.0)), snapshot.yes_ask
    if snapshot.no_ask is None:
        return None, None
    return int(round(snapshot.no_ask * 100.0)), snapshot.no_ask


def _exit_probability(snapshot: MarketSnapshot, side: str) -> float | None:
    if side == "yes":
        return snapshot.yes_bid
    return snapshot.no_bid


def _spread_cents(snapshot: MarketSnapshot, side: str) -> int | None:
    entry_price_cents, _ = _entry_price_and_probability(snapshot, side)
    exit_probability = _exit_probability(snapshot, side)
    if entry_price_cents is None or exit_probability is None:
        return None
    return entry_price_cents - int(round(exit_probability * 100.0))


def _staleness_seconds(snapshot: MarketSnapshot, *, as_of: datetime | None = None) -> float:
    now = as_of or datetime.now(timezone.utc)
    observed = snapshot.observed_at
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return max((now - observed).total_seconds(), 0.0)


def _conservative_probability(raw_probability: float, side: str, penalty: float) -> float:
    return clamp(raw_probability - penalty, 0.0, 1.0)


def generate_signal(
    snapshot: MarketSnapshot,
    estimate: ProbabilityEstimate,
    config: SignalConfig,
    *,
    as_of: datetime | None = None,
) -> TradingSignal:
    base_probability = estimate.raw_probability if estimate.raw_probability is not None else estimate.probability
    predicted_repricing_cents = estimate.inputs.get("predicted_delta_cents") if isinstance(estimate.inputs, dict) else None
    predicted_repricing_cents = float(predicted_repricing_cents) if predicted_repricing_cents is not None else None
    calibration_regime = str(estimate.inputs.get("calibration_regime", "")) if isinstance(estimate.inputs, dict) else ""
    spread_regime_label = str(snapshot.metadata.get("spread_regime_label", "normal"))
    liquidity_regime_label = str(snapshot.metadata.get("liquidity_regime_label", "normal"))
    btc_micro_jump_flag = float(snapshot.metadata.get("btc_micro_jump_flag", 0.0) or 0.0)
    stale_quote_flag = float(snapshot.metadata.get("stale_quote_flag", 0.0) or 0.0)
    if (
        config.max_near_money_bps is not None
        and snapshot.threshold is not None
        and snapshot.spot_price > 0
    ):
        distance_bps = abs(snapshot.threshold - snapshot.spot_price) / snapshot.spot_price * 10_000.0
        if distance_bps > config.max_near_money_bps:
            return TradingSignal(
                market_ticker=snapshot.market_ticker,
                action="no_action",
                side=None,
                raw_model_probability=base_probability,
                model_probability=estimate.probability,
                market_probability=snapshot.implied_probability,
                edge=None,
                raw_edge=None,
                entry_price_cents=None,
                fair_value_cents=None,
                expected_value_cents=None,
                predicted_repricing_cents=predicted_repricing_cents,
                reason="Contract is too far from spot.",
            )
    if _staleness_seconds(snapshot, as_of=as_of) > config.max_data_age_seconds:
        return TradingSignal(
            market_ticker=snapshot.market_ticker,
            action="no_action",
            side=None,
            raw_model_probability=base_probability,
            model_probability=estimate.probability,
            market_probability=snapshot.implied_probability,
            edge=None,
            raw_edge=None,
            entry_price_cents=None,
            fair_value_cents=None,
            expected_value_cents=None,
            predicted_repricing_cents=predicted_repricing_cents,
            reason="Snapshot data is stale.",
        )
    if snapshot.volume is not None and snapshot.volume < config.min_liquidity:
        return TradingSignal(
            market_ticker=snapshot.market_ticker,
            action="no_action",
            side=None,
            raw_model_probability=base_probability,
            model_probability=estimate.probability,
            market_probability=snapshot.implied_probability,
            edge=None,
            raw_edge=None,
            entry_price_cents=None,
            fair_value_cents=None,
            expected_value_cents=None,
            predicted_repricing_cents=predicted_repricing_cents,
            reason="Market volume below liquidity threshold.",
        )

    candidates: list[TradingSignal] = []
    for side, raw_probability, calibrated_probability in (
        ("yes", base_probability, estimate.probability),
        ("no", 1.0 - base_probability, 1.0 - estimate.probability),
    ):
        profit_probability = None
        if isinstance(estimate.inputs, dict):
            raw_profit_probability = estimate.inputs.get(f"profit_probability_{side}")
            if raw_profit_probability is not None:
                profit_probability = float(raw_profit_probability)
        entry_price_cents, market_probability = _entry_price_and_probability(snapshot, side)
        if entry_price_cents is None or market_probability is None:
            continue
        if entry_price_cents < config.min_contract_price_cents or entry_price_cents > config.max_contract_price_cents:
            continue
        spread_cents = _spread_cents(snapshot, side)
        tier_label, size_multiplier, tier_bonus, tier_enabled, tier_reason = _series_tier(
            snapshot,
            side,
            entry_price_cents,
            config,
        )
        if not tier_enabled:
            candidates.append(
                TradingSignal(
                    market_ticker=snapshot.market_ticker,
                    action="no_action",
                    side=side,  # type: ignore[arg-type]
                    raw_model_probability=raw_probability,
                    model_probability=calibrated_probability,
                    market_probability=market_probability,
                    edge=None,
                    raw_edge=None,
                    entry_price_cents=entry_price_cents,
                    fair_value_cents=None,
                    expected_value_cents=None,
                    predicted_repricing_cents=predicted_repricing_cents,
                    profit_probability=profit_probability,
                    reason=tier_reason or "Series tier disabled.",
                    tier_label=tier_label,
                    size_multiplier=size_multiplier,
                    spread_cents=spread_cents,
                    liquidity_penalty=0.0,
                )
            )
            continue
        conservative_probability = _conservative_probability(calibrated_probability, side, config.uncertainty_penalty)
        liquidity_penalty = 0.0
        if snapshot.volume is not None and snapshot.volume > 0:
            liquidity_penalty = config.liquidity_penalty_per_100_volume * (100.0 / snapshot.volume)
        edge = conservative_probability - market_probability - liquidity_penalty
        raw_edge = raw_probability - market_probability
        fair_value_cents = probability_to_cents(conservative_probability)
        expected_value_cents = round(fair_value_cents - entry_price_cents, 2)
        action = "no_action"
        reason = "Edge below threshold."
        spread_penalty = (spread_cents or 0) / 10.0
        liquidity_scale = 1.0 + liquidity_penalty * 25.0
        repricing_bonus = max(predicted_repricing_cents or 0.0, 0.0)
        profit_bonus = max((profit_probability or 0.5) - 0.5, 0.0) * 10.0
        quality_score = max(expected_value_cents or 0.0, 0.0) * max(edge or 0.0, 0.0) / max(1.0 + spread_penalty, 1.0)
        quality_score = quality_score / max(liquidity_scale, 1.0)
        quality_score += repricing_bonus / max(1.0 + spread_penalty + liquidity_scale, 1.0)
        quality_score += profit_bonus
        if spread_regime_label == "tight":
            quality_score += 0.35
        elif spread_regime_label == "wide":
            quality_score -= 0.5
        if liquidity_regime_label == "deep":
            quality_score += 0.25
        elif liquidity_regime_label == "thin":
            quality_score -= 0.25
        quality_score -= stale_quote_flag * 1.5
        quality_score -= btc_micro_jump_flag * 1.0
        quality_score += tier_bonus
        series_min_edge = config.min_edge
        if edge >= series_min_edge and conservative_probability >= config.min_confidence and expected_value_cents > 0:
            action = "buy_yes" if side == "yes" else "buy_no"
            tier_suffix = f" Tier {tier_label}." if tier_label else ""
            reason = f"Conservative edge {edge:.4f} exceeds threshold in {calibration_regime or 'global'} regime.{tier_suffix}"
        candidates.append(
            TradingSignal(
                market_ticker=snapshot.market_ticker,
                action=action,  # type: ignore[arg-type]
                side=side,  # type: ignore[arg-type]
                raw_model_probability=raw_probability,
                model_probability=conservative_probability,
                market_probability=market_probability,
                edge=edge,
                raw_edge=raw_edge,
                entry_price_cents=entry_price_cents,
                fair_value_cents=fair_value_cents,
                expected_value_cents=expected_value_cents,
                predicted_repricing_cents=predicted_repricing_cents,
                profit_probability=profit_probability,
                reason=reason,
                confidence=conservative_probability,
                quality_score=round(quality_score, 6),
                tier_label=tier_label,
                size_multiplier=size_multiplier,
                spread_cents=spread_cents,
                liquidity_penalty=round(liquidity_penalty, 6),
            )
        )

    if not candidates:
        return TradingSignal(
            market_ticker=snapshot.market_ticker,
            action="no_action",
            side=None,
            raw_model_probability=base_probability,
            model_probability=estimate.probability,
            market_probability=snapshot.implied_probability,
            edge=None,
            raw_edge=None,
            entry_price_cents=None,
            fair_value_cents=None,
            expected_value_cents=None,
            predicted_repricing_cents=predicted_repricing_cents,
            profit_probability=None,
            reason="No tradable prices within configured entry band.",
        )

    best = max(candidates, key=lambda signal: (signal.quality_score, (signal.edge or -999), signal.expected_value_cents or -999))
    if best.action != "no_action":
        return best
    return TradingSignal(
        market_ticker=snapshot.market_ticker,
        action="no_action",
        side=best.side,
        raw_model_probability=best.raw_model_probability,
        model_probability=best.model_probability,
        market_probability=best.market_probability,
        edge=best.edge,
        raw_edge=best.raw_edge,
        entry_price_cents=best.entry_price_cents,
        fair_value_cents=best.fair_value_cents,
        expected_value_cents=best.expected_value_cents,
        predicted_repricing_cents=best.predicted_repricing_cents,
        profit_probability=best.profit_probability,
        reason=best.reason,
        confidence=best.confidence,
        quality_score=best.quality_score,
        tier_label=best.tier_label,
        size_multiplier=best.size_multiplier,
        spread_cents=best.spread_cents,
        liquidity_penalty=best.liquidity_penalty,
    )
