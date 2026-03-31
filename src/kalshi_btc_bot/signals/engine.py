from __future__ import annotations

from dataclasses import dataclass
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
    adjusted = raw_probability - penalty if side == "yes" else raw_probability + penalty
    return clamp(adjusted, 0.0, 1.0)


def generate_signal(
    snapshot: MarketSnapshot,
    estimate: ProbabilityEstimate,
    config: SignalConfig,
    *,
    as_of: datetime | None = None,
) -> TradingSignal:
    base_probability = estimate.raw_probability if estimate.raw_probability is not None else estimate.probability
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
            reason="Market volume below liquidity threshold.",
        )

    candidates: list[TradingSignal] = []
    for side, raw_probability, calibrated_probability in (
        ("yes", base_probability, estimate.probability),
        ("no", 1.0 - base_probability, 1.0 - estimate.probability),
    ):
        entry_price_cents, market_probability = _entry_price_and_probability(snapshot, side)
        if entry_price_cents is None or market_probability is None:
            continue
        if entry_price_cents < config.min_contract_price_cents or entry_price_cents > config.max_contract_price_cents:
            continue
        spread_cents = _spread_cents(snapshot, side)
        if spread_cents is not None and spread_cents > config.max_spread_cents:
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
        quality_score = max(expected_value_cents or 0.0, 0.0) * max(edge or 0.0, 0.0) / max(1.0 + spread_penalty, 1.0)
        quality_score = quality_score / max(liquidity_scale, 1.0)
        if edge >= config.min_edge and conservative_probability >= config.min_confidence and expected_value_cents > 0:
            action = "buy_yes" if side == "yes" else "buy_no"
            reason = f"Conservative edge {edge:.4f} exceeds threshold."
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
                reason=reason,
                confidence=conservative_probability,
                quality_score=round(quality_score, 6),
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
        reason=best.reason,
        confidence=best.confidence,
        quality_score=best.quality_score,
        spread_cents=best.spread_cents,
        liquidity_penalty=best.liquidity_penalty,
    )
