from __future__ import annotations

from dataclasses import dataclass

from kalshi_btc_bot.types import TradingSignal


@dataclass(frozen=True)
class FusionConfig:
    mode: str
    primary_model: str
    confirm_model: str
    primary_weight: float
    confirm_weight: float
    require_side_agreement: bool = True
    min_combined_edge: float = 0.0
    allow_primary_unconfirmed: bool = False


def fuse_signals(
    signals: dict[str, TradingSignal],
    *,
    config: FusionConfig,
) -> TradingSignal:
    primary = signals[config.primary_model]
    confirm = signals[config.confirm_model]
    if config.mode != "hybrid":
        return primary

    primary_actionable = primary.action != "no_action" and primary.side is not None
    confirm_actionable = confirm.action != "no_action" and confirm.side is not None

    if config.require_side_agreement and primary.side is not None and confirm.side is not None and primary.side != confirm.side:
        return TradingSignal(
            market_ticker=primary.market_ticker,
            action="no_action",
            side=None,
            raw_model_probability=primary.raw_model_probability,
            model_probability=primary.model_probability,
            market_probability=primary.market_probability,
            edge=None,
            raw_edge=None,
            entry_price_cents=None,
            fair_value_cents=None,
            expected_value_cents=None,
            reason="Primary and confirm models disagree on side.",
        )

    if not primary_actionable and not confirm_actionable:
        best = max(signals.values(), key=lambda signal: signal.quality_score)
        return TradingSignal(
            market_ticker=best.market_ticker,
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

    if primary_actionable and not confirm_actionable and not config.allow_primary_unconfirmed:
        return TradingSignal(
            market_ticker=primary.market_ticker,
            action="no_action",
            side=primary.side,
            raw_model_probability=primary.raw_model_probability,
            model_probability=primary.model_probability,
            market_probability=primary.market_probability,
            edge=primary.edge,
            raw_edge=primary.raw_edge,
            entry_price_cents=primary.entry_price_cents,
            fair_value_cents=primary.fair_value_cents,
            expected_value_cents=primary.expected_value_cents,
            reason=f"Primary model actionable but {config.confirm_model} did not confirm.",
            confidence=primary.confidence,
            quality_score=primary.quality_score,
            spread_cents=primary.spread_cents,
            liquidity_penalty=primary.liquidity_penalty,
        )

    chosen = primary if primary_actionable else confirm
    side = chosen.side
    primary_weight = config.primary_weight
    confirm_weight = config.confirm_weight
    total_weight = primary_weight + confirm_weight
    combined_probability = (
        primary.model_probability * primary_weight + confirm.model_probability * confirm_weight
    ) / total_weight
    combined_raw_probability = (
        primary.raw_model_probability * primary_weight + confirm.raw_model_probability * confirm_weight
    ) / total_weight
    combined_edge = ((primary.edge or 0.0) * primary_weight + (confirm.edge or 0.0) * confirm_weight) / total_weight
    combined_raw_edge = ((primary.raw_edge or 0.0) * primary_weight + (confirm.raw_edge or 0.0) * confirm_weight) / total_weight
    combined_quality = ((primary.quality_score or 0.0) * primary_weight + (confirm.quality_score or 0.0) * confirm_weight) / total_weight
    entry_price = chosen.entry_price_cents
    fair_value = round(((primary.fair_value_cents or 0) * primary_weight + (confirm.fair_value_cents or 0) * confirm_weight) / total_weight)
    expected_value = round(fair_value - (entry_price or 0), 2) if entry_price is not None else None
    action = chosen.action
    reason = "Hybrid fusion confirmed trade."
    if combined_edge < config.min_combined_edge:
        action = "no_action"
        reason = "Combined edge below hybrid threshold."
    return TradingSignal(
        market_ticker=chosen.market_ticker,
        action=action,
        side=side,
        raw_model_probability=combined_raw_probability,
        model_probability=combined_probability,
        market_probability=chosen.market_probability,
        edge=combined_edge,
        raw_edge=combined_raw_edge,
        entry_price_cents=entry_price,
        fair_value_cents=fair_value,
        expected_value_cents=expected_value,
        reason=reason,
        confidence=combined_probability,
        quality_score=combined_quality,
        spread_cents=chosen.spread_cents,
        liquidity_penalty=chosen.liquidity_penalty,
    )
