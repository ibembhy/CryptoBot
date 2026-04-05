from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from kalshi_btc_bot.types import ExitDecision, MarketSnapshot


@dataclass(frozen=True)
class ExitConfig:
    take_profit_cents: int
    stop_loss_cents: int
    fair_value_buffer_cents: int
    time_exit_minutes_before_expiry: int
    min_hold_edge: float = 1.0
    min_ev_to_hold_cents: int = 999
    exit_liquidity_floor: float = 0.0
    disable_stop_loss: bool = False
    disable_early_exits: bool = False


def mark_price_cents(snapshot: MarketSnapshot, side: str) -> int | None:
    if side == "yes":
        return None if snapshot.yes_bid is None else int(round(snapshot.yes_bid * 100.0))
    return None if snapshot.no_bid is None else int(round(snapshot.no_bid * 100.0))


def unrealized_pnl(entry_price_cents: int, mark_price: int | None, contracts: int) -> float | None:
    if mark_price is None:
        return None
    return round(((mark_price - entry_price_cents) * contracts) / 100.0, 2)


def evaluate_exit(
    *,
    snapshot: MarketSnapshot,
    side: str,
    entry_price_cents: int,
    contracts: int,
    fair_value_cents: int,
    model_probability: float,
    config: ExitConfig,
    as_of: datetime,
    entry_edge: float = 0.0,
) -> ExitDecision:
    current_price = mark_price_cents(snapshot, side)
    pnl = unrealized_pnl(entry_price_cents, current_price, contracts)
    if current_price is None or pnl is None:
        return ExitDecision(action="hold", trigger=None, exit_price_cents=None, unrealized_pnl=None, reason="No exit bid available.")
    if snapshot.volume is not None and snapshot.volume < config.exit_liquidity_floor:
        return ExitDecision(action="hold", trigger=None, exit_price_cents=current_price, unrealized_pnl=pnl, reason="Exit liquidity below floor.")

    price_delta = current_price - entry_price_cents
    current_edge = model_probability - (current_price / 100.0)

    if not config.disable_stop_loss:
        if price_delta <= -config.stop_loss_cents:
            return ExitDecision(action="exit", trigger="stop_loss", exit_price_cents=current_price, unrealized_pnl=pnl, reason="Stop-loss threshold hit.")

    if not config.disable_early_exits:
        hold_edge_threshold = entry_edge if entry_edge > 0 else config.min_hold_edge
        if current_edge >= hold_edge_threshold:
            if price_delta >= config.take_profit_cents:
                return ExitDecision(action="hold", trigger=None, exit_price_cents=current_price, unrealized_pnl=pnl, reason="Take profit reached but model edge still above entry edge.")
        if price_delta >= config.take_profit_cents:
            return ExitDecision(action="exit", trigger="take_profit", exit_price_cents=current_price, unrealized_pnl=pnl, reason="Take-profit threshold hit.")
        if abs(current_price - fair_value_cents) <= config.fair_value_buffer_cents:
            return ExitDecision(action="exit", trigger="fair_value_convergence", exit_price_cents=current_price, unrealized_pnl=pnl, reason="Market converged to fair value.")
        if current_edge <= 0:
            return ExitDecision(action="exit", trigger="fair_value_convergence", exit_price_cents=current_price, unrealized_pnl=pnl, reason="Hold EV collapsed below exit EV.")

    minutes_to_expiry = (snapshot.expiry - snapshot.observed_at).total_seconds() / 60.0
    if minutes_to_expiry <= config.time_exit_minutes_before_expiry:
        return ExitDecision(action="exit", trigger="time_exit", exit_price_cents=current_price, unrealized_pnl=pnl, reason="Time exit before expiry.")
    return ExitDecision(action="hold", trigger=None, exit_price_cents=current_price, unrealized_pnl=pnl, reason="Hold position.")
