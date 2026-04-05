from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from kalshi_btc_bot.types import Position


@dataclass
class PositionBook:
    positions: dict[str, Position] = field(default_factory=dict)
    _counter: int = 0

    def has_open_position(self, market_ticker: str) -> bool:
        position = self.positions.get(market_ticker)
        return bool(position and position.status == "open")

    def open_position(
        self,
        *,
        market_ticker: str,
        side: str,
        contracts: int,
        entry_time: datetime,
        entry_price_cents: int,
        strategy_mode: str,
        signal_time: datetime | None = None,
        intended_entry_price_cents: int | None = None,
        entry_slippage_cents: int | None = None,
        expiry: datetime | None = None,
        entry_fees_paid: float = 0.0,
        entry_edge: float = 0.0,
    ) -> Position:
        self._counter += 1
        position = Position(
            position_id=f"pos-{self._counter}",
            market_ticker=market_ticker,
            side=side,  # type: ignore[arg-type]
            contracts=contracts,
            entry_time=entry_time,
            entry_price_cents=entry_price_cents,
            strategy_mode=strategy_mode,
            signal_time=signal_time,
            intended_entry_price_cents=intended_entry_price_cents,
            entry_slippage_cents=entry_slippage_cents,
            expiry=expiry,
            entry_fees_paid=entry_fees_paid,
            entry_edge=entry_edge,
        )
        self.positions[market_ticker] = position
        return position

    def close_position(
        self,
        market_ticker: str,
        *,
        exit_time: datetime,
        exit_price_cents: int,
        exit_trigger: str,
        realized_pnl: float,
    ) -> Position:
        position = self.positions[market_ticker]
        position.status = "closed"
        position.exit_time = exit_time
        position.exit_price_cents = exit_price_cents
        position.exit_trigger = exit_trigger  # type: ignore[assignment]
        position.realized_pnl = realized_pnl
        return position
