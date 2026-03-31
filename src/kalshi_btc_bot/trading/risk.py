from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class RiskConfig:
    max_trade_notional: float
    max_session_notional: float
    max_open_positions: int
    max_positions_per_expiry: int
    max_daily_loss: float
    max_drawdown: float


@dataclass(frozen=True)
class RiskCheckResult:
    allowed: bool
    reason: str


@dataclass
class RiskManager:
    config: RiskConfig
    session_notional: float = 0.0
    realized_pnl: float = 0.0
    peak_equity: float = 0.0
    open_positions_by_expiry: dict[str, int] = field(default_factory=dict)

    def initialize(self, starting_bankroll: float) -> None:
        self.peak_equity = starting_bankroll

    def check_entry(
        self,
        *,
        entry_notional: float,
        expiry: datetime,
        open_position_count: int,
        current_equity: float,
    ) -> RiskCheckResult:
        if self.config.max_trade_notional > 0 and entry_notional > self.config.max_trade_notional:
            return RiskCheckResult(False, "Trade notional exceeds cap.")
        if self.config.max_session_notional > 0 and (self.session_notional + entry_notional) > self.config.max_session_notional:
            return RiskCheckResult(False, "Session notional cap exceeded.")
        if self.config.max_open_positions > 0 and open_position_count >= self.config.max_open_positions:
            return RiskCheckResult(False, "Open position cap exceeded.")
        expiry_key = expiry.isoformat()
        if self.config.max_positions_per_expiry > 0 and self.open_positions_by_expiry.get(expiry_key, 0) >= self.config.max_positions_per_expiry:
            return RiskCheckResult(False, "Per-expiry position cap exceeded.")
        if self.config.max_daily_loss > 0 and self.realized_pnl <= -self.config.max_daily_loss:
            return RiskCheckResult(False, "Daily loss cap exceeded.")
        drawdown = max(self.peak_equity - current_equity, 0.0)
        if self.config.max_drawdown > 0 and drawdown >= self.config.max_drawdown:
            return RiskCheckResult(False, "Drawdown cap exceeded.")
        return RiskCheckResult(True, "allowed")

    def record_entry(self, *, entry_notional: float, expiry: datetime) -> None:
        self.session_notional += entry_notional
        expiry_key = expiry.isoformat()
        self.open_positions_by_expiry[expiry_key] = self.open_positions_by_expiry.get(expiry_key, 0) + 1

    def record_exit(self, *, realized_pnl: float, expiry: datetime, current_equity: float) -> None:
        self.realized_pnl += realized_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        expiry_key = expiry.isoformat()
        remaining = self.open_positions_by_expiry.get(expiry_key, 0) - 1
        if remaining > 0:
            self.open_positions_by_expiry[expiry_key] = remaining
        else:
            self.open_positions_by_expiry.pop(expiry_key, None)
