from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PaperFill:
    market_ticker: str
    side: str
    contracts: int
    price_cents: int
    timestamp: datetime
    fees_paid: float


@dataclass
class PaperBroker:
    fills: list[PaperFill] = field(default_factory=list)

    def record_fill(
        self,
        *,
        market_ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        timestamp: datetime,
        fees_paid: float = 0.0,
    ) -> PaperFill:
        fill = PaperFill(
            market_ticker=market_ticker,
            side=side,
            contracts=contracts,
            price_cents=price_cents,
            timestamp=timestamp,
            fees_paid=fees_paid,
        )
        self.fills.append(fill)
        return fill
