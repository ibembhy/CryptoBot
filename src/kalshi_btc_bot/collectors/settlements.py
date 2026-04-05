from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging

import requests

from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.markets.normalize import _settlement_price
from kalshi_btc_bot.storage.snapshots import SnapshotStore

log = logging.getLogger(__name__)


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class SettlementEnrichmentConfig:
    series_ticker: str | None = None
    max_markets: int | None = None
    only_expired: bool = True
    recent_hours: int | None = 48


class SettlementEnricher:
    def __init__(
        self,
        *,
        kalshi_client: KalshiClient,
        snapshot_store: SnapshotStore,
        config: SettlementEnrichmentConfig,
    ) -> None:
        self.kalshi_client = kalshi_client
        self.snapshot_store = snapshot_store
        self.config = config

    def enrich(self) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        expired_after = None
        if self.config.recent_hours is not None:
            expired_after = now - timedelta(hours=self.config.recent_hours)
        market_records = self.snapshot_store.list_market_records(
            series_ticker=self.config.series_ticker,
            expired_before=now if self.config.only_expired else None,
            expired_after=expired_after,
            unsettled_only=True,
            limit=self.config.max_markets,
        )
        market_tickers = [str(record["market_ticker"]) for record in market_records]

        updated_markets = 0
        settled_markets = 0
        for index, market_ticker in enumerate(market_tickers, start=1):
            log.info("Settlement enrichment %s/%s: %s", index, len(market_tickers), market_ticker)
            market_payload = self._fetch_market_payload(market_ticker)
            if not market_payload:
                continue
            settlement_price = self._extract_settlement_price(market_payload)
            if settlement_price is None:
                continue
            updated_rows = self.snapshot_store.update_market_settlement(market_ticker, settlement_price)
            if updated_rows > 0:
                updated_markets += 1
                settled_markets += 1
        return {
            "markets_checked": len(market_tickers),
            "markets_updated": updated_markets,
            "settled_markets_found": settled_markets,
            "expired_only": int(self.config.only_expired),
        }

    def _fetch_market_payload(self, market_ticker: str) -> dict | None:
        for fetcher in (self.kalshi_client.get_market, self.kalshi_client.get_historical_market):
            try:
                payload = fetcher(market_ticker)
                market = payload.get("market")
                if market:
                    return market
            except requests.HTTPError:
                continue
        return None

    def _extract_settlement_price(self, market: dict) -> float | None:
        return _settlement_price(market)
