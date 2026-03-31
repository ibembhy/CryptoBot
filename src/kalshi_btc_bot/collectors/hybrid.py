from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import websockets

from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.markets.normalize import normalize_market
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot

log = logging.getLogger(__name__)


@dataclass
class HybridCollectorConfig:
    series_ticker: str
    status: str = "open"
    market_limit: int = 200
    min_minutes_to_expiry: int = 0
    max_minutes_to_expiry: int | None = 180
    reconcile_interval_seconds: int = 60
    spot_refresh_interval_seconds: int = 10
    websocket_channels: list[str] = field(default_factory=lambda: ["ticker"])
    websocket_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"


class HybridCollector:
    def __init__(
        self,
        *,
        kalshi_client: KalshiClient,
        coinbase_client: CoinbaseClient,
        snapshot_store: SnapshotStore,
        config: HybridCollectorConfig,
        websocket_headers_factory: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        self.kalshi_client = kalshi_client
        self.coinbase_client = coinbase_client
        self.snapshot_store = snapshot_store
        self.config = config
        self.websocket_headers_factory = websocket_headers_factory
        self.market_registry: dict[str, dict[str, Any]] = {}
        self._spot_cache: tuple[float, datetime] | None = None

    async def collect_forever(self) -> None:
        await self.bootstrap()
        await asyncio.gather(self._reconcile_loop(), self._websocket_loop())

    async def bootstrap(self) -> list[MarketSnapshot]:
        observed_at = datetime.now(timezone.utc)
        spot = self._get_spot_price(observed_at)
        raw_markets = self._discover_markets(observed_at=observed_at, spot_price=spot)
        snapshots = [
            normalize_market(
                raw,
                spot_price=spot,
                observed_at=observed_at,
                series_ticker_override=self.config.series_ticker,
            )
            for raw in raw_markets
        ]
        for raw, snapshot in zip(raw_markets, snapshots, strict=False):
            self.market_registry[snapshot.market_ticker] = dict(raw)
            self.snapshot_store.insert_snapshot(snapshot)
        return snapshots

    async def reconcile_once(self) -> list[MarketSnapshot]:
        return await self.bootstrap()

    async def handle_websocket_payload(self, payload: dict[str, Any]) -> MarketSnapshot | None:
        if payload.get("type") != "ticker":
            return None
        msg = payload.get("msg", {})
        market_ticker = msg.get("market_ticker")
        if not market_ticker or market_ticker not in self.market_registry:
            return None
        observed_at = datetime.fromtimestamp(float(msg.get("ts", datetime.now(timezone.utc).timestamp())), tz=timezone.utc)
        raw_market = dict(self.market_registry[market_ticker])
        raw_market.update(
            {
                "ticker": market_ticker,
                "yes_bid_dollars": msg.get("yes_bid_dollars"),
                "yes_ask_dollars": msg.get("yes_ask_dollars"),
                "yes_bid": msg.get("yes_bid"),
                "yes_ask": msg.get("yes_ask"),
                "volume": msg.get("volume"),
                "open_interest": msg.get("open_interest"),
            }
        )
        spot = self._get_spot_price(observed_at)
        snapshot = normalize_market(
            raw_market,
            spot_price=spot,
            observed_at=observed_at,
            series_ticker_override=self.config.series_ticker,
        )
        self.snapshot_store.insert_snapshot(snapshot)
        return snapshot

    def _discover_markets(self, *, observed_at: datetime, spot_price: float) -> list[dict[str, Any]]:
        raw_markets: list[dict[str, Any]] = []
        cursor: str | None = None
        max_pages = 10
        target_pool_size = max(self.config.market_limit * 3, self.config.market_limit)
        for _ in range(max_pages):
            page = self.kalshi_client.list_markets_page(
                series_ticker=self.config.series_ticker,
                status=self.config.status,
                limit=self.config.market_limit,
                cursor=cursor,
            )
            page_markets = list(page.get("markets", []))
            if not page_markets:
                break
            raw_markets.extend(page_markets)
            cursor = page.get("cursor")
            if len(raw_markets) >= target_pool_size or not cursor:
                break
        return self._filter_and_rank_markets(raw_markets, observed_at, spot_price)

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await self.reconcile_once()
            except Exception as exc:
                log.warning("REST reconcile failed: %s", exc)
            await asyncio.sleep(self.config.reconcile_interval_seconds)

    async def _websocket_loop(self) -> None:
        reconnect_delay = 1
        while True:
            try:
                market_tickers = sorted(self.market_registry)
                if not market_tickers:
                    await self.bootstrap()
                    market_tickers = sorted(self.market_registry)
                websocket_headers = self.websocket_headers_factory() if self.websocket_headers_factory else None
                async with websockets.connect(
                    self.config.websocket_url,
                    additional_headers=websocket_headers,
                    proxy=None,
                ) as websocket:
                    await websocket.send(json.dumps(self._build_subscribe_message(market_tickers)))
                    async for raw_message in websocket:
                        payload = json.loads(raw_message)
                        await self.handle_websocket_payload(payload)
                reconnect_delay = 1
            except Exception as exc:
                if "HTTP 401" in str(exc):
                    log.warning("Kalshi websocket unavailable without auth; continuing with REST-only reconciliation.")
                    return
                log.warning("Kalshi websocket disconnected: %s", exc)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30)

    def _build_subscribe_message(self, market_tickers: list[str]) -> dict[str, Any]:
        return {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": self.config.websocket_channels,
                "market_tickers": market_tickers,
            },
        }

    def _get_spot_price(self, observed_at: datetime) -> float:
        if self._spot_cache is not None:
            spot, ts = self._spot_cache
            if observed_at - ts <= timedelta(seconds=self.config.spot_refresh_interval_seconds):
                return spot
        spot = self.coinbase_client.get_spot_price()
        self._spot_cache = (spot, observed_at)
        return spot

    def _filter_and_rank_markets(self, raw_markets: list[dict], observed_at: datetime, spot_price: float) -> list[dict]:
        eligible: list[tuple[tuple[float, datetime], dict[str, Any]]] = []
        min_delta = timedelta(minutes=self.config.min_minutes_to_expiry)
        max_delta = timedelta(minutes=self.config.max_minutes_to_expiry) if self.config.max_minutes_to_expiry is not None else None
        for raw_market in raw_markets:
            expiry = self._parse_market_expiry(raw_market)
            if expiry is None:
                continue
            time_to_expiry = expiry - observed_at
            if time_to_expiry < min_delta:
                continue
            if max_delta is not None and time_to_expiry > max_delta:
                continue
            eligible.append((self._market_rank_key(raw_market, expiry=expiry, spot_price=spot_price), dict(raw_market)))
        eligible.sort(key=lambda item: item[0])
        return [raw for _, raw in eligible[: self.config.market_limit]]

    @staticmethod
    def _market_rank_key(raw_market: dict[str, Any], *, expiry: datetime, spot_price: float) -> tuple[float, datetime]:
        threshold_raw = raw_market.get("threshold") or raw_market.get("strike") or raw_market.get("floor_strike")
        distance = float("inf")
        if threshold_raw is not None:
            try:
                threshold = float(threshold_raw)
                if spot_price > 0:
                    distance = abs(threshold - spot_price) / spot_price
            except (TypeError, ValueError):
                pass
        return (distance, expiry)

    @staticmethod
    def _parse_market_expiry(raw_market: dict[str, Any]) -> datetime | None:
        for key in ("expiry", "expiration_time", "close_time", "settlement_time"):
            value = raw_market.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
        return None
