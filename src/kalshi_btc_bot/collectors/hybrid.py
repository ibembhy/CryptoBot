from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

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
    series_tickers: list[str] | None = None
    status: str = "open"
    market_limit: int = 200
    min_minutes_to_expiry: int = 0
    max_minutes_to_expiry: int | None = 180
    reconcile_interval_seconds: int = 60
    spot_refresh_interval_seconds: int = 3
    event_discovery_cache_seconds: int = 10
    websocket_channels: list[str] = field(default_factory=lambda: ["ticker"])
    websocket_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    persist_snapshots: bool = True


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
        self._event_ticker_cache: dict[str, tuple[list[str], datetime]] = {}
        self._websocket: Any | None = None

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
            )
            for raw in raw_markets
        ]
        new_market_registry = {snapshot.market_ticker: dict(raw) for raw, snapshot in zip(raw_markets, snapshots, strict=False)}
        registry_changed = set(new_market_registry) != set(self.market_registry)
        self.market_registry = new_market_registry
        if self.config.persist_snapshots:
            for raw, snapshot in zip(raw_markets, snapshots, strict=False):
                self.snapshot_store.insert_snapshot(snapshot)
        if registry_changed:
            await self._refresh_live_subscription()
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
        )
        if self.config.persist_snapshots:
            self.snapshot_store.insert_snapshot(snapshot)
        return snapshot

    def _discover_markets(self, *, observed_at: datetime, spot_price: float) -> list[dict[str, Any]]:
        targeted_markets = self._discover_event_markets(observed_at=observed_at)
        if targeted_markets:
            return self._filter_and_rank_markets(targeted_markets, observed_at, spot_price)

        raw_markets: list[dict[str, Any]] = []
        max_pages = 10
        target_pool_size = max(self.config.market_limit * 3, self.config.market_limit)
        for series_ticker in self._series_tickers():
            cursor: str | None = None
            for _ in range(max_pages):
                page = self.kalshi_client.list_markets_page(
                    series_ticker=series_ticker,
                    status=self.config.status,
                    limit=self.config.market_limit,
                    cursor=cursor,
                )
                page_markets = [
                    {
                        **dict(raw_market),
                        "series_ticker": raw_market.get("series_ticker") or series_ticker,
                    }
                    for raw_market in page.get("markets", [])
                ]
                if not page_markets:
                    break
                raw_markets.extend(page_markets)
                cursor = page.get("cursor")
                if len(raw_markets) >= target_pool_size or not cursor:
                    break
            if len(raw_markets) >= target_pool_size:
                break
        return self._filter_and_rank_markets(raw_markets, observed_at, spot_price)

    def _discover_event_markets(self, *, observed_at: datetime) -> list[dict[str, Any]]:
        raw_markets: list[dict[str, Any]] = []
        for series_ticker in self._series_tickers():
            event_tickers = self._discover_target_event_tickers(series_ticker=series_ticker, observed_at=observed_at)
            for event_ticker in event_tickers:
                cursor: str | None = None
                for _ in range(10):
                    page = self.kalshi_client.list_markets_page(
                        series_ticker=series_ticker,
                        event_ticker=event_ticker,
                        status=self.config.status,
                        limit=self.config.market_limit,
                        cursor=cursor,
                    )
                    page_markets = [
                        {
                            **dict(raw_market),
                            "series_ticker": raw_market.get("series_ticker") or series_ticker,
                        }
                        for raw_market in page.get("markets", [])
                    ]
                    if not page_markets:
                        break
                    raw_markets.extend(page_markets)
                    cursor = page.get("cursor")
                    if not cursor:
                        break
        deduped: list[dict[str, Any]] = []
        seen_tickers: set[str] = set()
        for raw_market in raw_markets:
            ticker = str(raw_market.get("ticker") or "")
            if ticker and ticker in seen_tickers:
                continue
            if ticker:
                seen_tickers.add(ticker)
            deduped.append(raw_market)
        return deduped

    def _discover_target_event_tickers(self, *, series_ticker: str, observed_at: datetime) -> list[str]:
        cached = self._event_ticker_cache.get(series_ticker)
        cache_ttl = timedelta(seconds=self.config.event_discovery_cache_seconds)
        if cached is not None:
            cached_tickers, cached_at = cached
            if observed_at - cached_at <= cache_ttl:
                return list(cached_tickers)

        heuristic_tickers = self._heuristic_target_event_tickers(series_ticker=series_ticker, observed_at=observed_at)
        if heuristic_tickers:
            self._event_ticker_cache[series_ticker] = (list(heuristic_tickers), observed_at)
            return heuristic_tickers

        try:
            payload = self.kalshi_client.list_events(series_ticker=series_ticker, status=self.config.status, limit=200)
        except Exception as exc:
            log.warning("Event discovery failed for %s: %s", series_ticker, exc)
            if cached is not None:
                cached_tickers, cached_at = cached
                if observed_at - cached_at <= max(cache_ttl * 3, timedelta(seconds=30)):
                    return list(cached_tickers)
            return []

        events = payload.get("events", [])
        if not isinstance(events, list):
            return []

        min_delta = timedelta(minutes=self.config.min_minutes_to_expiry)
        max_delta = timedelta(minutes=self.config.max_minutes_to_expiry) if self.config.max_minutes_to_expiry is not None else None
        targeted: list[tuple[datetime, str]] = []
        for raw_event in events:
            if not isinstance(raw_event, dict):
                continue
            event_ticker = str(raw_event.get("event_ticker") or raw_event.get("ticker") or "")
            event_close = self._parse_event_close(raw_event)
            if not event_ticker or event_close is None:
                continue
            time_to_expiry = event_close - observed_at
            if time_to_expiry < min_delta:
                continue
            if max_delta is not None and time_to_expiry > max_delta:
                continue
            targeted.append((event_close, event_ticker))

        targeted.sort(key=lambda item: item[0])
        event_tickers = [ticker for _, ticker in targeted]
        self._event_ticker_cache[series_ticker] = (list(event_tickers), observed_at)
        return event_tickers

    def _heuristic_target_event_tickers(self, *, series_ticker: str, observed_at: datetime) -> list[str]:
        normalized = str(series_ticker or "").upper()
        if normalized == "KXBTCD":
            interval_minutes = 60
            stamp_format = "%y%b%d%H"
        elif normalized == "KXBTC15M":
            interval_minutes = 15
            stamp_format = "%y%b%d%H%M"
        else:
            return []

        observed_local = observed_at.astimezone(ZoneInfo("America/New_York"))
        boundary = self._next_event_boundary(observed_local, interval_minutes=interval_minutes)
        min_delta = timedelta(minutes=self.config.min_minutes_to_expiry)
        max_delta = timedelta(minutes=self.config.max_minutes_to_expiry) if self.config.max_minutes_to_expiry is not None else None
        targeted: list[str] = []
        for _ in range(8):
            boundary_utc = boundary.astimezone(timezone.utc)
            time_to_expiry = boundary_utc - observed_at
            if time_to_expiry >= min_delta and (max_delta is None or time_to_expiry <= max_delta):
                targeted.append(f"{normalized}-{boundary.strftime(stamp_format).upper()}")
            if max_delta is not None and time_to_expiry > max_delta:
                break
            boundary = boundary + timedelta(minutes=interval_minutes)
        return targeted

    @staticmethod
    def _next_event_boundary(observed_local: datetime, *, interval_minutes: int) -> datetime:
        if (
            observed_local.second == 0
            and observed_local.microsecond == 0
            and observed_local.minute % interval_minutes == 0
        ):
            return observed_local
        base = observed_local.replace(second=0, microsecond=0)
        next_minute = ((base.minute // interval_minutes) + 1) * interval_minutes
        if next_minute >= 60:
            base = (base + timedelta(hours=1)).replace(minute=0)
        else:
            base = base.replace(minute=next_minute)
        return base

    def _series_tickers(self) -> list[str]:
        configured = list(self.config.series_tickers or [])
        if self.config.series_ticker and self.config.series_ticker not in configured:
            configured.insert(0, self.config.series_ticker)
        return [item for item in configured if item]

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
                    self._websocket = websocket
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
            finally:
                self._websocket = None

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

    async def _refresh_live_subscription(self) -> None:
        websocket = self._websocket
        if websocket is None:
            return
        if getattr(websocket, "closed", False):
            return
        await websocket.close()

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
        for key in ("close_time", "expected_expiration_time", "expiry", "settlement_time", "expiration_time"):
            value = raw_market.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_event_close(raw_event: dict[str, Any]) -> datetime | None:
        for key in ("close_time", "expected_expiration_time", "settlement_time", "expiration_time"):
            value = raw_event.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue

        event_ticker = str(raw_event.get("event_ticker") or raw_event.get("ticker") or "")
        if "-" not in event_ticker:
            return None
        stamp = event_ticker.split("-", 1)[1]
        fmt = None
        if len(stamp) == 9:
            fmt = "%y%b%d%H"
        elif len(stamp) == 11:
            fmt = "%y%b%d%H%M"
        if fmt is None:
            return None
        try:
            local_dt = datetime.strptime(stamp, fmt).replace(tzinfo=ZoneInfo("America/New_York"))
        except ValueError:
            return None
        return local_dt.astimezone(timezone.utc)
