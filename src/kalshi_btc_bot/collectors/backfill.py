from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import logging
import pandas as pd

from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.markets.normalize import normalize_market
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _market_priority(raw_market: dict) -> tuple[float, float, str]:
    volume = _safe_float(raw_market.get("volume_fp") or raw_market.get("volume")) or 0.0
    open_interest = _safe_float(raw_market.get("open_interest_fp") or raw_market.get("open_interest")) or 0.0
    liquidity = _safe_float(raw_market.get("liquidity_dollars")) or 0.0
    strike = _safe_float(
        raw_market.get("threshold")
        or raw_market.get("strike")
        or raw_market.get("floor_strike")
        or raw_market.get("cap_strike")
    )
    settlement_reference = _safe_float(
        raw_market.get("expiration_value")
        or raw_market.get("settlement_value")
        or raw_market.get("settlement_reference")
    )
    distance = abs((strike or 0.0) - settlement_reference) if strike is not None and settlement_reference is not None else float("inf")
    ticker = str(raw_market.get("ticker") or "")
    return (-(volume + open_interest + liquidity), distance, ticker)


def _parse_historical_market_expiry(raw_market: dict) -> datetime | None:
    for key in ("expiry", "expiration_time", "close_time", "settlement_time"):
        value = raw_market.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                pass

    ticker = str(raw_market.get("ticker") or "")
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    token = parts[1]
    if len(token) < 9:
        return None
    try:
        parsed = datetime.strptime(token[:9], "%y%b%d%H")
        return parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_api_datetime(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class BackfillConfig:
    series_ticker: str
    start_ts: int
    end_ts: int
    period_interval: int = 1
    include_latest_before_start: bool = True
    batch_size: int = 100
    max_markets: int | None = None
    use_historical_markets: bool = False
    historical_pages: int = 5


log = logging.getLogger(__name__)


class CandlestickBackfillService:
    def __init__(
        self,
        *,
        kalshi_client: KalshiClient,
        coinbase_client: CoinbaseClient,
        snapshot_store: SnapshotStore,
        config: BackfillConfig,
    ) -> None:
        self.kalshi_client = kalshi_client
        self.coinbase_client = coinbase_client
        self.snapshot_store = snapshot_store
        self.config = config
        self._spot_cache: float | None = None
        self._spot_history: pd.Series | None = None
        self._market_cutoff: datetime | None = None

    def backfill(self) -> list[MarketSnapshot]:
        raw_markets = self._discover_markets()
        raw_markets = sorted(raw_markets, key=_market_priority)
        registry = {str(market.get("ticker")): market for market in raw_markets}
        market_tickers = [str(market.get("ticker")) for market in raw_markets if market.get("ticker")]
        if self.config.max_markets is not None:
            market_tickers = market_tickers[: self.config.max_markets]
        written: list[MarketSnapshot] = []
        total_markets = len(market_tickers)
        log.info(
            "Starting backfill for %s markets in series %s from %s to %s",
            total_markets,
            self.config.series_ticker,
            self.config.start_ts,
            self.config.end_ts,
        )
        for index in range(0, len(market_tickers), self.config.batch_size):
            batch = market_tickers[index : index + self.config.batch_size]
            log.info(
                "Backfill batch %s-%s of %s markets",
                index + 1,
                min(index + len(batch), total_markets),
                total_markets,
            )
            if self.config.use_historical_markets:
                for market_ticker in batch:
                    base_market = registry.get(market_ticker)
                    if not base_market:
                        continue
                    route = str(base_market.get("_source_route") or "historical")
                    try:
                        if route == "live":
                            payload = self.kalshi_client.get_market_candlesticks(
                                series_ticker=self.config.series_ticker,
                                market_ticker=market_ticker,
                                start_ts=self.config.start_ts,
                                end_ts=self.config.end_ts,
                                period_interval=self.config.period_interval,
                                include_latest_before_start=self.config.include_latest_before_start,
                            )
                        else:
                            payload = self.kalshi_client.get_historical_market_candlesticks(
                                ticker=market_ticker,
                                start_ts=self.config.start_ts,
                                end_ts=self.config.end_ts,
                                period_interval=self.config.period_interval,
                                include_latest_before_start=self.config.include_latest_before_start,
                            )
                    except requests.HTTPError:
                        log.warning("Skipping %s market %s because candlestick fetch failed.", route, market_ticker)
                        continue
                    for candlestick in payload.get("candlesticks", []):
                        snapshot = self._snapshot_from_candlestick(base_market, candlestick, route=route)
                        if snapshot is None:
                            continue
                        self.snapshot_store.insert_snapshot(snapshot)
                        written.append(snapshot)
                    log.info(
                        "Processed %s market %s (%s snapshots written so far).",
                        route,
                        market_ticker,
                        len(written),
                    )
                continue
            try:
                payload = self.kalshi_client.get_batch_market_candlesticks(
                    market_tickers=batch,
                    start_ts=self.config.start_ts,
                    end_ts=self.config.end_ts,
                    period_interval=self.config.period_interval,
                    include_latest_before_start=self.config.include_latest_before_start,
                )
                written.extend(self._persist_batch_payload(payload, registry))
            except requests.HTTPError:
                log.warning(
                    "Batch candlesticks request failed for markets %s-%s, falling back to one market at a time.",
                    index + 1,
                    min(index + len(batch), total_markets),
                )
                for market_ticker in batch:
                    base_market = registry.get(market_ticker)
                    if not base_market:
                        continue
                    try:
                        payload = self.kalshi_client.get_market_candlesticks(
                            series_ticker=self.config.series_ticker,
                            market_ticker=market_ticker,
                            start_ts=self.config.start_ts,
                            end_ts=self.config.end_ts,
                            period_interval=self.config.period_interval,
                            include_latest_before_start=self.config.include_latest_before_start,
                        )
                    except requests.HTTPError:
                        log.warning("Skipping %s because candlestick fetch failed.", market_ticker)
                        continue
                    for candlestick in payload.get("candlesticks", []):
                        snapshot = self._snapshot_from_candlestick(base_market, candlestick)
                        if snapshot is None:
                            continue
                        self.snapshot_store.insert_snapshot(snapshot)
                        written.append(snapshot)
                    log.info(
                        "Processed market %s (%s snapshots written so far).",
                        market_ticker,
                        len(written),
                    )
        log.info("Backfill completed with %s snapshots written.", len(written))
        return written

    def _discover_markets(self) -> list[dict]:
        if not self.config.use_historical_markets:
            return self.kalshi_client.list_markets(
                series_ticker=self.config.series_ticker,
                status="open",
                limit=1000,
            )

        cutoff = self._load_market_cutoff()
        markets: list[dict] = []
        for event in self._discover_settled_events():
            event_ticker = str(event.get("event_ticker") or event.get("ticker") or "")
            if not event_ticker:
                continue
            event_time = self._event_reference_time(event)
            route = "live" if cutoff is not None and event_time is not None and event_time >= cutoff else "historical"
            for market in self._load_event_markets(event_ticker, route=route):
                ticker = str(market.get("ticker") or "")
                if ticker.startswith(f"{self.config.series_ticker}-"):
                    tagged_market = dict(market)
                    tagged_market["_source_route"] = route
                    markets.append(tagged_market)
        return markets

    def _discover_settled_events(self) -> list[dict]:
        events: list[dict] = []
        cursor: str | None = None
        pages = 0
        start_dt = datetime.fromtimestamp(self.config.start_ts, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(self.config.end_ts, tz=timezone.utc)

        while pages < self.config.historical_pages:
            payload = self.kalshi_client.list_events(
                series_ticker=self.config.series_ticker,
                status="settled",
                limit=200,
                cursor=cursor,
            )
            for event in payload.get("events", []):
                event_time = self._event_reference_time(event)
                if event_time is None:
                    continue
                if start_dt <= event_time <= end_dt:
                    events.append(event)
            cursor = payload.get("cursor")
            pages += 1
            if not cursor:
                break
        return events

    def _load_event_markets(self, event_ticker: str, *, route: str) -> list[dict]:
        if route == "live":
            return self.kalshi_client.list_markets(
                event_ticker=event_ticker,
                status="settled",
                limit=1000,
            )
        historical_markets: list[dict] = []
        cursor: str | None = None
        while True:
            payload = self.kalshi_client.list_historical_markets(
                limit=1000,
                cursor=cursor,
                event_ticker=event_ticker,
            )
            page_markets = payload.get("markets", [])
            historical_markets.extend(page_markets)
            cursor = payload.get("cursor")
            if not cursor:
                break
        return historical_markets

    def _load_market_cutoff(self) -> datetime | None:
        if self._market_cutoff is not None:
            return self._market_cutoff
        payload = self.kalshi_client.get_historical_cutoff()
        raw = payload.get("market_settled_ts")
        parsed = _parse_api_datetime(raw)
        if parsed is None and raw is not None:
            try:
                parsed = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except (TypeError, ValueError):
                parsed = None
        self._market_cutoff = parsed
        return self._market_cutoff

    def _event_reference_time(self, event: dict) -> datetime | None:
        for key in (
            "settlement_ts",
            "settlement_time",
            "close_time",
            "expiration_time",
            "strike_date",
            "end_date",
        ):
            parsed = _parse_api_datetime(event.get(key))
            if parsed is not None:
                return parsed
        return None

    def _persist_batch_payload(self, payload: dict, registry: dict[str, dict]) -> list[MarketSnapshot]:
        written: list[MarketSnapshot] = []
        for market_payload in payload.get("markets", []):
            market_ticker = market_payload.get("market_ticker")
            base_market = registry.get(str(market_ticker))
            if not base_market:
                continue
            for candlestick in market_payload.get("candlesticks", []):
                snapshot = self._snapshot_from_candlestick(base_market, candlestick)
                if snapshot is None:
                    continue
                self.snapshot_store.insert_snapshot(snapshot)
                written.append(snapshot)
        return written

    def _snapshot_from_candlestick(self, base_market: dict, candlestick: dict, *, route: str | None = None) -> MarketSnapshot | None:
        end_ts = candlestick.get("end_period_ts")
        if end_ts is None:
            return None
        observed_at = datetime.fromtimestamp(int(end_ts), tz=timezone.utc)
        spot = self._get_spot_price(observed_at)
        raw_market = dict(base_market)
        raw_market.update(
            {
                "yes_bid_dollars": candlestick.get("yes_bid", {}).get("close_dollars"),
                "yes_ask_dollars": candlestick.get("yes_ask", {}).get("close_dollars"),
                "volume": candlestick.get("volume_fp"),
                "open_interest": candlestick.get("open_interest_fp"),
            }
        )
        snapshot = normalize_market(
            raw_market,
            spot_price=spot,
            observed_at=observed_at,
            series_ticker_override=self.config.series_ticker,
        )
        snapshot.metadata.update(
            {
                "candle_price_close": _safe_float(candlestick.get("price", {}).get("close_dollars")),
                "candle_price_mean": _safe_float(candlestick.get("price", {}).get("mean_dollars")),
                "backfilled": True,
                "period_interval": self.config.period_interval,
                "source_route": route or str(base_market.get("_source_route") or "live"),
            }
        )
        return snapshot

    def _get_spot_price(self, observed_at: datetime) -> float:
        spot_history = self._load_spot_history()
        if spot_history is not None and not spot_history.empty:
            ts = pd.Timestamp(observed_at)
            eligible = spot_history.loc[spot_history.index <= ts]
            if not eligible.empty:
                return float(eligible.iloc[-1])

        if self._spot_cache is None:
            self._spot_cache = self.coinbase_client.get_spot_price()
        return self._spot_cache

    def _load_spot_history(self) -> pd.Series | None:
        if self._spot_history is not None:
            return self._spot_history

        start = datetime.fromtimestamp(self.config.start_ts, tz=timezone.utc)
        end = datetime.fromtimestamp(self.config.end_ts, tz=timezone.utc)
        try:
            candles = self.coinbase_client.fetch_candles(start=start, end=end, timeframe="1m")
        except Exception:
            self._spot_history = pd.Series(dtype=float)
            return self._spot_history

        if candles.empty:
            self._spot_history = pd.Series(dtype=float)
            return self._spot_history

        closes = candles["close"].astype(float).sort_index()
        self._spot_history = closes
        return self._spot_history
