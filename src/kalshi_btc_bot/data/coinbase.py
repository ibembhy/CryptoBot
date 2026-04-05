from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
import logging
import warnings

import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning

from kalshi_btc_bot.utils.time import ensure_utc


GRANULARITY_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}

BINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
}

KRAKEN_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}

log = logging.getLogger(__name__)


@dataclass
class CoinbaseClient:
    product_id: str = "BTC-USD"
    base_url: str = "https://api.exchange.coinbase.com"
    binance_symbol: str = "BTCUSDT"
    binance_base_url: str = "https://api.binance.com"
    kraken_pair: str = "XBTUSD"
    kraken_base_url: str = "https://api.kraken.com"
    provider_order: tuple[str, ...] = ("coinbase", "kraken")
    timeout: int = 10
    verify_ssl: bool = True

    def fetch_candles(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
        session: requests.Session | None = None,
    ) -> pd.DataFrame:
        if timeframe not in GRANULARITY_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        provider_order = self._provider_order_for_request(session)
        last_error: Exception | None = None
        for index, provider in enumerate(provider_order):
            try:
                if provider == "binance":
                    frame = self._fetch_candles_binance(start, end, timeframe)
                elif provider == "kraken":
                    frame = self._fetch_candles_kraken(start, end, timeframe)
                elif provider == "coinbase":
                    frame = self._fetch_candles_coinbase(start, end, timeframe, session=session if provider_order == ("coinbase",) else None)
                else:
                    raise ValueError(f"Unsupported market data provider: {provider}")
                if frame.empty:
                    raise RuntimeError(f"{provider} returned no candle data")
                if index > 0:
                    log.warning("Primary candle provider unavailable; using fallback provider %s.", provider)
                return frame
            except Exception as exc:
                last_error = exc
                if index < len(provider_order) - 1:
                    log.warning("Market data provider %s failed for candles: %s", provider, exc)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No market data provider available for candles")

    def get_spot_price(self, session: requests.Session | None = None) -> float:
        provider_order = self._provider_order_for_request(session)
        last_error: Exception | None = None
        for index, provider in enumerate(provider_order):
            try:
                if provider == "binance":
                    price = self._get_spot_price_binance()
                elif provider == "kraken":
                    price = self._get_spot_price_kraken()
                elif provider == "coinbase":
                    price = self._get_spot_price_coinbase(session=session if provider_order == ("coinbase",) else None)
                else:
                    raise ValueError(f"Unsupported market data provider: {provider}")
                if index > 0:
                    log.warning("Primary spot provider unavailable; using fallback provider %s.", provider)
                return price
            except Exception as exc:
                last_error = exc
                if index < len(provider_order) - 1:
                    log.warning("Market data provider %s failed for spot price: %s", provider, exc)
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No market data provider available for spot price")

    def stream_url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"

    def _provider_order_for_request(self, session: requests.Session | None) -> tuple[str, ...]:
        if session is not None:
            return ("coinbase",)
        ordered: list[str] = []
        for provider in self.provider_order:
            normalized = str(provider).strip().lower()
            if normalized and normalized not in ordered:
                ordered.append(normalized)
        return tuple(ordered) or ("coinbase",)

    def _fetch_candles_coinbase(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
        *,
        session: requests.Session | None = None,
    ) -> pd.DataFrame:
        granularity = GRANULARITY_SECONDS[timeframe]
        http = session or self._build_session()
        frames: list[pd.DataFrame] = []
        step = granularity * 300
        cursor = ensure_utc(start)
        end_ts = ensure_utc(end)
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            while cursor < end_ts:
                chunk_end = min(cursor + pd.Timedelta(seconds=step), end_ts)
                response = http.get(
                    f"{self.base_url}/products/{self.product_id}/candles",
                    params={
                        "start": cursor.isoformat(),
                        "end": chunk_end.isoformat(),
                        "granularity": granularity,
                    },
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                payload = response.json()
                if payload:
                    frame = pd.DataFrame(
                        payload,
                        columns=["timestamp", "low", "high", "open", "close", "volume"],
                    )
                    frames.append(frame)
                cursor = chunk_end
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True)
        merged = merged.sort_values("timestamp").set_index("timestamp")
        return merged[["open", "high", "low", "close", "volume"]].astype(float)

    def _fetch_candles_binance(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        interval = BINANCE_INTERVALS.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported Binance timeframe: {timeframe}")
        http = self._build_session()
        frames: list[pd.DataFrame] = []
        step = GRANULARITY_SECONDS[timeframe] * 1000
        cursor = ensure_utc(start)
        end_ts = ensure_utc(end)
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            while cursor < end_ts:
                chunk_end = min(cursor + pd.Timedelta(seconds=step), end_ts)
                response = http.get(
                    f"{self.binance_base_url}/api/v3/klines",
                    params={
                        "symbol": self.binance_symbol,
                        "interval": interval,
                        "startTime": int(cursor.timestamp() * 1000),
                        "endTime": max(int(chunk_end.timestamp() * 1000) - 1, int(cursor.timestamp() * 1000)),
                        "limit": 1000,
                    },
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                payload = response.json()
                if payload:
                    frame = pd.DataFrame(
                        payload,
                        columns=[
                            "open_time",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "close_time",
                            "quote_asset_volume",
                            "number_of_trades",
                            "taker_buy_base_volume",
                            "taker_buy_quote_volume",
                            "ignore",
                        ],
                    )
                    frames.append(frame[["open_time", "open", "high", "low", "close", "volume"]])
                cursor = chunk_end
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["open_time"])
        merged["timestamp"] = pd.to_datetime(merged["open_time"], unit="ms", utc=True)
        merged = merged.drop(columns=["open_time"]).sort_values("timestamp").set_index("timestamp")
        return merged[["open", "high", "low", "close", "volume"]].astype(float)

    def _fetch_candles_kraken(self, start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
        interval = KRAKEN_INTERVAL_MINUTES.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported Kraken timeframe: {timeframe}")
        http = self._build_session()
        cursor = ensure_utc(start)
        end_ts = ensure_utc(end)
        frames: list[pd.DataFrame] = []
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            while cursor < end_ts:
                response = http.get(
                    f"{self.kraken_base_url}/0/public/OHLC",
                    params={
                        "pair": self.kraken_pair,
                        "interval": interval,
                        "since": int(cursor.timestamp()),
                    },
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()
                payload = response.json()
                errors = payload.get("error", [])
                if errors:
                    raise RuntimeError(f"Kraken candle error: {errors}")
                result = payload.get("result", {})
                candles = self._extract_kraken_result_value(result)
                if candles:
                    frame = pd.DataFrame(
                        candles,
                        columns=[
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "vwap",
                            "volume",
                            "count",
                        ],
                    )
                    frames.append(frame[["timestamp", "open", "high", "low", "close", "volume"]])
                last = result.get("last")
                if not candles:
                    break
                next_cursor = datetime.fromtimestamp(max(int(last), int(cursor.timestamp()) + 1), tz=timezone.utc)
                if next_cursor <= cursor:
                    break
                cursor = next_cursor
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], unit="s", utc=True)
        merged = merged.sort_values("timestamp").set_index("timestamp")
        return merged[["open", "high", "low", "close", "volume"]].astype(float)

    def _get_spot_price_coinbase(self, *, session: requests.Session | None = None) -> float:
        http = session or self._build_session()
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            response = http.get(
                f"{self.base_url}/products/{self.product_id}/ticker",
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        response.raise_for_status()
        return float(response.json()["price"])

    def _get_spot_price_binance(self) -> float:
        http = self._build_session()
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            response = http.get(
                f"{self.binance_base_url}/api/v3/ticker/price",
                params={"symbol": self.binance_symbol},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        response.raise_for_status()
        return float(response.json()["price"])

    def _get_spot_price_kraken(self) -> float:
        http = self._build_session()
        with warnings.catch_warnings():
            if not self.verify_ssl:
                warnings.simplefilter("ignore", InsecureRequestWarning)
            response = http.get(
                f"{self.kraken_base_url}/0/public/Ticker",
                params={"pair": self.kraken_pair},
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("error", [])
        if errors:
            raise RuntimeError(f"Kraken spot error: {errors}")
        result = payload.get("result", {})
        ticker = self._extract_kraken_result_value(result)
        if not ticker:
            raise RuntimeError(f"Kraken returned no ticker for {self.kraken_pair}")
        close_values = ticker.get("c")
        if not close_values:
            raise RuntimeError(f"Kraken ticker missing close price for {self.kraken_pair}")
        return float(close_values[0])

    @staticmethod
    def _extract_kraken_result_value(result: dict) -> object:
        if not isinstance(result, dict):
            return None
        for key, value in result.items():
            if key == "last":
                continue
            return value
        return None

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        # Ignore machine-level proxy variables so local dev proxy misconfigurations
        # do not break public Coinbase market-data calls.
        session.trust_env = False
        return session


def default_history_window(hours: int = 24) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    return start, end
