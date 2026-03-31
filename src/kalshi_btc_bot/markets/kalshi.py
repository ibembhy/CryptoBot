from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from kalshi_btc_bot.markets.normalize import normalize_market
from kalshi_btc_bot.types import MarketSnapshot


@dataclass
class KalshiClient:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    timeout: int = 10

    def list_markets(
        self,
        *,
        series_ticker: str = "KXBTCD",
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
        cursor: str | None = None,
        session: requests.Session | None = None,
    ) -> list[dict]:
        http = session or self._build_session()
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        response = http.get(
            f"{self.base_url}/markets",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        markets = payload.get("markets", payload)
        return list(markets)

    def list_events(
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        params: dict[str, Any] = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        response = http.get(f"{self.base_url}/events", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def normalized_snapshots(
        self,
        *,
        spot_price: float,
        observed_at: datetime,
        series_ticker: str = "KXBTCD",
        session: requests.Session | None = None,
    ) -> list[MarketSnapshot]:
        return [
            normalize_market(
                raw_market,
                spot_price=spot_price,
                observed_at=observed_at,
                source="kalshi",
                series_ticker_override=series_ticker,
            )
            for raw_market in self.list_markets(series_ticker=series_ticker, session=session)
        ]

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        include_latest_before_start: bool = False,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(
            f"{self.base_url}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
                "include_latest_before_start": str(include_latest_before_start).lower(),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_batch_market_candlesticks(
        self,
        *,
        market_tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        include_latest_before_start: bool = False,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(
            f"{self.base_url}/markets/candlesticks",
            params={
                "market_tickers": ",".join(market_tickers),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
                "include_latest_before_start": str(include_latest_before_start).lower(),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_historical_cutoff(self, session: requests.Session | None = None) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(f"{self.base_url}/historical/cutoff", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_market(self, ticker: str, session: requests.Session | None = None) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(f"{self.base_url}/markets/{ticker}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_historical_market(self, ticker: str, session: requests.Session | None = None) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(f"{self.base_url}/historical/markets/{ticker}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_historical_markets(
        self,
        *,
        limit: int = 1000,
        cursor: str | None = None,
        tickers: list[str] | None = None,
        event_ticker: str | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if tickers:
            params["tickers"] = ",".join(tickers)
        if event_ticker:
            params["event_ticker"] = event_ticker
        response = http.get(f"{self.base_url}/historical/markets", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_historical_market_candlesticks(
        self,
        *,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        include_latest_before_start: bool = False,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        response = http.get(
            f"{self.base_url}/historical/markets/{ticker}/candlesticks",
            params={
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
                "include_latest_before_start": str(include_latest_before_start).lower(),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session
