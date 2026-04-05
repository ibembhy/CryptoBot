from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any

import requests

from kalshi_btc_bot.markets.auth import KalshiAuthSigner
from kalshi_btc_bot.markets.normalize import normalize_market
from kalshi_btc_bot.types import MarketSnapshot


@dataclass
class KalshiClient:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    timeout: int = 10
    auth_signer: KalshiAuthSigner | None = None
    rate_limit_retries: int = 2
    rate_limit_backoff_seconds: float = 0.5

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
        payload = self.list_markets_page(
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            status=status,
            limit=limit,
            cursor=cursor,
            session=session,
        )
        markets = payload.get("markets", payload)
        return list(markets)

    def list_markets_page(
        self,
        *,
        series_ticker: str = "KXBTCD",
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
        cursor: str | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        http = session or self._build_session()
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor
        return self._request_public_json_with_retry(
            f"{self.base_url}/markets",
            params=params,
            session=http,
        )

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
        return self._request_public_json_with_retry(
            f"{self.base_url}/events",
            params=params,
            session=http,
        )

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

    def get_balance(self, session: requests.Session | None = None) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance", session=session).json()

    def list_orders(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        status: str | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self._request("GET", "/portfolio/orders", params=params, session=session).json()

    def get_order(self, order_id: str, session: requests.Session | None = None) -> dict[str, Any]:
        return self._request("GET", f"/portfolio/orders/{order_id}", session=session).json()

    def create_order(self, payload: dict[str, Any], session: requests.Session | None = None) -> dict[str, Any]:
        return self._request("POST", "/portfolio/orders", json=payload, session=session).json()

    def cancel_order(self, order_id: str, session: requests.Session | None = None) -> dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}", session=session).json()

    def get_positions(self, session: requests.Session | None = None) -> dict[str, Any]:
        return self._request("GET", "/portfolio/positions", session=session).json()

    def get_fills(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/portfolio/fills", params=params, session=session).json()

    def _request_public_json_with_retry(
        self,
        url: str,
        *,
        params: dict[str, Any],
        session: requests.Session,
    ) -> dict[str, Any]:
        response: requests.Response | None = None
        for attempt in range(self.rate_limit_retries + 1):
            response = session.get(url, params=params, timeout=self.timeout)
            if response.status_code != 429 or attempt >= self.rate_limit_retries:
                response.raise_for_status()
                return response.json()
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header is not None:
                try:
                    sleep_seconds = max(float(retry_after_header), 0.0)
                except ValueError:
                    sleep_seconds = 0.0
            else:
                sleep_seconds = self.rate_limit_backoff_seconds * (2**attempt)
            time.sleep(sleep_seconds)
        assert response is not None
        response.raise_for_status()
        return response.json()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        session: requests.Session | None = None,
    ) -> requests.Response:
        http = session or self._build_session()
        headers: dict[str, str] = {}
        if self.auth_signer is not None:
            headers.update(self.auth_signer.request_headers(method=method, path=path))
        response = http.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            params=params,
            json=json,
            headers=headers,
            timeout=self.timeout,
        )
        if not response.ok:
            try:
                body = response.text[:500]
            except Exception:
                body = "<unreadable>"
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for url: {response.url} | body: {body}",
                response=response,
            )
        return response

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        return session
