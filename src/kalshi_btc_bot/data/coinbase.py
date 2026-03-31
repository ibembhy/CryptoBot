from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
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


@dataclass
class CoinbaseClient:
    product_id: str = "BTC-USD"
    base_url: str = "https://api.exchange.coinbase.com"
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

    def get_spot_price(self, session: requests.Session | None = None) -> float:
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

    def stream_url(self) -> str:
        return "wss://ws-feed.exchange.coinbase.com"

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
