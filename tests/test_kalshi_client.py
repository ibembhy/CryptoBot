from __future__ import annotations

import unittest

import requests

from kalshi_btc_bot.markets.kalshi import KalshiClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, *, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.reason = "Too Many Requests" if status_code == 429 else "OK"
        self.url = "https://example.test/markets"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self) -> dict:
        return dict(self._payload)


class _SequencedSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(self, url: str, params=None, timeout: int | None = None):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


class KalshiClientTests(unittest.TestCase):
    def test_list_markets_page_retries_after_rate_limit(self):
        session = _SequencedSession(
            [
                _FakeResponse(429, {}, headers={"Retry-After": "0"}),
                _FakeResponse(200, {"markets": [{"ticker": "KXBTC15M-TEST"}], "cursor": None}),
            ]
        )
        client = KalshiClient(rate_limit_retries=2, rate_limit_backoff_seconds=0.0)

        payload = client.list_markets_page(series_ticker="KXBTC15M", session=session)

        self.assertEqual(payload["markets"][0]["ticker"], "KXBTC15M-TEST")
        self.assertEqual(session.calls, 2)


if __name__ == "__main__":
    unittest.main()
