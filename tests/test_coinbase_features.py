from datetime import datetime, timezone
import unittest

import pandas as pd

from kalshi_btc_bot.data.candles import resample_candles
from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.data.features import build_feature_frame


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *args, **kwargs):
        return FakeResponse(self.payload)


class CoinbaseFeatureTests(unittest.TestCase):
    def test_coinbase_candle_fetch_normalizes_to_utc_index(self):
        payload = [
            [1711929720, 69000, 69200, 69100, 69150, 12.0],
            [1711929660, 68900, 69100, 69000, 69050, 10.0],
        ]
        client = CoinbaseClient()
        frame = client.fetch_candles(
            datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 4, 1, 0, 2, tzinfo=timezone.utc),
            "1m",
            session=FakeSession(payload),
        )
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(str(frame.index.tz), "UTC")
        self.assertTrue(frame.index.is_monotonic_increasing)

    def test_get_spot_price_falls_back_to_coinbase_when_binance_fails(self):
        class TestClient(CoinbaseClient):
            def _get_spot_price_binance(self) -> float:
                raise RuntimeError("binance unavailable")

            def _get_spot_price_coinbase(self, *, session=None) -> float:
                return 69000.0

        client = TestClient(provider_order=("binance", "coinbase"))
        self.assertEqual(client.get_spot_price(), 69000.0)

    def test_get_spot_price_uses_kraken_before_coinbase(self):
        class TestClient(CoinbaseClient):
            def _get_spot_price_binance(self) -> float:
                raise RuntimeError("binance unavailable")

            def _get_spot_price_kraken(self) -> float:
                return 68950.0

            def _get_spot_price_coinbase(self, *, session=None) -> float:
                raise AssertionError("coinbase fallback should not be used")

        client = TestClient(provider_order=("binance", "kraken", "coinbase"))
        self.assertEqual(client.get_spot_price(), 68950.0)

    def test_fetch_candles_uses_kraken_before_coinbase(self):
        index = pd.date_range("2024-04-01T00:00:00Z", periods=2, freq="1min")
        kraken_frame = pd.DataFrame(
            {
                "open": [69000.0, 69010.0],
                "high": [69020.0, 69030.0],
                "low": [68990.0, 69000.0],
                "close": [69010.0, 69020.0],
                "volume": [1.0, 2.0],
            },
            index=index,
        )

        class TestClient(CoinbaseClient):
            def _fetch_candles_binance(self, start, end, timeframe):
                raise RuntimeError("binance unavailable")

            def _fetch_candles_kraken(self, start, end, timeframe):
                return kraken_frame

            def _fetch_candles_coinbase(self, start, end, timeframe, *, session=None):
                raise AssertionError("coinbase fallback should not be used")

        client = TestClient(provider_order=("binance", "kraken", "coinbase"))
        frame = client.fetch_candles(
            datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 4, 1, 0, 2, tzinfo=timezone.utc),
            "1m",
        )
        self.assertEqual(len(frame), 2)

    def test_extract_kraken_result_value_ignores_last_key(self):
        value = CoinbaseClient._extract_kraken_result_value(
            {
                "XXBTZUSD": {"c": ["69000.1", "1"]},
                "last": "1711929720",
            }
        )
        self.assertEqual(value, {"c": ["69000.1", "1"]})

    def test_fetch_candles_uses_coinbase_wire_format_when_session_is_injected(self):
        payload = [
            [1711929720, 69000, 69200, 69100, 69150, 12.0],
            [1711929660, 68900, 69100, 69000, 69050, 10.0],
        ]
        client = CoinbaseClient(provider_order=("binance", "coinbase"))
        frame = client.fetch_candles(
            datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 4, 1, 0, 2, tzinfo=timezone.utc),
            "1m",
            session=FakeSession(payload),
        )
        self.assertEqual(len(frame), 2)

    def test_resample_and_features_avoid_lookahead(self):
        index = pd.date_range("2024-04-01T00:00:00Z", periods=6, freq="1min")
        candles = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105],
                "high": [101, 102, 103, 104, 105, 106],
                "low": [99, 100, 101, 102, 103, 104],
                "close": [100, 101, 102, 103, 104, 105],
                "volume": [1, 1, 1, 1, 1, 1],
            },
            index=index,
        )
        resampled = resample_candles(candles, "5min")
        features = build_feature_frame(candles, volatility_window=3, annualization_factor=365.0 * 24 * 60)
        self.assertEqual(len(resampled), 2)
        self.assertTrue(pd.isna(features.iloc[1]["realized_volatility"]))
        self.assertFalse(pd.isna(features.iloc[-1]["realized_volatility"]))


if __name__ == "__main__":
    unittest.main()
