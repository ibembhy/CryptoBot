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
