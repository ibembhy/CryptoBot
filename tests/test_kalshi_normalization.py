from datetime import datetime, timezone
import unittest

from kalshi_btc_bot.markets.normalize import normalize_market


class KalshiNormalizationTests(unittest.TestCase):
    def test_threshold_market_normalization(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTC",
                "ticker": "KXBTC-65000",
                "contract_type": "threshold",
                "expiry": "2026-03-30T16:00:00Z",
                "threshold": 65000,
                "direction": "above",
                "yes_ask": 0.43,
                "yes_bid": 0.41,
            },
            spot_price=64000,
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.contract_type, "threshold")
        self.assertEqual(snapshot.threshold, 65000)
        self.assertEqual(snapshot.implied_probability, 0.43)

    def test_range_market_normalization(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTC",
                "ticker": "KXBTC-RANGE",
                "contract_type": "range",
                "expiry": "2026-03-30T16:00:00Z",
                "floor": 63000,
                "cap": 65000,
                "yes_ask": 44,
                "yes_bid": 42,
            },
            spot_price=64000,
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.range_low, 63000)
        self.assertEqual(snapshot.range_high, 65000)
        self.assertEqual(snapshot.yes_ask, 0.44)

    def test_finalized_market_prefers_close_time_and_maps_settlement(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-26MAR3020-T75799.99",
                "contract_type": "threshold",
                "close_time": "2026-03-31T00:00:00Z",
                "expiration_time": "2026-04-07T00:00:00Z",
                "threshold": 75799.99,
                "direction": "above",
                "result": "no",
                "settlement_value_dollars": "0.0000",
            },
            spot_price=75000,
            observed_at=datetime(2026, 3, 30, 23, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.expiry, datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(snapshot.settlement_price, 0.0)

    def test_threshold_direction_can_be_inferred_from_subtitle(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-26MAR3012-T67599.99",
                "contract_type": "threshold",
                "close_time": "2026-03-30T16:00:00Z",
                "threshold": 67599.99,
                "subtitle": "$67,600 or above",
                "yes_sub_title": "$67,600 or above",
            },
            spot_price=67000,
            observed_at=datetime(2026, 3, 30, 15, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.direction, "above")

    def test_active_market_prefers_close_time_over_later_expiration_time(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-26MAR3117-T68999.99",
                "contract_type": "threshold",
                "close_time": "2026-03-31T21:00:00Z",
                "expected_expiration_time": "2026-03-31T21:05:00Z",
                "expiration_time": "2026-04-07T21:00:00Z",
                "subtitle": "$69,000 or above",
                "yes_sub_title": "$69,000 or above",
            },
            spot_price=68500,
            observed_at=datetime(2026, 3, 31, 20, 45, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.expiry, datetime(2026, 3, 31, 21, 0, tzinfo=timezone.utc))

    def test_kxbtc15m_market_infers_above_direction_from_up_title(self):
        snapshot = normalize_market(
            {
                "series_ticker": "KXBTC15M",
                "ticker": "KXBTC15M-26APR011645-45",
                "close_time": "2026-04-01T20:45:00Z",
                "expected_expiration_time": "2026-04-01T20:50:00Z",
                "floor_strike": 68051.43,
                "title": "BTC price up in next 15 mins?",
                "yes_sub_title": "Target Price: $68,051.43",
                "yes_ask": 0.54,
                "yes_bid": 0.52,
            },
            spot_price=68058.14,
            observed_at=datetime(2026, 4, 1, 20, 31, tzinfo=timezone.utc),
        )
        self.assertEqual(snapshot.contract_type, "threshold")
        self.assertEqual(snapshot.threshold, 68051.43)
        self.assertEqual(snapshot.direction, "above")
        self.assertEqual(snapshot.expiry, datetime(2026, 4, 1, 20, 45, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
