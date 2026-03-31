from pathlib import Path
import unittest
from datetime import datetime, timezone

import pandas as pd

from kalshi_btc_bot.collectors.backfill import BackfillConfig, CandlestickBackfillService
from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.storage.snapshots import SnapshotStore


class FakeCoinbaseClient(CoinbaseClient):
    def __init__(self, price: float):
        super().__init__()
        self.price = price

    def fetch_candles(self, start, end, timeframe, session=None):
        index = pd.to_datetime(
            [
                datetime.fromtimestamp(1774882740, tz=timezone.utc),
                datetime.fromtimestamp(1774882800, tz=timezone.utc),
            ],
            utc=True,
        )
        return pd.DataFrame(
            {
                "open": [self.price - 5, self.price],
                "high": [self.price, self.price + 5],
                "low": [self.price - 10, self.price - 5],
                "close": [self.price - 1, self.price + 1],
                "volume": [1.0, 1.0],
            },
            index=index,
        )

    def get_spot_price(self, session=None) -> float:
        return self.price


class FakeKalshiBackfillClient(KalshiClient):
    def __init__(self, batch_raises: bool = False):
        super().__init__()
        self.batch_raises = batch_raises

    def list_markets(self, **kwargs):
        if kwargs.get("event_ticker") == "KXBTCD-26MAR3018":
            return [
                {
                    "series_ticker": "KXBTCD",
                    "ticker": "KXBTCD-26MAR3018-T65000",
                    "contract_type": "threshold",
                    "expiry": "2026-03-30T18:00:00Z",
                    "threshold": 65000,
                    "direction": "above",
                    "status": "finalized",
                }
            ]
        if kwargs.get("event_ticker"):
            return []
        return [
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-65000",
                "contract_type": "threshold",
                "expiry": "2026-03-30T16:00:00Z",
                "threshold": 65000,
                "direction": "above",
            }
        ]

    def get_batch_market_candlesticks(self, **kwargs):
        if self.batch_raises:
            import requests

            response = requests.Response()
            response.status_code = 400
            raise requests.HTTPError(response=response)
        return {
            "markets": [
                {
                    "market_ticker": "KXBTCD-65000",
                    "candlesticks": [
                        {
                            "end_period_ts": 1774882800,
                            "yes_bid": {"close_dollars": "0.44"},
                            "yes_ask": {"close_dollars": "0.45"},
                            "price": {"close_dollars": "0.445", "mean_dollars": "0.442"},
                            "volume_fp": "12.0",
                            "open_interest_fp": "20.0",
                        }
                    ],
                }
            ]
        }

    def get_market_candlesticks(self, **kwargs):
        return {
            "candlesticks": [
                {
                    "end_period_ts": 1774882800,
                    "yes_bid": {"close_dollars": "0.44"},
                    "yes_ask": {"close_dollars": "0.45"},
                    "price": {"close_dollars": "0.445", "mean_dollars": "0.442"},
                    "volume_fp": "12.0",
                    "open_interest_fp": "20.0",
                }
            ]
        }

    def list_historical_markets(self, **kwargs):
        if kwargs.get("event_ticker") == "KXBTCD-26MAR3018":
            return {
                "markets": [
                    {
                        "series_ticker": "KXBTCD",
                        "ticker": "KXBTCD-26MAR3018-T65000",
                        "contract_type": "threshold",
                        "expiry": "2026-03-30T18:00:00Z",
                        "threshold": 65000,
                        "direction": "above",
                    }
                ],
                "cursor": None,
            }
        return {
            "markets": [
                {
                    "series_ticker": "KXBTCD",
                    "ticker": "KXBTCD-25DEC2913-T80000",
                    "contract_type": "threshold",
                    "expiry": "2025-12-29T13:00:00Z",
                    "threshold": 80000,
                    "direction": "above",
                }
            ],
            "cursor": None,
        }

    def list_events(self, **kwargs):
        return {
            "events": [
                {
                    "event_ticker": "KXBTCD-26MAR3018",
                    "settlement_ts": "2026-03-30T18:00:00Z",
                },
                {
                    "event_ticker": "KXBTCD-25DEC2913",
                    "settlement_ts": "2025-12-29T13:00:00Z",
                },
            ],
            "cursor": None,
        }

    def get_historical_cutoff(self, **kwargs):
        return {"market_settled_ts": "2025-12-30T00:00:00Z"}

    def get_historical_market_candlesticks(self, **kwargs):
        return self.get_market_candlesticks(**kwargs)


class BackfillTests(unittest.TestCase):
    def test_backfill_writes_snapshots(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "backfill_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        service = CandlestickBackfillService(
            kalshi_client=FakeKalshiBackfillClient(),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=SnapshotStore(db_path),
            config=BackfillConfig(series_ticker="KXBTCD", start_ts=1, end_ts=2),
        )
        snapshots = service.backfill()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(service.snapshot_store.snapshot_count(), 1)
        self.assertEqual(snapshots[0].spot_price, 64001)

    def test_backfill_can_fallback_to_single_market_requests(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "backfill_fallback_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        service = CandlestickBackfillService(
            kalshi_client=FakeKalshiBackfillClient(batch_raises=True),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=SnapshotStore(db_path),
            config=BackfillConfig(series_ticker="KXBTCD", start_ts=1, end_ts=2),
        )
        snapshots = service.backfill()
        self.assertEqual(len(snapshots), 1)

    def test_historical_backfill_discovers_historical_markets(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "backfill_historical_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        service = CandlestickBackfillService(
            kalshi_client=FakeKalshiBackfillClient(),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=SnapshotStore(db_path),
            config=BackfillConfig(
                series_ticker="KXBTCD",
                start_ts=1774890000,
                end_ts=1774900000,
                use_historical_markets=True,
            ),
        )
        snapshots = service.backfill()
        self.assertEqual(len(snapshots), 1)


if __name__ == "__main__":
    unittest.main()
