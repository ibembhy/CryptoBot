from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

from kalshi_btc_bot.collectors.hybrid import HybridCollector, HybridCollectorConfig
from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.storage.snapshots import SnapshotStore


class FakeCoinbaseClient(CoinbaseClient):
    def __init__(self, price: float):
        super().__init__()
        self.price = price

    def get_spot_price(self, session=None) -> float:
        return self.price


class FakeKalshiClient(KalshiClient):
    def __init__(self, markets):
        super().__init__()
        self._markets = markets

    def list_markets(self, **kwargs):
        return list(self._markets)

    def list_markets_page(self, **kwargs):
        return {"markets": list(self._markets), "cursor": None}


class HybridCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_and_ticker_payload_write_snapshots(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        markets = [
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-65000",
                "contract_type": "threshold",
                "expiry": (near_expiry.replace(minute=near_expiry.minute) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
                "threshold": 65000,
                "direction": "above",
                "yes_ask": 0.45,
                "yes_bid": 0.44,
            },
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-70000",
                "contract_type": "threshold",
                "expiry": (near_expiry + timedelta(hours=8)).isoformat().replace("+00:00", "Z"),
                "threshold": 70000,
                "direction": "above",
                "yes_ask": 0.20,
                "yes_bid": 0.19,
            },
        ]
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        collector = HybridCollector(
            kalshi_client=FakeKalshiClient(markets),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=SnapshotStore(db_path),
            config=HybridCollectorConfig(series_ticker="KXBTCD"),
        )
        snapshots = await collector.bootstrap()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].market_ticker, "KXBTCD-65000")
        payload = {
            "type": "ticker",
            "msg": {
                "market_ticker": "KXBTCD-65000",
                "yes_bid_dollars": "0.51",
                "yes_ask_dollars": "0.52",
                "ts": datetime(2026, 3, 30, 15, 10, tzinfo=timezone.utc).timestamp(),
            },
        }
        snapshot = await collector.handle_websocket_payload(payload)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.yes_ask, 0.52)
        self.assertEqual(collector.snapshot_store.snapshot_count(), 2)

    async def test_bootstrap_paginates_and_prefers_near_money_markets(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        far_markets = [
            {
                "series_ticker": "KXBTCD",
                "ticker": f"KXBTCD-FAR-{idx}",
                "contract_type": "threshold",
                "expiry": (near_expiry + timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
                "threshold": 76000 + idx,
                "direction": "above",
                "yes_ask": 0.05,
                "yes_bid": 0.04,
            }
            for idx in range(3)
        ]
        near_markets = [
            {
                "series_ticker": "KXBTCD",
                "ticker": f"KXBTCD-NEAR-{idx}",
                "contract_type": "threshold",
                "expiry": (near_expiry + timedelta(minutes=25)).isoformat().replace("+00:00", "Z"),
                "threshold": 64000 + idx,
                "direction": "above",
                "yes_ask": 0.45,
                "yes_bid": 0.44,
            }
            for idx in range(3)
        ]

        class PagedFakeKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.pages = [
                    {"markets": far_markets, "cursor": "page-2"},
                    {"markets": near_markets, "cursor": None},
                ]
                self.calls = 0

            def list_markets_page(self, **kwargs):
                page = self.pages[self.calls]
                self.calls += 1
                return page

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_paged.sqlite3"
        if db_path.exists():
            db_path.unlink()

        kalshi_client = PagedFakeKalshiClient()
        collector = HybridCollector(
            kalshi_client=kalshi_client,
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=SnapshotStore(db_path),
            config=HybridCollectorConfig(series_ticker="KXBTCD", market_limit=2, max_minutes_to_expiry=60),
        )

        snapshots = await collector.bootstrap()
        self.assertEqual(kalshi_client.calls, 2)
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all(snapshot.market_ticker.startswith("KXBTCD-NEAR-") for snapshot in snapshots))


if __name__ == "__main__":
    unittest.main()
