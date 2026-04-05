from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

from kalshi_btc_bot.collectors.hybrid import HybridCollector, HybridCollectorConfig
from kalshi_btc_bot.data.coinbase import CoinbaseClient
from kalshi_btc_bot.markets.kalshi import KalshiClient


class FakeSnapshotStore:
    def __init__(self):
        self.snapshots = []

    def insert_snapshot(self, snapshot):
        self.snapshots.append(snapshot)

    def snapshot_count(self):
        return len(self.snapshots)


class FakeCoinbaseClient(CoinbaseClient):
    def __init__(self, price: float):
        super().__init__()
        self.price = price

    def get_spot_price(self, session=None) -> float:
        return self.price


class SequencedCoinbaseClient(CoinbaseClient):
    def __init__(self, prices: list[float]):
        super().__init__()
        self._prices = list(prices)
        self.calls = 0

    def get_spot_price(self, session=None) -> float:
        price = self._prices[min(self.calls, len(self._prices) - 1)]
        self.calls += 1
        return price


class FakeKalshiClient(KalshiClient):
    def __init__(self, markets):
        super().__init__()
        self._markets = markets
        self._events = []

    def list_markets(self, **kwargs):
        return list(self._markets)

    def list_markets_page(self, **kwargs):
        return {"markets": list(self._markets), "cursor": None}

    def list_events(self, **kwargs):
        return {"events": list(self._events)}


class HybridCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_spot_cache_refreshes_after_configured_interval(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_spot_cache.sqlite3"
        if db_path.exists():
            db_path.unlink()

        coinbase_client = SequencedCoinbaseClient([64000.0, 64100.0])
        collector = HybridCollector(
            kalshi_client=FakeKalshiClient([]),
            coinbase_client=coinbase_client,
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", spot_refresh_interval_seconds=3),
        )

        observed_at = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
        first_spot = collector._get_spot_price(observed_at)
        cached_spot = collector._get_spot_price(observed_at + timedelta(seconds=2))
        refreshed_spot = collector._get_spot_price(observed_at + timedelta(seconds=4))

        self.assertEqual(first_spot, 64000.0)
        self.assertEqual(cached_spot, 64000.0)
        self.assertEqual(refreshed_spot, 64100.0)
        self.assertEqual(coinbase_client.calls, 2)

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
            snapshot_store=FakeSnapshotStore(),
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
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", market_limit=2, max_minutes_to_expiry=60),
        )

        snapshots = await collector.bootstrap()
        self.assertEqual(kalshi_client.calls, 2)
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all(snapshot.market_ticker.startswith("KXBTCD-NEAR-") for snapshot in snapshots))

    async def test_bootstrap_targets_current_event_markets_before_series_wide_scan(self):
        observed_at = datetime(2026, 4, 3, 21, 33, tzinfo=timezone.utc)
        current_event = "KXBTCD-26APR0318"
        far_event = "KXBTCD-26APR0417"
        near_markets = [
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-26APR0318-T66899.99",
                "contract_type": "threshold",
                "close_time": "2026-04-03T22:00:00Z",
                "threshold": 66899.99,
                "direction": "above",
                "yes_ask": 0.44,
                "yes_bid": 0.43,
            }
        ]
        far_markets = [
            {
                "series_ticker": "KXBTCD",
                "ticker": "KXBTCD-26APR0417-T74999.99",
                "contract_type": "threshold",
                "close_time": "2026-04-04T21:00:00Z",
                "threshold": 74999.99,
                "direction": "above",
                "yes_ask": 0.05,
                "yes_bid": 0.04,
            }
        ]

        class EventTargetingKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.calls: list[tuple[str | None, str | None]] = []

            def list_events(self, **kwargs):
                return {
                    "events": [
                        {"event_ticker": far_event},
                        {"event_ticker": current_event},
                    ]
                }

            def list_markets_page(self, **kwargs):
                self.calls.append((kwargs.get("series_ticker"), kwargs.get("event_ticker")))
                if kwargs.get("event_ticker") == current_event:
                    return {"markets": near_markets, "cursor": None}
                if kwargs.get("event_ticker") == far_event:
                    return {"markets": far_markets, "cursor": None}
                return {"markets": far_markets, "cursor": None}

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_event_targeting.sqlite3"
        if db_path.exists():
            db_path.unlink()

        kalshi_client = EventTargetingKalshiClient()
        collector = HybridCollector(
            kalshi_client=kalshi_client,
            coinbase_client=FakeCoinbaseClient(66950),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", market_limit=10, max_minutes_to_expiry=30),
        )

        snapshots = collector._discover_markets(observed_at=observed_at, spot_price=66950.0)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["ticker"], "KXBTCD-26APR0318-T66899.99")
        self.assertIn(("KXBTCD", current_event), kalshi_client.calls)

    async def test_parse_event_close_supports_hourly_and_quarter_hour_event_tickers(self):
        hourly = HybridCollector._parse_event_close({"event_ticker": "KXBTCD-26APR0318"})
        fifteen = HybridCollector._parse_event_close({"event_ticker": "KXBTC15M-26APR031645"})
        self.assertIsNotNone(hourly)
        self.assertIsNotNone(fifteen)
        assert hourly is not None
        assert fifteen is not None
        self.assertEqual(hourly.isoformat(), "2026-04-03T22:00:00+00:00")
        self.assertEqual(fifteen.isoformat(), "2026-04-03T20:45:00+00:00")

    async def test_parse_market_expiry_prefers_close_time_over_later_expiration_time(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        raw_market = {
            "close_time": (near_expiry + timedelta(minutes=15)).isoformat().replace("+00:00", "Z"),
            "expiration_time": (near_expiry + timedelta(days=7)).isoformat().replace("+00:00", "Z"),
        }
        parsed = HybridCollector._parse_market_expiry(raw_market)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertLess(parsed, near_expiry + timedelta(hours=1))

    async def test_event_discovery_uses_cache_and_stale_cache_on_failure(self):
        observed_at = datetime(2026, 4, 3, 21, 33, tzinfo=timezone.utc)

        class CachingKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.event_calls = 0
                self.raise_on_events = False

            def list_events(self, **kwargs):
                self.event_calls += 1
                if self.raise_on_events:
                    raise RuntimeError("429 Too Many Requests")
                return {
                    "events": [
                        {"event_ticker": "KXBTCMAXMON-26APR0318"},
                    ]
                }

        kalshi_client = CachingKalshiClient()
        collector = HybridCollector(
            kalshi_client=kalshi_client,
            coinbase_client=FakeCoinbaseClient(66950),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(
                series_ticker="KXBTCMAXMON",
                market_limit=10,
                max_minutes_to_expiry=30,
                event_discovery_cache_seconds=10,
            ),
        )

        first = collector._discover_target_event_tickers(series_ticker="KXBTCMAXMON", observed_at=observed_at)
        second = collector._discover_target_event_tickers(series_ticker="KXBTCMAXMON", observed_at=observed_at + timedelta(seconds=5))
        kalshi_client.raise_on_events = True
        third = collector._discover_target_event_tickers(series_ticker="KXBTCMAXMON", observed_at=observed_at + timedelta(seconds=20))

        self.assertEqual(first, ["KXBTCMAXMON-26APR0318"])
        self.assertEqual(second, ["KXBTCMAXMON-26APR0318"])
        self.assertEqual(third, ["KXBTCMAXMON-26APR0318"])
        self.assertEqual(kalshi_client.event_calls, 2)

    async def test_event_discovery_uses_heuristics_for_hourly_and_fifteen_minute_series(self):
        class CountingKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.event_calls = 0

            def list_events(self, **kwargs):
                self.event_calls += 1
                return {"events": []}

        kalshi_client = CountingKalshiClient()
        collector = HybridCollector(
            kalshi_client=kalshi_client,
            coinbase_client=FakeCoinbaseClient(66950),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", max_minutes_to_expiry=35),
        )

        hourly = collector._discover_target_event_tickers(
            series_ticker="KXBTCD",
            observed_at=datetime(2026, 4, 3, 21, 50, tzinfo=timezone.utc),
        )
        fifteen = collector._discover_target_event_tickers(
            series_ticker="KXBTC15M",
            observed_at=datetime(2026, 4, 3, 20, 37, tzinfo=timezone.utc),
        )

        self.assertEqual(hourly, ["KXBTCD-26APR0318"])
        self.assertEqual(fifteen, ["KXBTC15M-26APR031645", "KXBTC15M-26APR031700"])
        self.assertEqual(kalshi_client.event_calls, 0)

    async def test_bootstrap_replaces_registry_when_market_roster_changes(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        class ChangingKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def list_markets_page(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    markets = [
                        {
                            "series_ticker": "KXBTCD",
                            "ticker": "KXBTCD-OLD",
                            "contract_type": "threshold",
                            "expiry": (near_expiry + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                            "threshold": 65000,
                            "direction": "above",
                            "yes_ask": 0.45,
                            "yes_bid": 0.44,
                        }
                    ]
                else:
                    markets = [
                        {
                            "series_ticker": "KXBTCD",
                            "ticker": "KXBTCD-NEW",
                            "contract_type": "threshold",
                            "expiry": (near_expiry + timedelta(minutes=9)).isoformat().replace("+00:00", "Z"),
                            "threshold": 65010,
                            "direction": "above",
                            "yes_ask": 0.46,
                            "yes_bid": 0.45,
                        }
                    ]
                return {"markets": markets, "cursor": None}

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_registry_replace.sqlite3"
        if db_path.exists():
            db_path.unlink()

        collector = HybridCollector(
            kalshi_client=ChangingKalshiClient(),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", max_minutes_to_expiry=60),
        )

        await collector.bootstrap()
        self.assertEqual(set(collector.market_registry), {"KXBTCD-OLD"})

        await collector.bootstrap()
        self.assertEqual(set(collector.market_registry), {"KXBTCD-NEW"})

    async def test_bootstrap_closes_live_websocket_when_market_roster_changes(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)

        class StaticKalshiClient(KalshiClient):
            def __init__(self):
                super().__init__()
                self.ticker = "KXBTCD-OLD"

            def list_markets_page(self, **kwargs):
                return {
                    "markets": [
                        {
                            "series_ticker": "KXBTCD",
                            "ticker": self.ticker,
                            "contract_type": "threshold",
                            "expiry": (near_expiry + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                            "threshold": 65000,
                            "direction": "above",
                            "yes_ask": 0.45,
                            "yes_bid": 0.44,
                        }
                    ],
                    "cursor": None,
                }

        class FakeWebSocket:
            def __init__(self):
                self.closed = False
                self.close_calls = 0

            async def close(self):
                self.closed = True
                self.close_calls += 1

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_refresh_ws.sqlite3"
        if db_path.exists():
            db_path.unlink()

        kalshi_client = StaticKalshiClient()
        collector = HybridCollector(
            kalshi_client=kalshi_client,
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(series_ticker="KXBTCD", max_minutes_to_expiry=60),
        )

        await collector.bootstrap()
        fake_websocket = FakeWebSocket()
        collector._websocket = fake_websocket

        kalshi_client.ticker = "KXBTCD-NEW"
        await collector.bootstrap()

        self.assertEqual(fake_websocket.close_calls, 1)

    async def test_bootstrap_collects_from_multiple_series(self):
        near_expiry = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        markets_by_series = {
            "KXBTCD": [
                {
                    "series_ticker": "KXBTCD",
                    "ticker": "KXBTCD-NEAR",
                    "contract_type": "threshold",
                    "expiry": (near_expiry + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
                    "threshold": 65000,
                    "direction": "above",
                    "yes_ask": 0.45,
                    "yes_bid": 0.44,
                }
            ],
            "KXBTC15M": [
                {
                    "series_ticker": "KXBTC15M",
                    "ticker": "KXBTC15M-NEAR",
                    "close_time": (near_expiry + timedelta(minutes=9)).isoformat().replace("+00:00", "Z"),
                    "expected_expiration_time": (near_expiry + timedelta(minutes=14)).isoformat().replace("+00:00", "Z"),
                    "floor_strike": 64990.0,
                    "title": "BTC price up in next 15 mins?",
                    "yes_sub_title": "Target Price: $64,990.00",
                    "yes_ask": 0.52,
                    "yes_bid": 0.50,
                }
            ],
        }

        class MultiSeriesFakeKalshiClient(KalshiClient):
            def list_markets_page(self, **kwargs):
                series_ticker = kwargs.get("series_ticker")
                return {"markets": list(markets_by_series.get(series_ticker, [])), "cursor": None}

        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "hybrid_collector_multi_series.sqlite3"
        if db_path.exists():
            db_path.unlink()

        collector = HybridCollector(
            kalshi_client=MultiSeriesFakeKalshiClient(),
            coinbase_client=FakeCoinbaseClient(64000),
            snapshot_store=FakeSnapshotStore(),
            config=HybridCollectorConfig(
                series_ticker="KXBTCD",
                series_tickers=["KXBTCD", "KXBTC15M"],
                max_minutes_to_expiry=60,
            ),
        )

        snapshots = await collector.bootstrap()
        self.assertEqual(len(snapshots), 2)
        self.assertEqual({snapshot.series_ticker for snapshot in snapshots}, {"KXBTCD", "KXBTC15M"})
        self.assertEqual({snapshot.market_ticker for snapshot in snapshots}, {"KXBTCD-NEAR", "KXBTC15M-NEAR"})


if __name__ == "__main__":
    unittest.main()
