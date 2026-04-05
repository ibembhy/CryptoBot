from pathlib import Path
import unittest

from kalshi_btc_bot.collectors.settlements import SettlementEnricher, SettlementEnrichmentConfig
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot
from datetime import datetime, timedelta, timezone


class FakeSettlementKalshiClient(KalshiClient):
    def get_market(self, ticker: str, session=None):
        return {"market": {"ticker": ticker, "result": "yes"}}

    def get_historical_market(self, ticker: str, session=None):
        return {"market": {"ticker": ticker, "result": "yes"}}


class SettlementTests(unittest.TestCase):
    def test_enrich_updates_settlement_values(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "settlement_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        observed_at = datetime.now(timezone.utc) - timedelta(hours=2)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-65000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=observed_at + timedelta(hours=1),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
        )
        store.insert_snapshot(snapshot)
        enricher = SettlementEnricher(
            kalshi_client=FakeSettlementKalshiClient(),
            snapshot_store=store,
            config=SettlementEnrichmentConfig(series_ticker="KXBTCD"),
        )
        result = enricher.enrich()
        self.assertEqual(result["markets_updated"], 1)
        loaded = store.load_snapshots(series_ticker="KXBTCD")
        self.assertEqual(loaded[0].settlement_price, 1.0)

    def test_enrich_skips_unexpired_markets_by_default(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "settlement_future_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-FUTURE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2099, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
        )
        store.insert_snapshot(snapshot)
        enricher = SettlementEnricher(
            kalshi_client=FakeSettlementKalshiClient(),
            snapshot_store=store,
            config=SettlementEnrichmentConfig(series_ticker="KXBTCD"),
        )
        result = enricher.enrich()
        self.assertEqual(result["markets_checked"], 0)


if __name__ == "__main__":
    unittest.main()
