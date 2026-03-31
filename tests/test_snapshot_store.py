from datetime import datetime, timezone
from pathlib import Path
import unittest

from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.types import MarketSnapshot


class SnapshotStoreTests(unittest.TestCase):
    def test_insert_snapshot_persists_row(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTC",
            market_ticker="KXBTC-65000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
        )
        store.insert_snapshot(snapshot)
        self.assertEqual(store.snapshot_count(), 1)

    def test_load_snapshots_returns_domain_objects(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_load_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-65000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
        )
        store.insert_snapshot(snapshot)
        loaded = store.load_snapshots(series_ticker="KXBTCD")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].market_ticker, "KXBTCD-65000")

    def test_dataset_summary_reports_basic_counts(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_summary_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-65000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
            settlement_price=65100,
        )
        store.insert_snapshot(snapshot)
        summary = store.dataset_summary(series_ticker="KXBTCD")
        self.assertEqual(summary["snapshot_count"], 1)
        self.assertEqual(summary["unique_markets"], 1)
        self.assertEqual(summary["settled_markets"], 1)
        self.assertEqual(summary["expired_markets"], 1)
        self.assertEqual(summary["replay_ready_markets"], 1)
        self.assertEqual(summary["unresolved_expired_markets"], 0)

    def test_load_snapshots_can_filter_to_replay_ready_markets(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_replay_ready_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        replay_ready = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-READY",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
            settlement_price=65100,
        )
        unresolved = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-OPEN",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 4, 1, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
        )
        store.insert_snapshot(replay_ready)
        store.insert_snapshot(unresolved)
        loaded = store.load_snapshots(
            series_ticker="KXBTCD",
            replay_ready_only=True,
            reference_time=datetime(2026, 3, 31, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].market_ticker, "KXBTCD-READY")

    def test_load_snapshots_prefers_latest_duplicate_for_same_market_and_timestamp(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_dedupe_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        observed_at = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
        original = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-DUPE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64000,
            threshold=65000,
            direction=None,
            yes_bid=0.44,
            yes_ask=0.45,
        )
        corrected = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-DUPE",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=observed_at,
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=65050,
            threshold=65000,
            direction="above",
            yes_bid=0.46,
            yes_ask=0.47,
        )
        store.insert_snapshot(original)
        store.insert_snapshot(corrected)
        loaded = store.load_snapshots(series_ticker="KXBTCD")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].spot_price, 65050)
        self.assertEqual(loaded[0].direction, "above")

    def test_summary_and_loading_use_effective_expiry_from_metadata(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_effective_expiry_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-WRONG-EXPIRY",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 31, 14, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 4, 7, 21, 0, tzinfo=timezone.utc),
            spot_price=68000,
            threshold=69000,
            direction="above",
            yes_bid=0.44,
            yes_ask=0.45,
            settlement_price=1.0,
            metadata={
                "close_time": "2026-03-31T15:00:00Z",
                "expiration_time": "2026-04-07T21:00:00Z",
            },
        )
        store.insert_snapshot(snapshot)
        summary = store.dataset_summary(
            series_ticker="KXBTCD",
            reference_time=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(summary["expired_markets"], 1)
        self.assertEqual(summary["replay_ready_markets"], 1)
        loaded = store.load_snapshots(
            series_ticker="KXBTCD",
            replay_ready_only=True,
            reference_time=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].expiry, datetime(2026, 3, 31, 15, 0, tzinfo=timezone.utc))

    def test_load_snapshots_accepts_z_suffix_timestamps(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        db_path = base_dir / "snapshot_store_z_suffix_test.sqlite3"
        if db_path.exists():
            db_path.unlink()
        store = SnapshotStore(db_path)
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-Z-SUFFIX",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 4, 7, 21, 0, tzinfo=timezone.utc),
            spot_price=68000,
            threshold=69000,
            direction="above",
            metadata={
                "close_time": "2026-03-31T17:00:00Z",
                "expiration_time": "2026-04-07T21:00:00Z",
            },
        )
        store.insert_snapshot(snapshot)
        loaded = store.load_snapshots(series_ticker="KXBTCD")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].observed_at, datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc))
        self.assertEqual(loaded[0].expiry, datetime(2026, 3, 31, 17, 0, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
