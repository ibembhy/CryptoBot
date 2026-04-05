from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kalshi_btc_bot.backtest.replay import build_feature_frame_from_snapshots, build_replay_dataset
from kalshi_btc_bot.types import MarketSnapshot


def make_snapshots() -> list[MarketSnapshot]:
    return [
        MarketSnapshot(
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
            yes_bid=0.40,
            yes_ask=0.41,
        ),
        MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-65000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 5, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64100,
            threshold=65000,
            direction="above",
            yes_bid=0.42,
            yes_ask=0.43,
        ),
        MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-66000",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 30, 15, 5, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            spot_price=64100,
            threshold=66000,
            direction="above",
            yes_bid=0.22,
            yes_ask=0.23,
        ),
    ]


class ReplayTests(unittest.TestCase):
    def test_build_feature_frame_from_snapshots(self):
        frame = build_feature_frame_from_snapshots(
            make_snapshots(),
            volatility_window=2,
            annualization_factor=365.0 * 24 * 60,
        )
        self.assertGreaterEqual(len(frame), 2)
        self.assertIn("realized_volatility", frame.columns)
        self.assertIn("realized_volatility_3", frame.columns)
        self.assertIn("btc_micro_jump_flag", frame.columns)

    def test_build_replay_dataset_filters_time_window(self):
        dataset = build_replay_dataset(
            make_snapshots(),
            volatility_window=2,
            annualization_factor=365.0 * 24 * 60,
            observed_from=datetime(2026, 3, 30, 15, 5, tzinfo=timezone.utc),
        )
        self.assertEqual(len(dataset.snapshots), 2)

    def test_build_replay_dataset_uses_disk_cache(self):
        with TemporaryDirectory() as temp_dir:
            dataset = build_replay_dataset(
                make_snapshots(),
                volatility_window=2,
                annualization_factor=365.0 * 24 * 60,
                cache_dir=temp_dir,
            )
            cache_files = list(Path(temp_dir).glob("*.pkl"))
            self.assertEqual(len(cache_files), 1)
            with patch(
                "kalshi_btc_bot.backtest.replay.build_feature_frame_from_snapshots",
                side_effect=AssertionError("feature frame should have been loaded from cache"),
            ):
                cached = build_replay_dataset(
                    make_snapshots(),
                    volatility_window=2,
                    annualization_factor=365.0 * 24 * 60,
                    cache_dir=temp_dir,
                )
            self.assertEqual(len(cached.feature_frame), len(dataset.feature_frame))
            self.assertEqual(list(cached.feature_frame.columns), list(dataset.feature_frame.columns))


if __name__ == "__main__":
    unittest.main()
