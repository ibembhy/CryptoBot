from __future__ import annotations

from pathlib import Path
import unittest

from kalshi_btc_bot.models.repricing_target import (
    RepricingRegimeProfile,
    RepricingModelProfile,
    load_repricing_profile,
    save_repricing_profile,
)


class RepricingCacheTests(unittest.TestCase):
    def test_save_and_load_repricing_profile_round_trip(self):
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        cache_path = base_dir / "repricing_profile_test.json"
        if cache_path.exists():
            cache_path.unlink()

        profile = RepricingModelProfile(
            horizon_minutes=5,
            sample_count=250,
            fallback=RepricingRegimeProfile(
                intercept_cents=1.25,
                coefficients={"current_yes_probability": 2.0, "spread_cents": -0.5},
                sample_count=250,
                target_mean_cents=0.3,
                target_std_cents=4.2,
                yes_profit_rate=0.61,
                no_profit_rate=0.44,
            ),
            regime_profiles={
                "near|low_vol": RepricingRegimeProfile(
                    intercept_cents=1.5,
                    coefficients={"current_yes_probability": 1.0, "spread_cents": -0.25},
                    sample_count=100,
                    target_mean_cents=0.5,
                    target_std_cents=3.1,
                    yes_profit_rate=0.66,
                    no_profit_rate=0.40,
                )
            },
            near_money_regime_bps=150.0,
            high_vol_regime_threshold=0.8,
        )
        metadata = {"series_ticker": "KXBTCD", "snapshot_count": 123, "last_observed_at": "2026-03-31T20:00:00+00:00", "horizon_minutes": 5}
        save_repricing_profile(cache_path, profile, metadata=metadata)

        loaded = load_repricing_profile(cache_path, expected_metadata=metadata)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.sample_count, 250)
        self.assertAlmostEqual(loaded.fallback.intercept_cents, 1.25, places=4)
        self.assertIn("near|low_vol", loaded.regime_profiles)

        mismatch = load_repricing_profile(cache_path, expected_metadata={**metadata, "snapshot_count": 999})
        self.assertIsNone(mismatch)


if __name__ == "__main__":
    unittest.main()
