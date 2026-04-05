from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.models.repricing_target import RepricingModelProfile, RepricingRegimeProfile, RepricingTargetModel
from kalshi_btc_bot.signals.calibration import (
    build_engine_calibrators,
    calibration_regime_key,
    fit_probability_calibrator,
    load_engine_calibrators,
    save_engine_calibrators,
)
from kalshi_btc_bot.signals.engine import SignalConfig, generate_signal
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.types import MarketSnapshot, ProbabilityEstimate


class SignalCalibrationTests(unittest.TestCase):
    def test_fit_probability_calibrator_builds_monotonic_mapping(self):
        trades = pd.DataFrame(
            {
                "raw_model_probability": [0.2, 0.25, 0.3, 0.55, 0.6, 0.65, 0.8, 0.85, 0.9],
                "contract_won": [0, 0, 0, 1, 0, 1, 1, 1, 1],
            }
        )
        calibrator = fit_probability_calibrator(trades, min_samples=5, min_bucket_count=1, bucket_width=0.2)
        self.assertIsNotNone(calibrator)
        assert calibrator is not None
        self.assertLessEqual(calibrator.apply(0.25), calibrator.apply(0.65))

    def test_generate_signal_preserves_raw_probability_and_uses_calibrated_probability(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-CAL",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 31, 16, 20, tzinfo=timezone.utc),
            spot_price=67000.0,
            threshold=67100.0,
            direction="above",
            yes_bid=0.37,
            yes_ask=0.39,
            no_bid=0.60,
            no_ask=0.62,
            volume=1000.0,
        )
        estimate = ProbabilityEstimate(
            model_name="gbm_threshold",
            observed_at=snapshot.observed_at,
            expiry=snapshot.expiry,
            spot_price=snapshot.spot_price,
            target_price=67100.0,
            volatility=0.8,
            drift=0.0,
            probability=0.58,
            raw_probability=0.66,
        )
        signal = generate_signal(
            snapshot,
            estimate,
            SignalConfig(0.01, 0.0, 25, 80, uncertainty_penalty=0.0, max_spread_cents=10, max_data_age_seconds=10**9),
        )
        self.assertAlmostEqual(signal.raw_model_probability, 0.66, places=4)
        self.assertAlmostEqual(signal.model_probability, 0.58, places=4)

    def test_build_engine_calibrators_skips_repricing_models(self):
        observed_at = datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc)
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTCD",
                market_ticker="KXBTCD-CAL-A",
                contract_type="threshold",
                underlying_symbol="BTC-USD",
                observed_at=observed_at,
                expiry=observed_at,
                spot_price=67000.0,
                threshold=67100.0,
                direction="above",
                yes_bid=0.0,
                yes_ask=0.01,
                no_bid=0.99,
                no_ask=1.0,
                settlement_price=1.0,
            )
        ]
        feature_frame = pd.DataFrame({"realized_volatility": [0.8]}, index=pd.to_datetime([observed_at], utc=True))
        repricing = RepricingTargetModel(
            profile=RepricingModelProfile(
                horizon_minutes=5,
                sample_count=10,
                fallback=RepricingRegimeProfile(
                    intercept_cents=0.0,
                    coefficients={name: 0.0 for name in (
                        "current_yes_probability",
                        "recent_log_return_bps",
                        "return_3_bps",
                        "return_5_bps",
                        "price_acceleration_bps",
                        "realized_volatility_1",
                        "realized_volatility_3",
                        "realized_volatility_5",
                        "realized_volatility",
                        "vol_of_vol",
                        "signed_distance_bps",
                        "minutes_to_expiry",
                        "spread_cents",
                        "liquidity_log",
                        "kalshi_momentum_1_cents",
                        "kalshi_momentum_3_cents",
                        "kalshi_reversion_gap_cents",
                        "spread_regime_cents",
                        "liquidity_regime",
                        "quote_age_seconds",
                        "stale_quote_flag",
                        "btc_micro_jump_flag",
                    )},
                    sample_count=10,
                    target_mean_cents=0.0,
                    target_std_cents=1.0,
                    yes_profit_rate=0.5,
                    no_profit_rate=0.5,
                ),
                regime_profiles={},
                near_money_regime_bps=150.0,
                high_vol_regime_threshold=0.8,
            )
        )
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            models={"gbm_threshold": GBMThresholdModel(), "repricing_target": repricing},
            signal_config=SignalConfig(0.01, 0.0, 1, 99, max_data_age_seconds=10**9),
            exit_config=ExitConfig(8, 10, 3, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
        )
        calibrators = build_engine_calibrators(
            engine,
            snapshots=snapshots,
            feature_frame=feature_frame,
            min_samples=1,
            min_bucket_count=1,
        )
        self.assertNotIn("repricing_target", calibrators)

    def test_calibration_regime_key_uses_expected_buckets(self):
        snapshot = MarketSnapshot(
            source="test",
            series_ticker="KXBTCD",
            market_ticker="KXBTCD-REGIME",
            contract_type="threshold",
            underlying_symbol="BTC-USD",
            observed_at=datetime(2026, 3, 31, 16, 0, tzinfo=timezone.utc),
            expiry=datetime(2026, 3, 31, 16, 10, tzinfo=timezone.utc),
            spot_price=67000.0,
            threshold=67050.0,
            direction="above",
            yes_bid=0.48,
            yes_ask=0.50,
            no_bid=0.49,
            no_ask=0.51,
            metadata={"signed_distance_bps": -7.46, "realized_volatility": 0.9, "spread_cents": 2.0},
        )
        regime = calibration_regime_key(snapshot)
        self.assertEqual(regime, "near|expiry_0_15|high_vol|tight_spread")

    def test_save_and_load_engine_calibrators_round_trip(self):
        trades = pd.DataFrame(
            {
                "raw_model_probability": [0.2, 0.3, 0.7, 0.8],
                "contract_won": [0, 0, 1, 1],
            }
        )
        calibrator = fit_probability_calibrator(trades, min_samples=1, min_bucket_count=1, bucket_width=0.2)
        assert calibrator is not None
        wrapped = {
            "gbm_threshold": build_engine_calibrators(
                BacktestEngine(
                    model=GBMThresholdModel(),
                    models={"gbm_threshold": GBMThresholdModel()},
                    signal_config=SignalConfig(0.01, 0.0, 1, 99, max_data_age_seconds=10**9),
                    exit_config=ExitConfig(8, 10, 3, 5),
                    backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
                ),
                snapshots=[],
                feature_frame=pd.DataFrame(),
                min_samples=999,
            )
        }
        # Replace the empty build result with a known calibrator payload for serialization coverage.
        from kalshi_btc_bot.signals.calibration import RegimeAwareProbabilityCalibrator

        wrapped = {
            "gbm_threshold": RegimeAwareProbabilityCalibrator(
                global_calibrator=calibrator,
                regime_calibrators={"near|expiry_0_15|high_vol|tight_spread": calibrator},
                sample_count=calibrator.sample_count,
            )
        }
        base_dir = Path("test_artifacts")
        base_dir.mkdir(exist_ok=True)
        cache_path = base_dir / "calibration_cache_test.json"
        if cache_path.exists():
            cache_path.unlink()
        metadata = {"series_ticker": "KXBTCD", "snapshot_count": 10, "last_observed_at": "2026-03-31T20:00:00+00:00"}
        save_engine_calibrators(cache_path, wrapped, metadata=metadata)
        loaded = load_engine_calibrators(cache_path, expected_metadata=metadata)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIn("gbm_threshold", loaded)
        self.assertIn("near|expiry_0_15|high_vol|tight_spread", loaded["gbm_threshold"].regime_calibrators)
        mismatch = load_engine_calibrators(cache_path, expected_metadata={**metadata, "snapshot_count": 999})
        self.assertIsNone(mismatch)


if __name__ == "__main__":
    unittest.main()
