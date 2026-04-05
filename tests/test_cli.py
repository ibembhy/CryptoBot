import unittest

from datetime import datetime, timezone

from kalshi_btc_bot.cli import build_parser, split_snapshots_by_fraction, _resolve_start_of_day_balance
from kalshi_btc_bot.settings import load_settings
from kalshi_btc_bot.types import MarketSnapshot


class CliTests(unittest.TestCase):
    def test_collect_forever_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["collect-forever", "--max-minutes-to-expiry", "60"])
        self.assertEqual(args.command, "collect-forever")
        self.assertEqual(args.log_level, "INFO")
        self.assertEqual(args.max_minutes_to_expiry, 60)

    def test_replay_backtest_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-backtest", "--series", "KXBTCD"])
        self.assertEqual(args.command, "replay-backtest")
        self.assertEqual(args.series, "KXBTCD")

    def test_backfill_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["backfill-candles", "--start-ts", "1", "--end-ts", "2", "--historical"])
        self.assertEqual(args.command, "backfill-candles")
        self.assertTrue(args.historical)

    def test_historical_list_markets_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["historical-list-markets", "--start-ts", "1", "--end-ts", "2"])
        self.assertEqual(args.command, "historical-list-markets")

    def test_historical_cutoff_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["historical-cutoff"])
        self.assertEqual(args.command, "historical-cutoff")

    def test_historical_inspect_market_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["historical-inspect-market", "--ticker", "KXBTCD-TEST"])
        self.assertEqual(args.command, "historical-inspect-market")

    def test_historical_fetch_candles_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["historical-fetch-candles", "--ticker", "KXBTCD-TEST", "--start-ts", "1", "--end-ts", "2", "--series", "KXBTCD"])
        self.assertEqual(args.command, "historical-fetch-candles")
        self.assertEqual(args.series, "KXBTCD")

    def test_dataset_summary_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["dataset-summary"])
        self.assertEqual(args.command, "dataset-summary")

    def test_replay_diagnostics_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-diagnostics"])
        self.assertEqual(args.command, "replay-diagnostics")

    def test_replay_compare_models_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-compare-models"])
        self.assertEqual(args.command, "replay-compare-models")

    def test_replay_failure_analysis_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-failure-analysis"])
        self.assertEqual(args.command, "replay-failure-analysis")

    def test_replay_maker_proxy_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-maker-proxy"])
        self.assertEqual(args.command, "replay-maker-proxy")

    def test_replay_maker_sim_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-maker-sim"])
        self.assertEqual(args.command, "replay-maker-sim")

    def test_replay_walkforward_calibration_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-walkforward-calibration",
                "--sqlite-path",
                "test.sqlite3",
                "--series",
                "KXBTC15M",
                "--train-fraction",
                "0.7",
            ]
        )
        self.assertEqual(args.command, "replay-walkforward-calibration")
        self.assertEqual(args.sqlite_path, "test.sqlite3")
        self.assertEqual(args.series, "KXBTC15M")
        self.assertEqual(args.train_fraction, 0.7)

    def test_replay_compare_models_accepts_focus_filters(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-compare-models",
                "--near-money-bps",
                "100",
                "--max-minutes-to-expiry",
                "60",
                "--min-price-cents",
                "20",
                "--max-price-cents",
                "80",
            ]
        )
        self.assertEqual(args.near_money_bps, 100.0)
        self.assertEqual(args.max_minutes_to_expiry, 60.0)
        self.assertEqual(args.min_price_cents, 20)
        self.assertEqual(args.max_price_cents, 80)

    def test_replay_failure_analysis_accepts_model_and_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-failure-analysis",
                "--model",
                "gbm_threshold",
                "--mode",
                "early_exit",
                "--top",
                "5",
            ]
        )
        self.assertEqual(args.model, "gbm_threshold")
        self.assertEqual(args.mode, "early_exit")
        self.assertEqual(args.top, 5)

    def test_replay_maker_proxy_accepts_model_and_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-maker-proxy",
                "--model",
                "gbm_threshold",
                "--mode",
                "early_exit",
            ]
        )
        self.assertEqual(args.model, "gbm_threshold")
        self.assertEqual(args.mode, "early_exit")

    def test_replay_maker_sim_accepts_model_and_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-maker-sim",
                "--model",
                "gbm_threshold",
                "--mode",
                "early_exit",
            ]
        )
        self.assertEqual(args.model, "gbm_threshold")
        self.assertEqual(args.mode, "early_exit")

    def test_replay_maker_sim_accepts_conservative_fill_assumptions(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-maker-sim",
                "--maker-max-wait-seconds",
                "180",
                "--maker-min-fill-probability",
                "0.45",
                "--maker-stale-quote-age-seconds",
                "75",
                "--maker-max-posted-spread-cents",
                "8",
                "--maker-min-liquidity-score",
                "50",
                "--maker-max-concurrent-positions-per-side",
                "2",
            ]
        )
        self.assertEqual(args.maker_max_wait_seconds, 180)
        self.assertEqual(args.maker_min_fill_probability, 0.45)
        self.assertEqual(args.maker_stale_quote_age_seconds, 75)
        self.assertEqual(args.maker_max_posted_spread_cents, 8)
        self.assertEqual(args.maker_min_liquidity_score, 50.0)
        self.assertEqual(args.maker_max_concurrent_positions_per_side, 2)

    def test_replay_grid_search_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-grid-search", "--top", "5"])
        self.assertEqual(args.command, "replay-grid-search")
        self.assertEqual(args.top, 5)

    def test_rebuild_repricing_profile_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["rebuild-repricing-profile"])
        self.assertEqual(args.command, "rebuild-repricing-profile")

    def test_replay_grid_search_accepts_value_lists(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "replay-grid-search",
                "--near-money-bps-values",
                "100,150",
                "--max-minutes-to-expiry-values",
                "30,60",
                "--min-price-cents-values",
                "20,25",
                "--max-price-cents-values",
                "75,80",
            ]
        )
        self.assertEqual(args.near_money_bps_values, "100,150")
        self.assertEqual(args.max_minutes_to_expiry_values, "30,60")
        self.assertEqual(args.min_price_cents_values, "20,25")
        self.assertEqual(args.max_price_cents_values, "75,80")

    def test_enrich_settlements_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["enrich-settlements", "--recent-hours", "24"])
        self.assertEqual(args.command, "enrich-settlements")
        self.assertEqual(args.recent_hours, 24)

    def test_paper_trade_once_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["paper-trade-once", "--max-minutes-to-expiry", "30"])
        self.assertEqual(args.command, "paper-trade-once")
        self.assertEqual(args.max_minutes_to_expiry, 30)

    def test_paper_trade_forever_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["paper-trade-forever", "--min-minutes-to-expiry", "0"])
        self.assertEqual(args.command, "paper-trade-forever")
        self.assertEqual(args.min_minutes_to_expiry, 0)

    def test_real_order_preview_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(
            ["real-order-preview", "--market", "KXBTC15M-TEST", "--side", "yes", "--yes-price-cents", "47", "--series", "KXBTC15M"]
        )
        self.assertEqual(args.command, "real-order-preview")
        self.assertEqual(args.market, "KXBTC15M-TEST")
        self.assertEqual(args.side, "yes")
        self.assertEqual(args.yes_price_cents, 47)
        self.assertEqual(args.series, "KXBTC15M")

    def test_real_order_submit_command_accepts_live_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            ["real-order-submit", "--market", "KXBTC15M-TEST", "--side", "yes", "--yes-price-cents", "49", "--live"]
        )
        self.assertEqual(args.command, "real-order-submit")
        self.assertTrue(args.live)

    def test_real_trade_once_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["real-trade-once", "--series", "KXBTC15M", "--max-minutes-to-expiry", "15"])
        self.assertEqual(args.command, "real-trade-once")
        self.assertEqual(args.series, "KXBTC15M")
        self.assertEqual(args.max_minutes_to_expiry, 15)

    def test_real_trade_forever_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["real-trade-forever", "--series", "KXBTC15M", "--live"])
        self.assertEqual(args.command, "real-trade-forever")
        self.assertEqual(args.series, "KXBTC15M")
        self.assertTrue(args.live)

    def test_real_trading_status_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["real-trading-status", "--series", "KXBTC15M"])
        self.assertEqual(args.command, "real-trading-status")
        self.assertEqual(args.series, "KXBTC15M")

    def test_real_kill_switch_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["real-kill-switch", "--series", "KXBTC15M", "--enable", "--reason", "test"])
        self.assertEqual(args.command, "real-kill-switch")
        self.assertTrue(args.enable)
        self.assertEqual(args.series, "KXBTC15M")
        self.assertEqual(args.reason, "test")

    def test_default_settings_expose_real_config(self):
        settings = load_settings()
        self.assertEqual(settings.real["ledger_path"], "data/real_trading_ledger.json")
        self.assertEqual(settings.real["stale_order_timeout_seconds"], 10)
        self.assertEqual(settings.real["max_daily_orders"], 0)
        self.assertEqual(settings.real["max_open_orders"], 2)
        self.assertEqual(settings.real["max_open_positions"], 2)
        self.assertEqual(settings.real["max_trade_notional"], 0.0)
        self.assertEqual(settings.real["max_total_open_notional"], 0.0)
        self.assertEqual(settings.real["max_notional_per_market"], 0.0)
        self.assertEqual(settings.real["max_notional_per_expiry"], 0.0)
        self.assertEqual(settings.real["max_same_side_notional_per_expiry"], 0.0)
        self.assertEqual(settings.real["max_trade_notional_fraction"], 0.0)
        self.assertEqual(settings.real["max_total_open_notional_fraction"], 0.45)
        self.assertEqual(settings.real["max_notional_per_market_fraction"], 0.35)
        self.assertEqual(settings.real["max_notional_per_expiry_fraction"], 0.45)
        self.assertEqual(settings.real["max_same_side_notional_per_expiry_fraction"], 0.45)
        self.assertEqual(settings.real["replace_order_timeout_seconds"], 12)
        self.assertEqual(settings.real["cancel_order_timeout_seconds"], 45)
        self.assertEqual(settings.real["keep_competitive_order_timeout_seconds"], 60)
        self.assertEqual(settings.real["max_replace_attempts"], 2)
        self.assertEqual(settings.real["max_chase_cents"], 3)
        self.assertFalse(settings.real["skip_cycle_after_cancel"])
        self.assertEqual(settings.real["cycle_spend_fraction"], 0.25)
        self.assertEqual(settings.real["max_daily_loss_fraction"], 0.25)
        self.assertEqual(settings.real["series_ledger_paths"]["KXBTC15M"], "data/real_trading_ledger_kxbtc15m.json")
        self.assertEqual(settings.real["series_kill_switch_paths"]["KXBTCD"], "data/real_trading_kill_switch_kxbtcd.json")

    def test_resolve_start_of_day_balance_resets_for_large_same_day_deposit(self):
        sod_balance, updated = _resolve_start_of_day_balance(
            sod_data={"date": "2026-04-04", "balance": 1.23},
            current_equity=31.23,
            today="2026-04-04",
        )
        self.assertTrue(updated)
        self.assertEqual(sod_balance, 31.23)

    def test_resolve_start_of_day_balance_keeps_existing_balance_without_large_jump(self):
        sod_balance, updated = _resolve_start_of_day_balance(
            sod_data={"date": "2026-04-04", "balance": 30.0},
            current_equity=29.5,
            today="2026-04-04",
        )
        self.assertFalse(updated)
        self.assertEqual(sod_balance, 30.0)

    def test_split_snapshots_by_fraction_creates_non_empty_train_and_test_windows(self):
        base = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        snapshots = [
            MarketSnapshot(
                source="test",
                series_ticker="KXBTC15M",
                market_ticker=f"KXBTC15M-{idx}",
                contract_type="direction",
                underlying_symbol="BTC-USD",
                observed_at=base.replace(minute=idx),
                expiry=base.replace(minute=idx + 1),
                spot_price=68000.0 + idx,
            )
            for idx in range(5)
        ]
        train, test, cutoff = split_snapshots_by_fraction(snapshots, 0.6)
        self.assertTrue(train)
        self.assertTrue(test)
        self.assertIsNotNone(cutoff)
        self.assertLess(max(s.observed_at for s in train), min(s.observed_at for s in test))


if __name__ == "__main__":
    unittest.main()
