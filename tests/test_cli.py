import unittest

from kalshi_btc_bot.cli import build_parser


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

    def test_replay_grid_search_command_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["replay-grid-search", "--top", "5"])
        self.assertEqual(args.command, "replay-grid-search")
        self.assertEqual(args.top, 5)

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


if __name__ == "__main__":
    unittest.main()
