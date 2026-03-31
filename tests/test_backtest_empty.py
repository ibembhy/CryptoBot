import unittest

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.trading.exits import ExitConfig


class EmptyBacktestTests(unittest.TestCase):
    def test_run_strategy_handles_empty_snapshots(self):
        engine = BacktestEngine(
            model=GBMThresholdModel(),
            signal_config=SignalConfig(0.05, 0.0, 5, 95),
            exit_config=ExitConfig(8, 10, 2, 5),
            backtest_config=BacktestConfig(1, 0, 0, 0.0, 1000.0),
        )
        result = engine.run_strategy("early_exit", [], pd.DataFrame())
        self.assertEqual(result.summary["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()
