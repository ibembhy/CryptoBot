from __future__ import annotations

import unittest

import pandas as pd

from kalshi_btc_bot.backtest.bankroll import BankrollSizingConfig, simulate_bankroll_constrained_compounding


class BankrollSimulatorTests(unittest.TestCase):
    def test_simulator_scales_one_contract_trades_when_affordable(self):
        trades = pd.DataFrame(
            [
                {
                    "entry_time": pd.Timestamp("2026-04-01T00:00:00Z"),
                    "market_ticker": "A",
                    "side": "yes",
                    "entry_price_cents": 50,
                    "contracts": 1,
                    "entry_notional": 0.5,
                    "realized_pnl": 0.1,
                },
                {
                    "entry_time": pd.Timestamp("2026-04-01T00:05:00Z"),
                    "market_ticker": "B",
                    "side": "yes",
                    "entry_price_cents": 40,
                    "contracts": 1,
                    "entry_notional": 0.4,
                    "realized_pnl": -0.05,
                },
            ]
        )
        simulated, summary = simulate_bankroll_constrained_compounding(
            trades,
            config=BankrollSizingConfig(starting_bankroll=100.0, max_contracts_per_trade=1),
        )
        self.assertEqual(summary["taken_trade_count"], 2)
        self.assertAlmostEqual(summary["ending_bankroll"], 100.05)
        self.assertTrue(all(simulated["simulated_taken"]))

    def test_simulator_skips_when_bankroll_too_small(self):
        trades = pd.DataFrame(
            [
                {
                    "entry_time": pd.Timestamp("2026-04-01T00:00:00Z"),
                    "market_ticker": "A",
                    "side": "yes",
                    "entry_price_cents": 50,
                    "contracts": 1,
                    "entry_notional": 0.5,
                    "realized_pnl": 0.1,
                }
            ]
        )
        simulated, summary = simulate_bankroll_constrained_compounding(
            trades,
            config=BankrollSizingConfig(starting_bankroll=0.1, max_contracts_per_trade=1),
        )
        self.assertEqual(summary["taken_trade_count"], 0)
        self.assertEqual(summary["skipped_trade_count"], 1)
        self.assertFalse(bool(simulated.iloc[0]["simulated_taken"]))

    def test_simulator_respects_fractional_bankroll_budget(self):
        trades = pd.DataFrame(
            [
                {
                    "entry_time": pd.Timestamp("2026-04-01T00:00:00Z"),
                    "market_ticker": "A",
                    "side": "yes",
                    "entry_price_cents": 50,
                    "contracts": 2,
                    "entry_notional": 1.0,
                    "realized_pnl": 0.2,
                }
            ]
        )
        simulated, summary = simulate_bankroll_constrained_compounding(
            trades,
            config=BankrollSizingConfig(
                starting_bankroll=1.0,
                bankroll_fraction_per_trade=0.5,
                max_contracts_per_trade=2,
                allow_fractional_contracts=False,
            ),
        )
        self.assertEqual(summary["taken_trade_count"], 1)
        self.assertEqual(float(simulated.iloc[0]["simulated_contracts"]), 1.0)
        self.assertAlmostEqual(summary["ending_bankroll"], 1.1)


if __name__ == "__main__":
    unittest.main()
