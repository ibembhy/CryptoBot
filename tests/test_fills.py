from __future__ import annotations

import unittest

from kalshi_btc_bot.backtest.fills import apply_entry_fill, apply_exit_fill, estimate_dynamic_slippage_cents


class FillModelTests(unittest.TestCase):
    def test_dynamic_slippage_increases_for_wide_thin_markets(self):
        tight = estimate_dynamic_slippage_cents(
            1,
            spread_cents=2,
            volume=1000.0,
            open_interest=500.0,
            contracts=1,
        )
        thin = estimate_dynamic_slippage_cents(
            1,
            spread_cents=8,
            volume=20.0,
            open_interest=10.0,
            contracts=1,
        )
        self.assertGreater(thin, tight)

    def test_entry_and_exit_fills_include_contract_level_fees(self):
        entry = apply_entry_fill(
            40,
            1,
            100.0,
            spread_cents=6,
            volume=50.0,
            open_interest=25.0,
            contracts=2,
        )
        exit_fill = apply_exit_fill(
            55,
            1,
            100.0,
            spread_cents=6,
            volume=50.0,
            open_interest=25.0,
            contracts=2,
        )
        self.assertGreater(entry.price_cents, 40)
        self.assertLess(exit_fill.price_cents, 55)
        self.assertGreater(entry.fees_paid, 0.0)
        self.assertGreater(exit_fill.fees_paid, 0.0)


if __name__ == "__main__":
    unittest.main()
