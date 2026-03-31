from __future__ import annotations

from datetime import datetime, timezone
import unittest

from kalshi_btc_bot.trading.risk import RiskConfig, RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_trade_notional_cap_blocks_large_trade(self):
        manager = RiskManager(RiskConfig(10.0, 100.0, 3, 2, 20.0, 50.0))
        manager.initialize(1000.0)
        result = manager.check_entry(
            entry_notional=12.0,
            expiry=datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc),
            open_position_count=0,
            current_equity=1000.0,
        )
        self.assertFalse(result.allowed)

    def test_session_notional_and_drawdown_update(self):
        manager = RiskManager(RiskConfig(25.0, 30.0, 3, 2, 20.0, 50.0))
        expiry = datetime(2026, 3, 30, 16, 0, tzinfo=timezone.utc)
        manager.initialize(1000.0)
        manager.record_entry(entry_notional=20.0, expiry=expiry)
        manager.record_exit(realized_pnl=-15.0, expiry=expiry, current_equity=985.0)
        blocked = manager.check_entry(
            entry_notional=20.0,
            expiry=expiry,
            open_position_count=0,
            current_equity=985.0,
        )
        self.assertFalse(blocked.allowed)


if __name__ == "__main__":
    unittest.main()
