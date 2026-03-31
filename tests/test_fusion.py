from __future__ import annotations

import unittest

from kalshi_btc_bot.signals.fusion import FusionConfig, fuse_signals
from kalshi_btc_bot.types import TradingSignal


class FusionTests(unittest.TestCase):
    def _signal(self, *, action: str, side: str | None, edge: float, reason: str = "ok") -> TradingSignal:
        return TradingSignal(
            market_ticker="KXBTCD-TEST",
            action=action,  # type: ignore[arg-type]
            side=side,  # type: ignore[arg-type]
            raw_model_probability=0.6,
            model_probability=0.58,
            market_probability=0.4,
            edge=edge,
            raw_edge=edge,
            entry_price_cents=40 if side else None,
            fair_value_cents=58 if side else None,
            expected_value_cents=18 if side else None,
            reason=reason,
            confidence=0.58,
            quality_score=edge,
            spread_cents=2,
            liquidity_penalty=0.0,
        )

    def test_hybrid_requires_side_agreement(self):
        config = FusionConfig("hybrid", "latency_repricing", "gbm_threshold", 0.6, 0.4, True, 0.04, False)
        result = fuse_signals(
            {
                "latency_repricing": self._signal(action="buy_yes", side="yes", edge=0.12),
                "gbm_threshold": self._signal(action="buy_no", side="no", edge=0.11),
            },
            config=config,
        )
        self.assertEqual(result.action, "no_action")
        self.assertIn("disagree", result.reason.lower())

    def test_hybrid_combines_confirmed_signals(self):
        config = FusionConfig("hybrid", "latency_repricing", "gbm_threshold", 0.6, 0.4, True, 0.04, False)
        result = fuse_signals(
            {
                "latency_repricing": self._signal(action="buy_yes", side="yes", edge=0.12),
                "gbm_threshold": self._signal(action="buy_yes", side="yes", edge=0.08),
            },
            config=config,
        )
        self.assertEqual(result.action, "buy_yes")
        self.assertGreater(result.edge or 0, 0.04)


if __name__ == "__main__":
    unittest.main()
