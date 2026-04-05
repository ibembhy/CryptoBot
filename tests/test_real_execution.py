from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

from kalshi_btc_bot.trading.real_execution import RealExecutionConfig, RealOrderExecutor, RealOrderRequest


class _FakeKalshiClient:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.resting_orders: list[dict] = []
        self.positions: list[dict] = []
        self.fills: list[dict] = []
        self.balance_cents: int = 0
        self.orders_by_id: dict[str, dict] = {}
        self.cancelled_order_ids: list[str] = []
        self.market_by_ticker: dict[str, dict] = {}
        self._next_order_number = 1
        self.list_orders_calls = 0
        self.get_fills_calls = 0
        self.get_positions_calls = 0

    def create_order(self, payload: dict):
        self.payloads.append(payload)
        order_id = f"test-order-{self._next_order_number}"
        self._next_order_number += 1
        order = {"order_id": order_id, "status": "resting", **payload}
        self.orders_by_id[order_id] = order
        return {"order": order, "echo": payload}

    def list_orders(self, **kwargs):
        self.list_orders_calls += 1
        if self.orders_by_id:
            return {"orders": list(self.orders_by_id.values())}
        return {"orders": list(self.resting_orders)}

    def get_order(self, order_id: str):
        return {"order": self.orders_by_id[order_id]}

    def get_positions(self):
        self.get_positions_calls += 1
        return {"market_positions": list(self.positions)}

    def get_fills(self, **kwargs):
        self.get_fills_calls += 1
        return {"fills": list(self.fills)}

    def get_balance(self):
        return {"balance": self.balance_cents}

    def cancel_order(self, order_id: str):
        self.cancelled_order_ids.append(order_id)
        if order_id in self.orders_by_id:
            self.orders_by_id[order_id]["status"] = "cancel_requested"
        return {"order_id": order_id, "status": "cancel_requested"}

    def get_market(self, ticker: str):
        market = self.market_by_ticker.get(ticker)
        if market is None:
            market = {"ticker": ticker}
        return {"market": market}


class RealExecutionTests(unittest.TestCase):
    def test_preview_order_does_not_submit(self):
        executor = RealOrderExecutor(
            kalshi_client=_FakeKalshiClient(),
            config=RealExecutionConfig(
                ledger_path="test_artifacts/real_exec_preview.json",
                kill_switch_path="test_artifacts/real_exec_preview_kill.json",
                dry_run=True,
            ),
        )
        preview = executor.preview_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=47,
            )
        )
        self.assertEqual(preview["mode"], "dry_run")
        self.assertEqual(preview["request"]["ticker"], "KXBTC15M-TEST")

    def test_submit_order_in_dry_run_writes_ledger_without_api_call(self):
        ledger = Path("test_artifacts/real_exec_dry_run.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_dry_run_kill.json",
                dry_run=True,
            ),
        )
        result = executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=49,
            )
        )
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(client.payloads, [])
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["request"]["yes_price"], 49)

    def test_submit_order_live_uses_client(self):
        ledger = Path("test_artifacts/real_exec_live.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_live_kill.json",
                dry_run=False,
            ),
        )
        result = executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=2,
                yes_price_cents=51,
                client_order_id="abc-123",
            )
        )
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(result["lifecycle_state"], "submitted")
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(client.payloads[0]["client_order_id"], "abc-123")

    def test_kill_switch_blocks_submission(self):
        ledger = Path("test_artifacts/real_exec_kill.json")
        kill = Path("test_artifacts/real_exec_kill_switch.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(ledger_path=str(ledger), kill_switch_path=str(kill), dry_run=False),
        )
        executor.set_kill_switch(True, reason="test")
        result = executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(client.payloads, [])

    def test_preflight_blocks_when_resting_orders_exceed_cap(self):
        ledger = Path("test_artifacts/real_exec_resting.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.resting_orders = [{"order_id": "1", "status": "resting"}, {"order_id": "2", "status": "resting"}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_resting_kill.json",
                dry_run=True,
                max_open_orders=1,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("resting-order cap", preflight.reason.lower())

    def test_preflight_blocks_when_same_market_position_uses_position_fp(self):
        ledger = Path("test_artifacts/real_exec_position_fp_block.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"ticker": "KXBTC15M-TEST", "position_fp": "2.00"}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_position_fp_block_kill.json",
                dry_run=True,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("open position cap", preflight.reason.lower())

    def test_preflight_blocks_when_trade_notional_exceeds_cap(self):
        ledger = Path("test_artifacts/real_exec_notional_cap.json")
        if ledger.exists():
            ledger.unlink()
        executor = RealOrderExecutor(
            kalshi_client=_FakeKalshiClient(),
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_notional_cap_kill.json",
                dry_run=True,
                max_trade_notional=5.0,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=20,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("notional cap", preflight.reason.lower())

    def test_preflight_blocks_when_total_open_notional_exceeds_cap(self):
        ledger = Path("test_artifacts/real_exec_total_open_notional_cap.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"ticker": "KXBTC15M-TEST-A", "position_fp": "5.00", "market_exposure_dollars": "2.50"}]
        ledger.write_text(
            json.dumps(
                [
                    {
                        "recorded_at": "2026-04-03T20:00:00+00:00",
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "request": {"ticker": "KXBTC15M-TEST-A", "side": "yes", "action": "buy", "count": 5, "yes_price": 50},
                        "filled_count": 5,
                        "avg_fill_price_cents": 50,
                    }
                ]
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_total_open_notional_cap_kill.json",
                dry_run=True,
                max_open_positions=0,
                max_total_open_notional=4.0,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST-B",
                side="yes",
                action="buy",
                count=4,
                yes_price_cents=50,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("total open notional cap", preflight.reason.lower())

    def test_preflight_blocks_when_total_open_notional_fraction_exceeds_cap(self):
        ledger = Path("test_artifacts/real_exec_total_open_notional_fraction_cap.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"ticker": "KXBTC15M-TEST-A", "position_fp": "5.00", "market_exposure_dollars": "2.50"}]
        ledger.write_text(
            json.dumps(
                [
                    {
                        "recorded_at": "2026-04-03T20:00:00+00:00",
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "request": {"ticker": "KXBTC15M-TEST-A", "side": "yes", "action": "buy", "count": 5, "yes_price": 50},
                        "filled_count": 5,
                        "avg_fill_price_cents": 50,
                    }
                ]
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_total_open_notional_fraction_cap_kill.json",
                dry_run=True,
                max_open_positions=0,
                max_total_open_notional_fraction=0.25,
            ),
        )
        executor.set_current_equity(10.0)
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST-B",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=25,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("total open notional cap", preflight.reason.lower())

    def test_preflight_blocks_when_same_expiry_side_notional_exceeds_cap(self):
        ledger = Path("test_artifacts/real_exec_same_expiry_cap.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"ticker": "KXBTC15M-26APR032230-25", "position_fp": "5.00", "market_exposure_dollars": "2.50"}]
        ledger.write_text(
            json.dumps(
                [
                    {
                        "recorded_at": "2026-04-03T20:00:00+00:00",
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "request": {"ticker": "KXBTC15M-26APR032230-25", "side": "yes", "action": "buy", "count": 5, "yes_price": 50},
                        "filled_count": 5,
                        "avg_fill_price_cents": 50,
                    }
                ]
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_same_expiry_cap_kill.json",
                dry_run=True,
                max_open_positions=0,
                max_same_side_notional_per_expiry=4.0,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-26APR032230-30",
                side="yes",
                action="buy",
                count=4,
                yes_price_cents=50,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("per-side expiry notional cap", preflight.reason.lower())

    def test_reconcile_exchange_state_enriches_submitted_entry(self):
        ledger = Path("test_artifacts/real_exec_reconcile.json")
        kill = Path("test_artifacts/real_exec_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        submit_result = executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
                client_order_id="real-test-1",
            )
        )
        self.assertEqual(submit_result["status"], "submitted")

        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "filled",
            "filled_count": 1,
            "remaining_count": 0,
            "ticker": "KXBTC15M-TEST",
        }
        client.fills = [
            {
                "order_id": "test-order-1",
                "count": 1,
                "yes_price": 50,
                "created_time": "2026-04-01T15:00:00+00:00",
            }
        ]
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 1}]

        summary = executor.reconcile_exchange_state()
        self.assertEqual(summary["ledger_entries"], 1)
        self.assertEqual(summary["updated_entries"], 1)

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["order_id"], "test-order-1")
        self.assertEqual(entries[0]["exchange_status"], "filled")
        self.assertEqual(entries[0]["filled_count"], 1)
        self.assertEqual(entries[0]["remaining_count"], 0)
        self.assertEqual(entries[0]["avg_fill_price_cents"], 50)
        self.assertTrue(entries[0]["has_open_position"])
        self.assertEqual(entries[0]["fills"][0]["order_id"], "test-order-1")
        self.assertEqual(entries[0]["lifecycle_state"], "filled")

    def test_fetch_exchange_state_reuses_cache_within_ttl(self):
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path="test_artifacts/real_exec_cache.json",
                kill_switch_path="test_artifacts/real_exec_cache_kill.json",
                dry_run=True,
            ),
        )

        executor.fetch_exchange_state(order_limit=200, fill_limit=200)
        executor.fetch_exchange_state(order_limit=200, fill_limit=50)

        self.assertEqual(client.list_orders_calls, 1)
        self.assertEqual(client.get_fills_calls, 1)
        self.assertEqual(client.get_positions_calls, 1)

    def test_submit_order_invalidates_exchange_state_cache(self):
        ledger = Path("test_artifacts/real_exec_cache_invalidate.json")
        kill = Path("test_artifacts/real_exec_cache_invalidate_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )

        executor.fetch_exchange_state(order_limit=200, fill_limit=200)
        self.assertEqual(client.list_orders_calls, 1)

        executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )

        executor.fetch_exchange_state(order_limit=200, fill_limit=200)
        self.assertEqual(client.list_orders_calls, 2)
        self.assertEqual(client.get_fills_calls, 2)
        self.assertEqual(client.get_positions_calls, 2)

    def test_reconcile_exchange_state_marks_partial_fill(self):
        ledger = Path("test_artifacts/real_exec_partial_reconcile.json")
        kill = Path("test_artifacts/real_exec_partial_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=2,
                yes_price_cents=51,
            )
        )
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "resting",
            "filled_count": 1,
            "remaining_count": 1,
            "ticker": "KXBTC15M-TEST",
        }
        client.fills = [
            {
                "order_id": "test-order-1",
                "count": 1,
                "yes_price": 51,
                "created_time": "2026-04-01T15:00:00+00:00",
            }
        ]
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 1}]

        executor.reconcile_exchange_state()
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["filled_count"], 1)
        self.assertEqual(entries[0]["remaining_count"], 1)
        self.assertEqual(entries[0]["lifecycle_state"], "partially_filled")

    def test_reconcile_exchange_state_marks_open_position_when_positions_use_position_fp(self):
        ledger = Path("test_artifacts/real_exec_position_fp_reconcile.json")
        kill = Path("test_artifacts/real_exec_position_fp_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=52,
            )
        )
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "filled",
            "ticker": "KXBTC15M-TEST",
        }
        client.fills = [{"order_id": "test-order-1", "count_fp": "1.00", "yes_price_dollars": "0.5200"}]
        client.positions = [{"ticker": "KXBTC15M-TEST", "position_fp": "1.00", "total_traded_dollars": "0.520000"}]

        executor.reconcile_exchange_state()
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertTrue(entries[0]["has_open_position"])
        self.assertEqual(entries[0]["exchange_position_count"], 1.0)

    def test_reconcile_exchange_state_clears_stale_exchange_position_when_closed(self):
        ledger = Path("test_artifacts/real_exec_position_clear_reconcile.json")
        kill = Path("test_artifacts/real_exec_position_clear_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "filled_count": 1,
                        "has_open_position": True,
                        "exchange_position": {"ticker": "KXBTC15M-TEST", "position_fp": "1.00"},
                        "exchange_position_count": 1.0,
                        "request": {"ticker": "KXBTC15M-TEST", "action": "buy", "side": "yes", "count": 1, "yes_price": 52},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )

        executor.reconcile_exchange_state()
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertFalse(entries[0]["has_open_position"])
        self.assertIsNone(entries[0]["exchange_position"])
        self.assertEqual(entries[0]["exchange_position_count"], 0.0)

    def test_reconcile_exchange_state_uses_no_side_fill_price_for_no_orders(self):
        ledger = Path("test_artifacts/real_exec_no_fill_price_reconcile.json")
        kill = Path("test_artifacts/real_exec_no_fill_price_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="no",
                action="buy",
                count=1,
                no_price_cents=70,
            )
        )
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "filled",
            "filled_count": 1,
            "remaining_count": 0,
            "ticker": "KXBTC15M-TEST",
        }
        client.fills = [
            {
                "order_id": "test-order-1",
                "count_fp": "1.00",
                "yes_price_dollars": "0.3000",
                "no_price_dollars": "0.7000",
                "created_time": "2026-04-01T15:00:00+00:00",
            }
        ]
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position_fp": "1.00", "total_traded_dollars": "0.700000"}]

        executor.reconcile_exchange_state()
        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["avg_fill_price_cents"], 70)

    def test_open_position_views_use_no_side_fill_price_for_entry_price(self):
        ledger = Path("test_artifacts/real_exec_no_open_position_view_price.json")
        kill = Path("test_artifacts/real_exec_no_open_position_view_price_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "filled_count": 1,
                        "avg_fill_price_cents": 70,
                        "request": {"ticker": "KXBTC15M-TEST", "action": "buy", "side": "no", "count": 1, "no_price": 70},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.positions = [{"ticker": "KXBTC15M-TEST", "position_fp": "1.00", "total_traded_dollars": "0.700000"}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        views = executor.open_position_views()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["side"], "no")
        self.assertEqual(views[0]["entry_price_cents"], 70)

    def test_reconcile_empty_ledger_is_safe(self):
        ledger = Path("test_artifacts/real_exec_empty_reconcile.json")
        kill = Path("test_artifacts/real_exec_empty_reconcile_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        executor = RealOrderExecutor(
            kalshi_client=_FakeKalshiClient(),
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        summary = executor.reconcile_exchange_state()
        self.assertEqual(summary["ledger_entries"], 0)
        self.assertEqual(summary["updated_entries"], 0)

    def test_open_position_views_uses_exchange_positions_even_if_ledger_flag_is_false(self):
        ledger = Path("test_artifacts/real_exec_open_position_views.json")
        kill = Path("test_artifacts/real_exec_open_position_views_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "lifecycle_state": "filled",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "filled_count": 1,
                        "has_open_position": False,
                        "request": {"ticker": "KXBTC15M-TEST", "action": "buy", "side": "yes", "count": 1, "yes_price": 52},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.positions = [{"ticker": "KXBTC15M-TEST", "position_fp": "1.00", "total_traded_dollars": "0.520000"}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        views = executor.open_position_views()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["market_ticker"], "KXBTC15M-TEST")
        self.assertEqual(views[0]["contracts"], 1)
        self.assertEqual(views[0]["entry_price_cents"], 52)

    def test_cancel_stale_orders_marks_and_cancels_resting_order(self):
        ledger = Path("test_artifacts/real_exec_cancel_stale.json")
        kill = Path("test_artifacts/real_exec_cancel_stale_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                stale_order_timeout_seconds=60,
            ),
        )
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "request": {"ticker": "KXBTC15M-TEST"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders(max_age_seconds=1)
        self.assertEqual(summary["cancelled_count"], 1)
        self.assertEqual(client.cancelled_order_ids, ["test-order-1"])

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["cancel_status"], "cancel_requested")
        self.assertEqual(entries[0]["cancel_response"]["order_id"], "test-order-1")

    def test_cancel_stale_orders_replaces_resting_buy_when_market_moves_up_within_chase_limit(self):
        ledger = Path("test_artifacts/real_exec_replace_stale.json")
        kill = Path("test_artifacts/real_exec_replace_stale_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.54, "yes_ask": 0.55}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                stale_order_timeout_seconds=10,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=30,
                max_replace_attempts=1,
                max_chase_cents=2,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=12)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 53,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 2, "yes_price": 53, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders(max_age_seconds=1)
        self.assertEqual(summary["replaced_count"], 1)
        self.assertEqual(summary["cancelled_count"], 0)
        self.assertEqual(client.cancelled_order_ids, ["test-order-1"])
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(client.payloads[0]["yes_price"], 55)

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["lifecycle_state"], "replace_requested")
        self.assertEqual(entries[1]["status"], "submitted")
        self.assertEqual(entries[1]["request"]["yes_price"], 55)
        self.assertEqual(entries[1]["replacement_count"], 1)
        self.assertEqual(entries[1]["replaced_order_id"], "test-order-1")

    def test_cancel_stale_orders_keeps_resting_buy_when_still_competitive(self):
        ledger = Path("test_artifacts/real_exec_keep_competitive.json")
        kill = Path("test_artifacts/real_exec_keep_competitive_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.53, "yes_ask": 0.54}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                stale_order_timeout_seconds=10,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=30,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=12)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 53,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 1, "yes_price": 53, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders(max_age_seconds=1)
        self.assertEqual(summary["replaced_count"], 0)
        self.assertEqual(summary["cancelled_count"], 0)
        self.assertEqual(client.cancelled_order_ids, [])
        self.assertEqual(client.payloads, [])

    def test_cancel_stale_orders_keeps_competitive_order_past_cancel_age_until_keepalive_limit(self):
        ledger = Path("test_artifacts/real_exec_keepalive_competitive.json")
        kill = Path("test_artifacts/real_exec_keepalive_competitive_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.53, "yes_ask": 0.54}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=30,
                keep_competitive_order_timeout_seconds=60,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 53,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 1, "yes_price": 53, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders()
        self.assertEqual(summary["replaced_count"], 0)
        self.assertEqual(summary["cancelled_count"], 0)
        self.assertEqual(client.cancelled_order_ids, [])

    def test_cancel_stale_orders_cancels_competitive_order_after_keepalive_limit(self):
        ledger = Path("test_artifacts/real_exec_cancel_after_keepalive.json")
        kill = Path("test_artifacts/real_exec_cancel_after_keepalive_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.53, "yes_ask": 0.54}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=30,
                keep_competitive_order_timeout_seconds=60,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=70)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 53,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 1, "yes_price": 53, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders()
        self.assertEqual(summary["cancelled_count"], 0)
        self.assertEqual(client.cancelled_order_ids, [])

    def test_cancel_stale_orders_cancels_buy_when_resting_price_is_above_mid_guard(self):
        ledger = Path("test_artifacts/real_exec_cancel_over_mid.json")
        kill = Path("test_artifacts/real_exec_cancel_over_mid_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.53, "yes_ask": 0.55}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=45,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=12)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 57,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 1, "yes_price": 57, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders()
        self.assertEqual(summary["replaced_count"], 0)
        self.assertEqual(summary["cancelled_count"], 1)
        self.assertEqual(client.cancelled_order_ids, ["test-order-1"])

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["cancel_reason"], "overpaying_vs_mid")

    def test_cancel_stale_orders_cancels_instead_of_replacing_near_expiry(self):
        ledger = Path("test_artifacts/real_exec_cancel_near_expiry.json")
        kill = Path("test_artifacts/real_exec_cancel_near_expiry_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        close_time = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        client.market_by_ticker["KXBTC15M-TEST"] = {"ticker": "KXBTC15M-TEST", "yes_bid": 0.54, "yes_ask": 0.55, "close_time": close_time}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                replace_order_timeout_seconds=10,
                cancel_order_timeout_seconds=45,
            ),
        )
        submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=12)).isoformat()
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": submitted_at,
                        "last_submitted_at": submitted_at,
                        "initial_price_cents": 53,
                        "replacement_count": 0,
                        "request": {"ticker": "KXBTC15M-TEST", "side": "yes", "action": "buy", "count": 1, "yes_price": 53, "type": "limit"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders()
        self.assertEqual(summary["replaced_count"], 0)
        self.assertEqual(summary["cancelled_count"], 1)
        self.assertEqual(client.cancelled_order_ids, ["test-order-1"])

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["cancel_reason"], "near_expiry")

    def test_cancel_stale_orders_dry_run_does_not_call_exchange_cancel(self):
        ledger = Path("test_artifacts/real_exec_cancel_stale_dry.json")
        kill = Path("test_artifacts/real_exec_cancel_stale_dry_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=True,
                stale_order_timeout_seconds=60,
            ),
        )
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "request": {"ticker": "KXBTC15M-TEST"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders(max_age_seconds=1)
        self.assertEqual(summary["cancelled_count"], 1)
        self.assertEqual(client.cancelled_order_ids, [])

        entries = json.loads(ledger.read_text(encoding="utf-8"))
        self.assertEqual(entries[0]["cancel_status"], "cancel_dry_run")

    def test_cancel_stale_orders_skips_non_resting_orders(self):
        ledger = Path("test_artifacts/real_exec_cancel_non_resting.json")
        kill = Path("test_artifacts/real_exec_cancel_non_resting_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
                stale_order_timeout_seconds=60,
            ),
        )
        ledger.write_text(
            json.dumps(
                [
                    {
                        "status": "submitted",
                        "order_id": "test-order-1",
                        "recorded_at": "2026-04-01T00:00:00+00:00",
                        "request": {"ticker": "KXBTC15M-TEST"},
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "filled", "ticker": "KXBTC15M-TEST"}

        summary = executor.cancel_stale_orders(max_age_seconds=1)
        self.assertEqual(summary["cancelled_count"], 0)
        self.assertEqual(client.cancelled_order_ids, [])

    def test_preflight_blocks_same_market_position(self):
        ledger = Path("test_artifacts/real_exec_same_market_position.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 1}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_same_market_position_kill.json",
                dry_run=True,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("position", preflight.reason.lower())

    def test_preflight_blocks_same_market_resting_order(self):
        ledger = Path("test_artifacts/real_exec_same_market_resting.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.orders_by_id["test-order-1"] = {"order_id": "test-order-1", "status": "resting", "ticker": "KXBTC15M-TEST"}
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_same_market_resting_kill.json",
                dry_run=True,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preflight.allowed)
        self.assertIn("resting order", preflight.reason.lower())

    def test_real_execution_raises_on_corrupt_ledger(self):
        ledger = Path("test_artifacts/real_exec_corrupt.json")
        kill = Path("test_artifacts/real_exec_corrupt_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        ledger.write_text("{not valid json", encoding="utf-8")
        executor = RealOrderExecutor(
            kalshi_client=_FakeKalshiClient(),
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=True,
            ),
        )
        with self.assertRaises(RuntimeError):
            executor.preview_order(
                RealOrderRequest(
                    market_ticker="KXBTC15M-TEST",
                    side="yes",
                    action="buy",
                    count=1,
                    yes_price_cents=51,
                )
            )

    def test_preflight_blocks_when_kill_switch_file_is_corrupt(self):
        ledger = Path("test_artifacts/real_exec_corrupt_kill_ledger.json")
        kill = Path("test_artifacts/real_exec_corrupt_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        kill.write_text("{not valid json", encoding="utf-8")
        executor = RealOrderExecutor(
            kalshi_client=_FakeKalshiClient(),
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=True,
            ),
        )
        preview = executor.preview_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=1,
                yes_price_cents=51,
            )
        )
        self.assertFalse(preview["preflight"]["allowed"])
        self.assertIn("failing closed", preview["preflight"]["reason"].lower())

    def test_preflight_allows_sell_when_exchange_position_exists(self):
        ledger = Path("test_artifacts/real_exec_sell_allowed.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 2}]
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_sell_allowed_kill.json",
                dry_run=True,
                max_open_positions=0,
            ),
        )
        preflight = executor.preflight_check(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="sell",
                count=2,
                yes_price_cents=60,
            )
        )
        self.assertTrue(preflight.allowed)

    def test_submit_exit_order_keeps_existing_resting_exit_when_already_aggressive_enough(self):
        ledger = Path("test_artifacts/real_exec_existing_exit_kept.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 2}]
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "resting",
            "ticker": "KXBTC15M-TEST",
            "side": "no",
            "action": "sell",
            "no_price": 12,
        }
        ledger.write_text(
            json.dumps(
                [
                    {
                        "order_id": "test-order-1",
                        "status": "submitted",
                        "lifecycle_state": "resting",
                        "is_resting": True,
                        "recorded_at": "2026-04-04T17:56:21+00:00",
                        "last_submitted_at": "2026-04-04T17:56:21+00:00",
                        "request": {"ticker": "KXBTC15M-TEST", "side": "no", "action": "sell", "count": 2, "no_price": 12},
                    }
                ]
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_existing_exit_kept_kill.json",
                dry_run=False,
            ),
        )
        result = executor.submit_exit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="no",
                action="sell",
                count=2,
                no_price_cents=15,
            )
        )
        self.assertEqual(result["status"], "already_resting")
        self.assertEqual(client.cancelled_order_ids, [])
        self.assertEqual(len(client.payloads), 0)

    def test_submit_exit_order_replaces_existing_resting_exit_when_new_price_is_more_aggressive(self):
        ledger = Path("test_artifacts/real_exec_existing_exit_replace.json")
        if ledger.exists():
            ledger.unlink()
        client = _FakeKalshiClient()
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 2}]
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "resting",
            "ticker": "KXBTC15M-TEST",
            "side": "no",
            "action": "sell",
            "no_price": 18,
        }
        ledger.write_text(
            json.dumps(
                [
                    {
                        "order_id": "test-order-1",
                        "status": "submitted",
                        "lifecycle_state": "resting",
                        "is_resting": True,
                        "recorded_at": "2026-04-04T17:56:21+00:00",
                        "last_submitted_at": "2026-04-04T17:56:21+00:00",
                        "request": {"ticker": "KXBTC15M-TEST", "side": "no", "action": "sell", "count": 2, "no_price": 18},
                    }
                ]
            ),
            encoding="utf-8",
        )
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path="test_artifacts/real_exec_existing_exit_replace_kill.json",
                dry_run=False,
            ),
        )
        result = executor.submit_exit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="no",
                action="sell",
                count=2,
                no_price_cents=12,
            )
        )
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(client.cancelled_order_ids, ["test-order-1"])
        self.assertEqual(len(client.payloads), 1)
        self.assertEqual(client.payloads[0]["no_price"], 12)

    def test_open_position_views_uses_reconciled_entry_and_exchange_position(self):
        ledger = Path("test_artifacts/real_exec_open_position_views.json")
        kill = Path("test_artifacts/real_exec_open_position_views_kill.json")
        for path in (ledger, kill):
            if path.exists():
                path.unlink()
        client = _FakeKalshiClient()
        executor = RealOrderExecutor(
            kalshi_client=client,
            config=RealExecutionConfig(
                ledger_path=str(ledger),
                kill_switch_path=str(kill),
                dry_run=False,
            ),
        )
        executor.submit_order(
            RealOrderRequest(
                market_ticker="KXBTC15M-TEST",
                side="yes",
                action="buy",
                count=2,
                yes_price_cents=51,
            )
        )
        client.orders_by_id["test-order-1"] = {
            "order_id": "test-order-1",
            "status": "filled",
            "filled_count": 2,
            "remaining_count": 0,
            "ticker": "KXBTC15M-TEST",
        }
        client.fills = [
            {
                "order_id": "test-order-1",
                "count": 2,
                "yes_price": 50,
                "created_time": "2026-04-01T15:00:00+00:00",
            }
        ]
        client.positions = [{"market_ticker": "KXBTC15M-TEST", "position": 2}]

        views = executor.open_position_views()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0]["market_ticker"], "KXBTC15M-TEST")
        self.assertEqual(views[0]["side"], "yes")
        self.assertEqual(views[0]["contracts"], 2)
        self.assertEqual(views[0]["entry_price_cents"], 50)


if __name__ == "__main__":
    unittest.main()
