from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.utils.files import read_json_strict, write_json_atomic


@dataclass(frozen=True)
class RealExecutionConfig:
    ledger_path: str
    series_ticker: str = ""
    kill_switch_path: str = "data/real_trading_kill_switch.json"
    dry_run: bool = True
    max_order_count: int = 1
    max_daily_orders: int = 5
    max_open_orders: int = 2
    max_open_positions: int = 1
    stale_order_timeout_seconds: int = 60
    replace_order_timeout_seconds: int = 15
    cancel_order_timeout_seconds: int = 30
    keep_competitive_order_timeout_seconds: int = 60
    max_replace_attempts: int = 2
    max_chase_cents: int = 3
    max_trade_notional: float = 0.0
    max_total_open_notional: float = 0.0
    max_notional_per_market: float = 0.0
    max_notional_per_expiry: float = 0.0
    max_same_side_notional_per_expiry: float = 0.0
    max_trade_notional_fraction: float = 0.0
    max_total_open_notional_fraction: float = 0.0
    max_notional_per_market_fraction: float = 0.0
    max_notional_per_expiry_fraction: float = 0.0
    max_same_side_notional_per_expiry_fraction: float = 0.0


@dataclass(frozen=True)
class RealPreflightResult:
    allowed: bool
    reason: str
    context: dict[str, Any]


@dataclass(frozen=True)
class RealOrderRequest:
    market_ticker: str
    side: str
    action: str
    count: int
    yes_price_cents: int | None = None
    no_price_cents: int | None = None
    order_type: str = "limit"
    client_order_id: str | None = None
    expiration_ts: int | None = None
    entry_edge: float = 0.0

    def to_api_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ticker": self.market_ticker,
            "action": self.action,
            "side": self.side,
            "count": self.count,
            "type": self.order_type,
        }
        if self.yes_price_cents is not None:
            payload["yes_price"] = self.yes_price_cents
        if self.no_price_cents is not None:
            payload["no_price"] = self.no_price_cents
        if self.client_order_id:
            payload["client_order_id"] = self.client_order_id
        if self.expiration_ts is not None:
            payload["expiration_ts"] = self.expiration_ts
        return payload


class RealOrderExecutor:
    def __init__(
        self,
        *,
        kalshi_client: KalshiClient,
        config: RealExecutionConfig,
    ) -> None:
        self.kalshi_client = kalshi_client
        self.config = config
        self.ledger_path = Path(config.ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.kill_switch_path = Path(config.kill_switch_path)
        self.kill_switch_path.parent.mkdir(parents=True, exist_ok=True)
        self._exchange_state_cache: dict[str, Any] | None = None
        self._exchange_state_cache_fetched_at: datetime | None = None
        self._exchange_state_cache_limits: tuple[int, int] = (0, 0)
        self._exchange_state_cache_ttl = timedelta(seconds=3)
        self._current_equity: float | None = None

    def set_current_equity(self, current_equity: float | None) -> None:
        try:
            self._current_equity = None if current_equity is None else max(float(current_equity), 0.0)
        except (TypeError, ValueError):
            self._current_equity = None

    def preview_order(self, request: RealOrderRequest) -> dict[str, Any]:
        preflight = self.preflight_check(request)
        return {
            "mode": "dry_run" if self.config.dry_run else "live_ready",
            "series_ticker": self.config.series_ticker,
            "request": request.to_api_payload(),
            "preflight": {
                "allowed": preflight.allowed,
                "reason": preflight.reason,
                "context": preflight.context,
            },
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def submit_order(self, request: RealOrderRequest) -> dict[str, Any]:
        preview = self.preview_order(request)
        preflight = preview["preflight"]
        if not bool(preflight["allowed"]):
            result = {
                "status": "blocked",
                "lifecycle_state": "blocked",
                "series_ticker": self.config.series_ticker,
                "request": request.to_api_payload(),
                "response": None,
                "recorded_at": preview["recorded_at"],
                "reason": preflight["reason"],
                "context": preflight["context"],
            }
            self._append_ledger(result)
            return result
        if self.config.dry_run:
            result = {
                "status": "dry_run",
                "lifecycle_state": "dry_run",
                "series_ticker": self.config.series_ticker,
                "request": request.to_api_payload(),
                "response": None,
                "recorded_at": preview["recorded_at"],
                "context": preflight["context"],
            }
            self._append_ledger(result)
            return result

        response = self.kalshi_client.create_order(request.to_api_payload())
        self._invalidate_exchange_state_cache()
        order_id = self._extract_order_id(response)
        result = {
            "status": "submitted",
            "lifecycle_state": "submitted",
            "series_ticker": self.config.series_ticker,
            "request": request.to_api_payload(),
            "response": response,
            "order_id": order_id,
            "requested_count": int(request.count),
            "recorded_at": preview["recorded_at"],
            "last_submitted_at": preview["recorded_at"],
            "initial_price_cents": self._request_price_cents(request.to_api_payload()),
            "replacement_count": 0,
            "entry_edge": request.entry_edge,
            "context": preflight["context"],
        }
        self._append_ledger(result)
        return result

    def submit_exit_order(self, request: RealOrderRequest) -> dict[str, Any]:
        if str(request.action).lower() != "sell":
            return self.submit_order(request)

        ledger_entries = self._read_ledger()
        exchange_state = self.fetch_exchange_state(order_limit=200, fill_limit=50)
        same_market_positions = [
            position
            for position in exchange_state["open_positions"]
            if str(position.get("market_ticker", position.get("ticker", "")) or "") == request.market_ticker
        ]
        if not same_market_positions:
            return self.submit_order(request)

        managed = self._manage_existing_resting_exit(
            request=request,
            ledger_entries=ledger_entries,
            exchange_state=exchange_state,
        )
        if managed is not None:
            return managed
        return self.submit_order(request)

    def reconcile_exchange_state(self, *, order_limit: int = 200, fill_limit: int = 200) -> dict[str, Any]:
        ledger_entries = self._read_ledger()
        if not ledger_entries:
            return {
                "series_ticker": self.config.series_ticker,
                "ledger_entries": 0,
                "updated_entries": 0,
                "resting_orders": 0,
                "filled_orders_seen": 0,
                "open_positions": 0,
                "reconciled_at": datetime.now(timezone.utc).isoformat(),
            }

        exchange_state = self.fetch_exchange_state(order_limit=order_limit, fill_limit=fill_limit)
        orders = exchange_state["orders"]
        fills = exchange_state["fills"]
        positions = exchange_state["positions"]

        orders_by_id = {
            order_id: order
            for order in orders
            if (order_id := self._extract_identifier(order, ("order_id", "id")))
        }
        fills_by_order_id: dict[str, list[dict[str, Any]]] = {}
        for fill in fills:
            fill_order_id = self._extract_identifier(fill, ("order_id", "maker_order_id", "taker_order_id"))
            if fill_order_id:
                fills_by_order_id.setdefault(fill_order_id, []).append(fill)

        open_positions_by_ticker: dict[str, dict[str, Any]] = {}
        for position in positions:
            ticker = str(position.get("market_ticker", position.get("ticker", "")) or "")
            position_count = self._extract_position_count(position)
            if ticker and position_count != 0:
                open_positions_by_ticker[ticker] = position

        updated_entries = 0
        reconciled_at = datetime.now(timezone.utc).isoformat()
        for entry in ledger_entries:
            order_id = str(entry.get("order_id") or self._extract_order_id(entry.get("response")) or "")
            if order_id and entry.get("order_id") != order_id:
                entry["order_id"] = order_id
            order = orders_by_id.get(order_id) if order_id else None
            order_fills = fills_by_order_id.get(order_id, []) if order_id else []
            market_ticker = str(entry.get("request", {}).get("ticker", ""))
            had_changes = False

            if order is not None:
                exchange_status = self._extract_exchange_status(order)
                if entry.get("exchange_status") != exchange_status:
                    entry["exchange_status"] = exchange_status
                    had_changes = True
                if entry.get("exchange_order") != order:
                    entry["exchange_order"] = order
                    had_changes = True
                resting = bool(self._extract_boolean(order, ("is_resting", "resting"), fallback=exchange_status == "resting"))
                if entry.get("is_resting") != resting:
                    entry["is_resting"] = resting
                    had_changes = True
                for field_name, keys in (
                    ("filled_count", ("filled_count", "filled_quantity", "matched_count")),
                    ("remaining_count", ("remaining_count", "remaining_quantity", "resting_count")),
                ):
                    value = self._extract_numeric(order, keys)
                    if value is not None and entry.get(field_name) != value:
                        entry[field_name] = value
                        had_changes = True

            if order_fills:
                request_payload = entry.get("request", {}) if isinstance(entry.get("request"), dict) else {}
                request_side = str(request_payload.get("side") or "").lower()
                fill_count = sum(int(self._extract_numeric(fill, ("count", "quantity", "filled_count", "count_fp")) or 0) for fill in order_fills)
                avg_fill_price = self._average_fill_price(order_fills, preferred_side=request_side)
                latest_fill_at = max(str(fill.get("created_time", fill.get("ts", fill.get("timestamp", ""))) or "") for fill in order_fills)
                if entry.get("fill_count") != fill_count:
                    entry["fill_count"] = fill_count
                    had_changes = True
                if entry.get("filled_count") != fill_count:
                    entry["filled_count"] = fill_count
                    had_changes = True
                if avg_fill_price is not None and entry.get("avg_fill_price_cents") != avg_fill_price:
                    entry["avg_fill_price_cents"] = avg_fill_price
                    had_changes = True
                if latest_fill_at and entry.get("latest_fill_at") != latest_fill_at:
                    entry["latest_fill_at"] = latest_fill_at
                    had_changes = True
                if entry.get("fills") != order_fills:
                    entry["fills"] = order_fills
                    had_changes = True

            has_open_position = market_ticker in open_positions_by_ticker if market_ticker else False
            if entry.get("has_open_position") != has_open_position:
                entry["has_open_position"] = has_open_position
                had_changes = True
            if has_open_position:
                position_snapshot = open_positions_by_ticker[market_ticker]
                if entry.get("exchange_position") != position_snapshot:
                    entry["exchange_position"] = position_snapshot
                    had_changes = True
                position_count = self._extract_position_count(position_snapshot)
                if entry.get("exchange_position_count") != position_count:
                    entry["exchange_position_count"] = position_count
                    had_changes = True
            else:
                if entry.get("exchange_position") is not None:
                    entry["exchange_position"] = None
                    had_changes = True
                stale_position_count = entry.get("exchange_position_count")
                if stale_position_count not in (None, 0, 0.0):
                    entry["exchange_position_count"] = 0.0
                    had_changes = True

            lifecycle_state = self._resolve_lifecycle_state(entry)
            if entry.get("lifecycle_state") != lifecycle_state:
                entry["lifecycle_state"] = lifecycle_state
                had_changes = True

            if had_changes or entry.get("reconciled_at") is None:
                entry["reconciled_at"] = reconciled_at
                updated_entries += 1

        write_json_atomic(self.ledger_path, ledger_entries)
        return {
            "series_ticker": self.config.series_ticker,
            "ledger_entries": len(ledger_entries),
            "updated_entries": updated_entries,
            "resting_orders": len(
                [order for order in orders if self._extract_exchange_status(order) == "resting"]
            ),
            "filled_orders_seen": len(fills_by_order_id),
            "open_positions": len(open_positions_by_ticker),
            "reconciled_at": reconciled_at,
        }

    def cancel_stale_orders(self, *, max_age_seconds: int | None = None) -> dict[str, Any]:
        reconcile_summary = self.reconcile_exchange_state()
        replace_age = int(max_age_seconds if max_age_seconds is not None else self.config.replace_order_timeout_seconds)
        cancel_age = max(int(self.config.cancel_order_timeout_seconds), replace_age)
        now = datetime.now(timezone.utc)
        ledger_entries = self._read_ledger()
        cancelled: list[dict[str, Any]] = []
        replaced: list[dict[str, Any]] = []

        for entry in ledger_entries:
            order_id = str(entry.get("order_id") or "")
            if not order_id:
                continue
            if not bool(entry.get("is_resting", False)):
                continue
            recorded_at = self._parse_iso_timestamp(entry.get("last_submitted_at") or entry.get("recorded_at"))
            if recorded_at is None:
                continue
            age_seconds = (now - recorded_at).total_seconds()
            if age_seconds < replace_age:
                continue

            replacement_request = None
            replacement_reason = None
            replacement_request, replacement_reason = self._build_replacement_request(entry)
            if replacement_request is not None:
                if self.config.dry_run:
                    cancel_response = {"status": "dry_run_cancel", "order_id": order_id}
                    replacement_response = {"status": "dry_run_replace", "request": replacement_request.to_api_payload()}
                    cancel_state = "cancel_dry_run"
                    replacement_status = "dry_run"
                    replacement_lifecycle = "dry_run"
                    replacement_order_id = None
                else:
                    cancel_response = self.kalshi_client.cancel_order(order_id)
                    replacement_response = self.kalshi_client.create_order(replacement_request.to_api_payload())
                    self._invalidate_exchange_state_cache()
                    cancel_state = "cancel_requested"
                    replacement_status = "submitted"
                    replacement_lifecycle = "submitted"
                    replacement_order_id = self._extract_order_id(replacement_response)

                entry["cancel_attempted_at"] = now.isoformat()
                entry["cancel_status"] = cancel_state
                entry["cancel_response"] = cancel_response
                entry["replacement_attempted_at"] = now.isoformat()
                entry["replacement_reason"] = replacement_reason
                entry["replacement_request"] = replacement_request.to_api_payload()
                entry["replacement_order_id"] = replacement_order_id
                entry["lifecycle_state"] = "replace_requested" if cancel_state == "cancel_requested" else "replace_dry_run"
                entry["reconciled_at"] = now.isoformat()

                replacement_entry = {
                    "status": replacement_status,
                    "lifecycle_state": replacement_lifecycle,
                    "series_ticker": self.config.series_ticker,
                    "request": replacement_request.to_api_payload(),
                    "response": replacement_response,
                    "order_id": replacement_order_id,
                    "requested_count": int(replacement_request.count),
                    "recorded_at": now.isoformat(),
                    "last_submitted_at": now.isoformat(),
                    "initial_price_cents": int(entry.get("initial_price_cents") or self._request_price_cents(entry.get("request", {})) or self._request_price_cents(replacement_request.to_api_payload()) or 0),
                    "replacement_count": int(entry.get("replacement_count") or 0) + 1,
                    "replaced_order_id": order_id,
                    "replacement_root_order_id": str(entry.get("replacement_root_order_id") or order_id),
                    "context": entry.get("context", {}),
                }
                ledger_entries.append(replacement_entry)
                replaced.append(
                    {
                        "old_order_id": order_id,
                        "new_order_id": replacement_order_id,
                        "market_ticker": entry.get("request", {}).get("ticker"),
                        "age_seconds": int(age_seconds),
                        "replacement_reason": replacement_reason,
                        "new_request": replacement_request.to_api_payload(),
                    }
                )
                continue
            if replacement_reason == "still_competitive":
                continue
            if age_seconds < cancel_age and replacement_reason not in {
                "overpaying_vs_mid",
                "underpricing_vs_mid",
                "near_expiry",
                "max_replacements_reached",
                "chase_limit_exceeded",
            }:
                continue

            if replacement_reason:
                entry["cancel_reason"] = replacement_reason

            if self.config.dry_run:
                cancel_response = {"status": "dry_run_cancel", "order_id": order_id}
                cancel_state = "cancel_dry_run"
            else:
                cancel_response = self.kalshi_client.cancel_order(order_id)
                self._invalidate_exchange_state_cache()
                cancel_state = "cancel_requested"

            entry["cancel_attempted_at"] = now.isoformat()
            entry["cancel_status"] = cancel_state
            entry["cancel_response"] = cancel_response
            entry["lifecycle_state"] = "cancel_requested" if cancel_state == "cancel_requested" else "cancel_dry_run"
            entry["reconciled_at"] = now.isoformat()
            cancelled.append(
                {
                    "order_id": order_id,
                    "market_ticker": entry.get("request", {}).get("ticker"),
                    "age_seconds": int(age_seconds),
                    "cancel_status": cancel_state,
                }
            )

        write_json_atomic(self.ledger_path, ledger_entries)
        return {
            "reconcile": reconcile_summary,
            "checked_entries": len(ledger_entries),
            "replace_order_timeout_seconds": replace_age,
            "cancel_order_timeout_seconds": cancel_age,
            "keep_competitive_order_timeout_seconds": int(self.config.keep_competitive_order_timeout_seconds),
            "stale_order_timeout_seconds": replace_age,
            "replaced_orders": replaced,
            "replaced_count": len(replaced),
            "cancelled_orders": cancelled,
            "cancelled_count": len(cancelled),
            "performed_at": now.isoformat(),
        }

    def preflight_check(self, request: RealOrderRequest) -> RealPreflightResult:
        try:
            kill_switch_enabled = self.is_kill_switch_enabled()
        except RuntimeError as exc:
            return RealPreflightResult(
                False,
                "Kill switch state is unreadable; failing closed.",
                {"kill_switch": "unreadable", "error": str(exc)},
            )
        if kill_switch_enabled:
            return RealPreflightResult(False, "Kill switch is enabled.", {"kill_switch": True})

        ledger_entries = self._read_ledger()
        now = datetime.now(timezone.utc)
        today_prefix = now.date().isoformat()
        todays_entries = [
            entry for entry in ledger_entries
            if str(entry.get("recorded_at", "")).startswith(today_prefix)
            and entry.get("status") == "submitted"
        ]

        exchange_state = self.fetch_exchange_state(order_limit=200, fill_limit=50)
        resting_orders = exchange_state["resting_orders"]
        open_positions = exchange_state["open_positions"]
        open_position_views = self._build_open_position_views(exchange_state=exchange_state, ledger_entries=ledger_entries)
        same_market_resting = [order for order in resting_orders if str(order.get("ticker", order.get("market_ticker", "")) or "") == request.market_ticker]
        same_market_positions = [
            position
            for position in open_positions
            if str(position.get("market_ticker", position.get("ticker", "")) or "") == request.market_ticker
        ]
        same_market_views = [view for view in open_position_views if str(view.get("market_ticker") or "") == request.market_ticker]

        context = {
            "series_ticker": self.config.series_ticker,
            "todays_recorded_orders": len(todays_entries),
            "resting_orders": len(resting_orders),
            "open_positions": len(open_positions),
            "same_market_resting_orders": len(same_market_resting),
            "same_market_open_positions": len(same_market_positions),
            "dry_run": self.config.dry_run,
            "action": request.action,
        }
        is_entry = str(request.action).lower() == "buy"
        is_exit = str(request.action).lower() == "sell"
        request_price_cents = self._request_price_cents(request.to_api_payload())
        request_notional = None
        if request_price_cents is not None:
            request_notional = round((float(request_price_cents) * float(request.count)) / 100.0, 4)
            context["request_notional"] = request_notional
        if is_entry and self.config.max_daily_orders > 0 and len(todays_entries) >= self.config.max_daily_orders:
            return RealPreflightResult(False, "Daily real-order cap reached.", context)
        if is_entry and self.config.max_open_orders > 0 and len(resting_orders) >= self.config.max_open_orders:
            return RealPreflightResult(False, "Open resting-order cap reached.", context)
        if is_entry and self.config.max_open_positions > 0 and len(open_positions) >= self.config.max_open_positions:
            return RealPreflightResult(False, "Open position cap reached on exchange.", context)
        effective_trade_notional_cap = self._effective_cap(
            fixed_cap=self.config.max_trade_notional,
            fractional_cap=self.config.max_trade_notional_fraction,
        )
        if effective_trade_notional_cap > 0:
            context["effective_trade_notional_cap"] = round(effective_trade_notional_cap, 4)
        if is_entry and effective_trade_notional_cap > 0 and request_notional is not None and request_notional > effective_trade_notional_cap:
            return RealPreflightResult(False, "Trade notional cap reached.", context)

        if same_market_resting:
            return RealPreflightResult(False, "Existing resting order already on this market.", context)
        if is_entry and request_notional is not None:
            market_notional = sum(self._position_view_notional(view) for view in same_market_views)
            expiry_key = self._market_event_key(request.market_ticker)
            expiry_notional = sum(
                self._position_view_notional(view)
                for view in open_position_views
                if self._market_event_key(str(view.get("market_ticker") or "")) == expiry_key
            )
            same_side_expiry_notional = sum(
                self._position_view_notional(view)
                for view in open_position_views
                if self._market_event_key(str(view.get("market_ticker") or "")) == expiry_key
                and str(view.get("side") or "").lower() == str(request.side).lower()
            )
            total_open_notional = sum(self._position_view_notional(view) for view in open_position_views)
            context.update(
                {
                    "same_market_open_notional": round(market_notional, 4),
                    "same_expiry_open_notional": round(expiry_notional, 4),
                    "same_side_expiry_open_notional": round(same_side_expiry_notional, 4),
                    "total_open_notional": round(total_open_notional, 4),
                }
            )
            effective_market_cap = self._effective_cap(
                fixed_cap=self.config.max_notional_per_market,
                fractional_cap=self.config.max_notional_per_market_fraction,
            )
            effective_expiry_cap = self._effective_cap(
                fixed_cap=self.config.max_notional_per_expiry,
                fractional_cap=self.config.max_notional_per_expiry_fraction,
            )
            effective_same_side_expiry_cap = self._effective_cap(
                fixed_cap=self.config.max_same_side_notional_per_expiry,
                fractional_cap=self.config.max_same_side_notional_per_expiry_fraction,
            )
            effective_total_cap = self._effective_cap(
                fixed_cap=self.config.max_total_open_notional,
                fractional_cap=self.config.max_total_open_notional_fraction,
            )
            if effective_market_cap > 0:
                context["effective_notional_per_market_cap"] = round(effective_market_cap, 4)
            if effective_expiry_cap > 0:
                context["effective_notional_per_expiry_cap"] = round(effective_expiry_cap, 4)
            if effective_same_side_expiry_cap > 0:
                context["effective_same_side_expiry_cap"] = round(effective_same_side_expiry_cap, 4)
            if effective_total_cap > 0:
                context["effective_total_open_notional_cap"] = round(effective_total_cap, 4)
            if effective_market_cap > 0 and market_notional + request_notional > effective_market_cap:
                return RealPreflightResult(False, "Per-market notional cap reached.", context)
            if effective_expiry_cap > 0 and expiry_notional + request_notional > effective_expiry_cap:
                return RealPreflightResult(False, "Per-expiry notional cap reached.", context)
            if (
                effective_same_side_expiry_cap > 0
                and same_side_expiry_notional + request_notional > effective_same_side_expiry_cap
            ):
                return RealPreflightResult(False, "Per-side expiry notional cap reached.", context)
            if effective_total_cap > 0 and total_open_notional + request_notional > effective_total_cap:
                return RealPreflightResult(False, "Total open notional cap reached.", context)
        # Ledger-based exposure check: block re-entry if we have filled buys with no
        # corresponding filled sells, regardless of what the positions endpoint reports.
        # This prevents the accumulation bug where has_open_position lags or is wrong.
        if is_entry:
            ledger_buy_fill_count = 0
            ledger_sell_fill_count = 0
            for entry in ledger_entries:
                entry_ticker = str(entry.get("request", {}).get("ticker", "") or "")
                if entry_ticker != request.market_ticker:
                    continue
                action = str(entry.get("request", {}).get("action", "") or "").lower()
                filled = int(entry.get("filled_count") or entry.get("fill_count") or 0)
                if action == "buy":
                    ledger_buy_fill_count += filled
                elif action == "sell":
                    ledger_sell_fill_count += filled
            if ledger_buy_fill_count > ledger_sell_fill_count and not same_market_views:
                return RealPreflightResult(False, "Ledger shows unfilled exposure on this market; blocking re-entry.", {**context, "ledger_buy_fills": ledger_buy_fill_count, "ledger_sell_fills": ledger_sell_fill_count})
            # Block re-entry on same side if already completed a LOSING round-trip on this ticker today.
            # Allows re-entry after a win (positive PnL), blocks re-entry after a loss.
            # Prevents re-entry cycling on oscillating markets (the main source of real losses).
            today_same_side_buy_fills = 0
            today_same_side_sell_fills = 0
            today_same_side_buy_price_cents: float | None = None
            today_same_side_sell_price_cents: float | None = None
            for entry in ledger_entries:
                if not str(entry.get("recorded_at", "")).startswith(today_prefix):
                    continue
                entry_ticker = str(entry.get("request", {}).get("ticker", "") or "")
                if entry_ticker != request.market_ticker:
                    continue
                action = str(entry.get("request", {}).get("action", "") or "").lower()
                side = str(entry.get("request", {}).get("side", "") or "").lower()
                filled = int(entry.get("filled_count") or entry.get("fill_count") or 0)
                if filled <= 0:
                    continue
                if action == "buy" and side == str(request.side).lower():
                    today_same_side_buy_fills += filled
                    raw_price = entry.get("avg_fill_price_cents")
                    if raw_price is not None:
                        try:
                            today_same_side_buy_price_cents = float(raw_price)
                        except (TypeError, ValueError):
                            pass
                elif action == "sell" and side == str(request.side).lower():
                    today_same_side_sell_fills += filled
                    raw_price = entry.get("avg_fill_price_cents")
                    if raw_price is not None:
                        try:
                            today_same_side_sell_price_cents = float(raw_price)
                        except (TypeError, ValueError):
                            pass
            if today_same_side_buy_fills > 0 and today_same_side_sell_fills > 0:
                # Determine if the previous round-trip was a win.
                # Win = sell price > buy price (positive PnL per contract).
                if (
                    today_same_side_buy_price_cents is not None
                    and today_same_side_sell_price_cents is not None
                    and today_same_side_sell_price_cents > today_same_side_buy_price_cents
                ):
                    pass  # Previous trade was profitable — allow re-entry
                else:
                    return RealPreflightResult(False, f"Already completed a losing {request.side} round-trip on this market today; blocking re-entry.", {**context, "today_same_side_buy_fills": today_same_side_buy_fills, "today_same_side_sell_fills": today_same_side_sell_fills, "buy_price_cents": today_same_side_buy_price_cents, "sell_price_cents": today_same_side_sell_price_cents})
        if is_exit and not same_market_positions:
            return RealPreflightResult(False, "No exchange position is open on this market to exit.", context)
        return RealPreflightResult(True, "allowed", context)

    def fetch_exchange_state(self, *, order_limit: int = 200, fill_limit: int = 200) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        cached_order_limit, cached_fill_limit = self._exchange_state_cache_limits
        if (
            self._exchange_state_cache is not None
            and self._exchange_state_cache_fetched_at is not None
            and now - self._exchange_state_cache_fetched_at <= self._exchange_state_cache_ttl
            and cached_order_limit >= order_limit
            and cached_fill_limit >= fill_limit
        ):
            return self._exchange_state_cache
        orders_payload = self.kalshi_client.list_orders(limit=order_limit)
        fills_payload = self.kalshi_client.get_fills(limit=fill_limit)
        positions_payload = self.kalshi_client.get_positions()
        orders = [
            order
            for order in self._extract_orders(orders_payload)
            if self._payload_matches_series(order, ("ticker", "market_ticker"))
        ]
        fills = [
            fill
            for fill in self._extract_fills(fills_payload)
            if self._payload_matches_series(fill, ("ticker", "market_ticker"))
        ]
        positions = [
            position
            for position in self._extract_positions(positions_payload)
            if self._payload_matches_series(position, ("market_ticker", "ticker"))
        ]
        resting_orders = [order for order in orders if self._extract_exchange_status(order) == "resting"]
        open_positions = [
            position
            for position in positions
            if self._extract_position_count(position) != 0.0
        ]
        exchange_state = {
            "orders": orders,
            "fills": fills,
            "positions": positions,
            "resting_orders": resting_orders,
            "open_positions": open_positions,
        }
        self._exchange_state_cache = exchange_state
        self._exchange_state_cache_fetched_at = now
        self._exchange_state_cache_limits = (order_limit, fill_limit)
        return exchange_state

    def is_kill_switch_enabled(self) -> bool:
        payload = read_json_strict(
            self.kill_switch_path,
            missing_default=None,
            label="Real trading kill-switch file",
        )
        if payload is None:
            return False
        return bool(payload.get("enabled", False))

    def set_kill_switch(self, enabled: bool, *, reason: str | None = None) -> dict[str, Any]:
        payload = {
            "enabled": bool(enabled),
            "reason": reason or "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(self.kill_switch_path, payload)
        return payload

    def _append_ledger(self, entry: dict[str, Any]) -> None:
        existing = self._read_ledger()
        existing.append(entry)
        write_json_atomic(self.ledger_path, existing)

    def _manage_existing_resting_exit(
        self,
        *,
        request: RealOrderRequest,
        ledger_entries: list[dict[str, Any]],
        exchange_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        same_market_resting = [
            order
            for order in exchange_state["resting_orders"]
            if str(order.get("ticker", order.get("market_ticker", "")) or "") == request.market_ticker
        ]
        if not same_market_resting:
            return None

        resting_order_ids = {
            str(self._extract_identifier(order, ("order_id", "id")) or "")
            for order in same_market_resting
        }
        matching_entries = [
            entry
            for entry in ledger_entries
            if str(entry.get("order_id") or "") in resting_order_ids
            and str(entry.get("request", {}).get("ticker") or "") == request.market_ticker
            and str(entry.get("request", {}).get("action") or "").lower() == "sell"
            and str(entry.get("request", {}).get("side") or "").lower() == str(request.side).lower()
            and bool(entry.get("is_resting", False))
        ]
        if not matching_entries:
            return None

        matching_entries.sort(
            key=lambda entry: self._parse_iso_timestamp(entry.get("last_submitted_at") or entry.get("recorded_at")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        existing_entry = matching_entries[0]
        existing_request = existing_entry.get("request", {}) if isinstance(existing_entry.get("request"), dict) else {}
        existing_price = self._request_price_cents(existing_request)
        desired_price = self._request_price_cents(request.to_api_payload())

        now = datetime.now(timezone.utc)
        existing_entry["last_exit_check_at"] = now.isoformat()

        if (
            existing_price is not None
            and desired_price is not None
            and self._request_is_at_least_as_aggressive(action="sell", existing_price=existing_price, desired_price=desired_price)
        ):
            existing_entry["exit_refresh_reason"] = "existing_resting_exit_kept"
            existing_entry["reconciled_at"] = now.isoformat()
            write_json_atomic(self.ledger_path, ledger_entries)
            return {
                "status": "already_resting",
                "lifecycle_state": str(existing_entry.get("lifecycle_state") or "resting"),
                "series_ticker": self.config.series_ticker,
                "request": existing_request,
                "response": existing_entry.get("response"),
                "order_id": existing_entry.get("order_id"),
                "recorded_at": now.isoformat(),
                "reason": "Existing resting exit already meets or beats the new exit price.",
                "context": {
                    "existing_order_id": existing_entry.get("order_id"),
                    "existing_price_cents": existing_price,
                    "desired_price_cents": desired_price,
                },
            }

        result = self._replace_resting_order_entry(
            ledger_entries=ledger_entries,
            entry=existing_entry,
            replacement_request=request,
            reason="exit_price_improved",
            now=now,
        )
        return result

    def _replace_resting_order_entry(
        self,
        *,
        ledger_entries: list[dict[str, Any]],
        entry: dict[str, Any],
        replacement_request: RealOrderRequest,
        reason: str,
        now: datetime,
    ) -> dict[str, Any]:
        order_id = str(entry.get("order_id") or "")
        if self.config.dry_run:
            cancel_response = {"status": "dry_run_cancel", "order_id": order_id}
            replacement_response = {"status": "dry_run_replace", "request": replacement_request.to_api_payload()}
            cancel_state = "cancel_dry_run"
            replacement_status = "dry_run"
            replacement_lifecycle = "dry_run"
            replacement_order_id = None
        else:
            cancel_response = self.kalshi_client.cancel_order(order_id)
            replacement_response = self.kalshi_client.create_order(replacement_request.to_api_payload())
            self._invalidate_exchange_state_cache()
            cancel_state = "cancel_requested"
            replacement_status = "submitted"
            replacement_lifecycle = "submitted"
            replacement_order_id = self._extract_order_id(replacement_response)

        entry["cancel_attempted_at"] = now.isoformat()
        entry["cancel_status"] = cancel_state
        entry["cancel_response"] = cancel_response
        entry["replacement_attempted_at"] = now.isoformat()
        entry["replacement_reason"] = reason
        entry["replacement_request"] = replacement_request.to_api_payload()
        entry["replacement_order_id"] = replacement_order_id
        entry["lifecycle_state"] = "replace_requested" if cancel_state == "cancel_requested" else "replace_dry_run"
        entry["reconciled_at"] = now.isoformat()

        replacement_entry = {
            "status": replacement_status,
            "lifecycle_state": replacement_lifecycle,
            "series_ticker": self.config.series_ticker,
            "request": replacement_request.to_api_payload(),
            "response": replacement_response,
            "order_id": replacement_order_id,
            "requested_count": int(replacement_request.count),
            "recorded_at": now.isoformat(),
            "last_submitted_at": now.isoformat(),
            "initial_price_cents": int(
                entry.get("initial_price_cents")
                or self._request_price_cents(entry.get("request", {}))
                or self._request_price_cents(replacement_request.to_api_payload())
                or 0
            ),
            "replacement_count": int(entry.get("replacement_count") or 0) + 1,
            "replaced_order_id": order_id,
            "replacement_root_order_id": str(entry.get("replacement_root_order_id") or order_id),
            "context": entry.get("context", {}),
        }
        ledger_entries.append(replacement_entry)
        write_json_atomic(self.ledger_path, ledger_entries)
        return {
            "status": replacement_status,
            "lifecycle_state": replacement_lifecycle,
            "series_ticker": self.config.series_ticker,
            "request": replacement_request.to_api_payload(),
            "response": replacement_response,
            "order_id": replacement_order_id,
            "recorded_at": now.isoformat(),
            "reason": reason,
            "context": {
                "replaced_order_id": order_id,
                "replacement_reason": reason,
            },
        }

    def _invalidate_exchange_state_cache(self) -> None:
        self._exchange_state_cache = None
        self._exchange_state_cache_fetched_at = None
        self._exchange_state_cache_limits = (0, 0)

    def _effective_cap(self, *, fixed_cap: float, fractional_cap: float) -> float:
        try:
            fixed_value = float(fixed_cap or 0.0)
        except (TypeError, ValueError):
            fixed_value = 0.0
        if fixed_value > 0:
            return fixed_value
        try:
            fractional_value = float(fractional_cap or 0.0)
        except (TypeError, ValueError):
            fractional_value = 0.0
        if fractional_value <= 0:
            return 0.0
        equity = self._current_equity
        if equity is None:
            try:
                balance_cents = int(self.kalshi_client.get_balance().get("balance", 0))
                equity = balance_cents / 100.0
                self._current_equity = equity
            except Exception:
                return 0.0
        return max(float(equity) * fractional_value, 0.0)

    def open_position_views(self) -> list[dict[str, Any]]:
        exchange_state = self.fetch_exchange_state(order_limit=200, fill_limit=200)
        self.reconcile_exchange_state()
        entries = self._read_ledger()
        return self._build_open_position_views(exchange_state=exchange_state, ledger_entries=entries)

    def _build_open_position_views(
        self,
        *,
        exchange_state: dict[str, Any],
        ledger_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        latest_buy_by_market: dict[str, dict[str, Any]] = {}
        for entry in ledger_entries:
            request = entry.get("request", {}) if isinstance(entry.get("request"), dict) else {}
            market_ticker = str(request.get("ticker", "") or "")
            action = str(request.get("action", "") or "").lower()
            if not market_ticker or action != "buy":
                continue
            filled_count = int(entry.get("filled_count") or entry.get("fill_count") or 0)
            lifecycle_state = str(entry.get("lifecycle_state") or "")
            if filled_count <= 0 and lifecycle_state not in {"filled", "partially_filled"}:
                continue
            recorded_at = self._parse_iso_timestamp(entry.get("recorded_at"))
            previous = latest_buy_by_market.get(market_ticker)
            previous_ts = self._parse_iso_timestamp(previous.get("recorded_at")) if previous else None
            if previous is None or (recorded_at is not None and (previous_ts is None or recorded_at >= previous_ts)):
                latest_buy_by_market[market_ticker] = entry

        views: list[dict[str, Any]] = []
        for position in exchange_state["open_positions"]:
            market_ticker = str(position.get("market_ticker", position.get("ticker", "")) or "")
            if not market_ticker:
                continue
            entry = latest_buy_by_market.get(market_ticker)
            request = entry.get("request", {}) if isinstance(entry, dict) and isinstance(entry.get("request"), dict) else {}
            side = str(request.get("side") or "").lower()
            count = int(round(abs(float(self._extract_position_count(position) or 0.0))))
            if not market_ticker or not side or count <= 0:
                continue
            avg_fill_price_cents = entry.get("avg_fill_price_cents") if entry else None
            if avg_fill_price_cents in (None, ""):
                avg_fill_price_cents = self._position_average_price_cents(position)
            if avg_fill_price_cents in (None, ""):
                avg_fill_price_cents = request.get("yes_price") if side == "yes" else request.get("no_price")
            try:
                entry_price_cents = int(round(float(avg_fill_price_cents)))
            except (TypeError, ValueError):
                continue
            views.append(
                {
                    "market_ticker": market_ticker,
                    "side": side,
                    "contracts": count,
                    "entry_price_cents": entry_price_cents,
                    "entry_edge": float(entry.get("entry_edge") or 0.0) if entry else 0.0,
                    "entry_recorded_at": entry.get("recorded_at") if entry else None,
                    "source_order_id": entry.get("order_id") if entry else None,
                    "exchange_position": position,
                }
            )
        return views

    @staticmethod
    def _market_event_key(market_ticker: str) -> str:
        ticker = str(market_ticker or "")
        if "-" not in ticker:
            return ticker
        return ticker.rsplit("-", 1)[0]

    @staticmethod
    def _position_view_notional(view: dict[str, Any]) -> float:
        try:
            contracts = float(view.get("contracts") or 0.0)
            entry_price_cents = float(view.get("entry_price_cents") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return round((contracts * entry_price_cents) / 100.0, 4)

    def _read_ledger(self) -> list[dict[str, Any]]:
        return list(
            read_json_strict(
                self.ledger_path,
                missing_default=[],
                label="Real trading ledger",
            )
        )

    @staticmethod
    def _parse_iso_timestamp(raw: Any) -> datetime | None:
        if not raw:
            return None
        try:
            text = str(raw)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _extract_order_id(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        order = payload.get("order")
        if isinstance(order, dict):
            return str(order.get("order_id") or order.get("id") or "") or None
        return str(payload.get("order_id") or payload.get("id") or "") or None

    @staticmethod
    def _extract_orders(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("orders", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_fills(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("fills", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_positions(payload: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("market_positions", "positions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _extract_identifier(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
        return None

    @staticmethod
    def _extract_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_boolean(payload: dict[str, Any], keys: tuple[str, ...], *, fallback: bool = False) -> bool:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
        return fallback

    @staticmethod
    def _extract_exchange_status(order: dict[str, Any]) -> str:
        for key in ("status", "state", "order_status"):
            value = order.get(key)
            if value not in (None, ""):
                return str(value)
        if RealOrderExecutor._extract_boolean(order, ("is_resting", "resting")):
            return "resting"
        return "unknown"

    @classmethod
    def _average_fill_price(cls, fills: list[dict[str, Any]], *, preferred_side: str | None = None) -> int | None:
        weighted_total = 0.0
        total_count = 0
        for fill in fills:
            price = cls._extract_fill_price_cents(fill, preferred_side=preferred_side)
            count = cls._extract_numeric(fill, ("count", "quantity", "filled_count", "count_fp"))
            if price is None or count is None or count <= 0:
                continue
            weighted_total += float(price) * float(count)
            total_count += count
        if total_count <= 0:
            return None
        return int(round(weighted_total / total_count))

    @classmethod
    def _extract_fill_price_cents(cls, fill: dict[str, Any], *, preferred_side: str | None = None) -> int | None:
        side = str(preferred_side or "").strip().lower()
        if side == "yes":
            cents_keys = ("yes_price", "price", "price_cents", "no_price")
            dollar_keys = ("yes_price_dollars", "price_dollars", "no_price_dollars")
        elif side == "no":
            cents_keys = ("no_price", "price", "price_cents", "yes_price")
            dollar_keys = ("no_price_dollars", "price_dollars", "yes_price_dollars")
        else:
            cents_keys = ("price", "price_cents", "yes_price", "no_price")
            dollar_keys = ("price_dollars", "yes_price_dollars", "no_price_dollars")

        price = cls._extract_numeric(fill, cents_keys)
        if price is not None:
            return price
        for key in dollar_keys:
            raw = fill.get(key)
            if raw in (None, ""):
                continue
            try:
                return int(round(float(raw) * 100.0))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_position_count(position: dict[str, Any]) -> float:
        raw_position = position.get("position")
        if raw_position in (None, ""):
            raw_position = position.get("position_fp")
        if raw_position in (None, ""):
            raw_position = position.get("count")
        if raw_position in (None, ""):
            raw_position = position.get("count_fp")
        try:
            return float(raw_position or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _position_average_price_cents(position: dict[str, Any]) -> int | None:
        total_traded = position.get("total_traded_dollars")
        if total_traded in (None, ""):
            total_traded = position.get("market_exposure_dollars")
        try:
            total_traded_value = float(total_traded)
        except (TypeError, ValueError):
            return None
        count = abs(RealOrderExecutor._extract_position_count(position))
        if count <= 0:
            return None
        return int(round((total_traded_value / count) * 100.0))

    @staticmethod
    def _resolve_lifecycle_state(entry: dict[str, Any]) -> str:
        requested_count = int(entry.get("requested_count") or entry.get("request", {}).get("count") or 0)
        filled_count = int(entry.get("filled_count") or entry.get("fill_count") or 0)
        remaining_count_raw = entry.get("remaining_count")
        try:
            remaining_count = int(remaining_count_raw) if remaining_count_raw is not None else None
        except (TypeError, ValueError):
            remaining_count = None
        exchange_status = str(entry.get("exchange_status") or "").lower()
        cancel_status = str(entry.get("cancel_status") or "").lower()
        has_open_position = bool(entry.get("has_open_position", False))
        is_resting = bool(entry.get("is_resting", False))

        base_status = str(entry.get("status") or "").lower()
        if base_status == "blocked":
            return "blocked"
        if base_status == "dry_run":
            return "dry_run"
        if base_status == "replaced":
            return "replaced"

        if cancel_status == "cancel_dry_run":
            if entry.get("replacement_request") is not None:
                return "replace_dry_run"
            return "cancel_dry_run"
        if cancel_status == "cancel_requested":
            if entry.get("replacement_request") is not None:
                if filled_count > 0:
                    return "partially_filled_replace_requested"
                return "replace_requested"
            if filled_count > 0:
                return "partially_filled_cancel_requested"
            return "cancel_requested"

        if exchange_status in {"cancelled", "canceled"}:
            if filled_count > 0:
                return "partially_filled_cancelled"
            return "cancelled"

        if requested_count > 0 and filled_count >= requested_count and requested_count > 0:
            return "filled"
        if exchange_status in {"filled", "executed", "complete", "completed"} and (filled_count > 0 or remaining_count == 0):
            return "filled"
        if filled_count > 0 and (remaining_count is None or remaining_count > 0 or is_resting or has_open_position):
            return "partially_filled"
        if is_resting or exchange_status == "resting":
            return "resting"
        if base_status == "submitted":
            return "submitted"
        return base_status or "unknown"

    def _payload_matches_series(self, payload: dict[str, Any], keys: tuple[str, ...]) -> bool:
        if not self.config.series_ticker:
            return True
        saw_key = False
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            saw_key = True
            return str(value).startswith(f"{self.config.series_ticker}-")
        return not saw_key

    @staticmethod
    def _request_price_cents(request: dict[str, Any]) -> int | None:
        for key in ("yes_price", "no_price"):
            value = request.get(key)
            if value in (None, ""):
                continue
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    def _build_replacement_request(self, entry: dict[str, Any]) -> tuple[RealOrderRequest | None, str]:
        request = entry.get("request", {}) if isinstance(entry.get("request"), dict) else {}
        market_ticker = str(request.get("ticker") or "")
        side = str(request.get("side") or "").lower()
        action = str(request.get("action") or "").lower()
        current_price = self._request_price_cents(request)
        if not market_ticker or side not in {"yes", "no"} or action not in {"buy", "sell"} or current_price is None:
            return None, "missing_request_context"

        replacement_count = int(entry.get("replacement_count") or 0)
        if replacement_count >= int(self.config.max_replace_attempts):
            return None, "max_replacements_reached"

        market_payload = self.kalshi_client.get_market(market_ticker)
        market = market_payload.get("market") if isinstance(market_payload.get("market"), dict) else market_payload
        best_bid, best_ask = self._same_side_market_prices(market, side=side)
        current_mid = self._same_side_mid_price(best_bid=best_bid, best_ask=best_ask)
        if action == "buy":
            if best_bid is None:
                return None, "missing_best_bid"
            if self._order_is_still_competitive(current_price=current_price, best_bid=best_bid, best_ask=best_ask, action=action):
                if self._fails_mid_guard(current_price=current_price, mid_price=current_mid, best_bid=best_bid, best_ask=best_ask, action=action):
                    return None, "overpaying_vs_mid"
                return None, "still_competitive"
            if self._fails_mid_guard(current_price=current_price, mid_price=current_mid, best_bid=best_bid, best_ask=best_ask, action=action):
                return None, "overpaying_vs_mid"
        else:
            if best_ask is None:
                return None, "missing_best_ask"
            if self._order_is_still_competitive(current_price=current_price, best_bid=best_bid, best_ask=best_ask, action=action):
                if self._fails_mid_guard(current_price=current_price, mid_price=current_mid, best_bid=best_bid, best_ask=best_ask, action=action):
                    return None, "underpricing_vs_mid"
                return None, "still_competitive"
            if self._fails_mid_guard(current_price=current_price, mid_price=current_mid, best_bid=best_bid, best_ask=best_ask, action=action):
                return None, "underpricing_vs_mid"

        minutes_to_expiry = self._market_minutes_to_expiry(market)
        if minutes_to_expiry is not None and minutes_to_expiry < 15.0:
            return None, "near_expiry"

        initial_price = int(entry.get("initial_price_cents") or current_price)
        replacement_price = self._replacement_price_cents(best_bid=best_bid, best_ask=best_ask, action=action)
        if replacement_price == current_price:
            return None, "no_price_change"

        if action == "buy":
            if replacement_price > initial_price + int(self.config.max_chase_cents):
                return None, "chase_limit_exceeded"
        else:
            if replacement_price < initial_price - int(self.config.max_chase_cents):
                return None, "chase_limit_exceeded"

        return (
            RealOrderRequest(
                market_ticker=market_ticker,
                side=side,
                action=action,
                count=int(request.get("count") or entry.get("requested_count") or 0),
                yes_price_cents=replacement_price if side == "yes" else None,
                no_price_cents=replacement_price if side == "no" else None,
                order_type=str(request.get("type") or "limit"),
                client_order_id=None,
                expiration_ts=request.get("expiration_ts"),
            ),
            "market_moved_away",
        )

    @staticmethod
    def _order_is_still_competitive(*, current_price: int, best_bid: int | None, best_ask: int | None, action: str) -> bool:
        if action == "buy":
            return best_bid is not None and current_price >= best_bid
        return best_ask is not None and current_price <= best_ask

    @staticmethod
    def _request_is_at_least_as_aggressive(*, action: str, existing_price: int, desired_price: int) -> bool:
        if action == "buy":
            return existing_price >= desired_price
        return existing_price <= desired_price

    @staticmethod
    def _replacement_price_cents(*, best_bid: int | None, best_ask: int | None, action: str) -> int:
        if action == "buy":
            if best_bid is None:
                raise ValueError("best_bid is required for buy replacements")
            target = min(best_bid + 1, 99)
            if best_ask is not None:
                target = min(target, best_ask)
            return target
        if best_ask is None:
            raise ValueError("best_ask is required for sell replacements")
        target = max(best_ask - 1, 1)
        if best_bid is not None:
            target = max(target, best_bid)
        return target

    @staticmethod
    def _same_side_market_prices(market: dict[str, Any], *, side: str) -> tuple[int | None, int | None]:
        if side == "yes":
            return (
                RealOrderExecutor._extract_market_field_cents(market, ("yes_bid", "yes_bid_price", "yes_bid_dollars")),
                RealOrderExecutor._extract_market_field_cents(market, ("yes_ask", "yes_ask_price", "yes_ask_dollars")),
            )
        return (
            RealOrderExecutor._extract_market_field_cents(market, ("no_bid", "no_bid_price", "no_bid_dollars")),
            RealOrderExecutor._extract_market_field_cents(market, ("no_ask", "no_ask_price", "no_ask_dollars")),
        )

    @staticmethod
    def _same_side_mid_price(*, best_bid: int | None, best_ask: int | None) -> float | None:
        if best_bid is None or best_ask is None:
            return None
        return (float(best_bid) + float(best_ask)) / 2.0

    @staticmethod
    def _mid_guard_tolerance_cents(*, best_bid: int | None, best_ask: int | None) -> float:
        if best_bid is None or best_ask is None:
            return 2.0
        half_spread = max(float(best_ask - best_bid), 0.0) / 2.0
        return min(2.0, half_spread + 1.0)

    @classmethod
    def _fails_mid_guard(
        cls,
        *,
        current_price: int,
        mid_price: float | None,
        best_bid: int | None,
        best_ask: int | None,
        action: str,
    ) -> bool:
        if mid_price is None:
            return False
        tolerance = cls._mid_guard_tolerance_cents(best_bid=best_bid, best_ask=best_ask)
        if action == "buy":
            return float(current_price) > mid_price + tolerance
        return float(current_price) < mid_price - tolerance

    @classmethod
    def _market_minutes_to_expiry(cls, market: dict[str, Any]) -> float | None:
        if not isinstance(market, dict):
            return None
        expiry = cls._parse_market_datetime(
            market.get("close_time")
            or market.get("expected_expiration_time")
            or market.get("expiry")
            or market.get("settlement_time")
            or market.get("expiration_time")
        )
        if expiry is None:
            return None
        return max((expiry - datetime.now(timezone.utc)).total_seconds() / 60.0, 0.0)

    @staticmethod
    def _parse_market_datetime(raw: Any) -> datetime | None:
        if raw in (None, ""):
            return None
        text = str(raw)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _extract_market_field_cents(market: dict[str, Any], keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = market.get(key)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if "dollars" in key and number <= 1.0:
                return int(round(number * 100.0))
            if number <= 1.0:
                return int(round(number * 100.0))
            return int(round(number))
        return None

    @staticmethod
    def _extract_market_price_cents(market: dict[str, Any], *, side: str, action: str) -> int | None:
        if not isinstance(market, dict):
            return None
        key_sets: tuple[str, ...]
        if side == "yes" and action == "buy":
            key_sets = ("yes_ask", "yes_ask_price", "yes_ask_dollars")
        elif side == "yes" and action == "sell":
            key_sets = ("yes_bid", "yes_bid_price", "yes_bid_dollars")
        elif side == "no" and action == "buy":
            key_sets = ("no_ask", "no_ask_price", "no_ask_dollars")
        else:
            key_sets = ("no_bid", "no_bid_price", "no_bid_dollars")

        for key in key_sets:
            value = market.get(key)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if "dollars" in key and number <= 1.0:
                return int(round(number * 100.0))
            if number <= 1.0:
                return int(round(number * 100.0))
            return int(round(number))
        return None
