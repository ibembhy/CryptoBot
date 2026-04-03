from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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
            "context": preflight["context"],
        }
        self._append_ledger(result)
        return result

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
            raw_position = position.get("position", position.get("count", 0))
            try:
                position_count = float(raw_position or 0)
            except (TypeError, ValueError):
                position_count = 0.0
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
                fill_count = sum(int(self._extract_numeric(fill, ("count", "quantity", "filled_count")) or 0) for fill in order_fills)
                avg_fill_price = self._average_fill_price(order_fills)
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
        max_age = int(max_age_seconds if max_age_seconds is not None else self.config.stale_order_timeout_seconds)
        now = datetime.now(timezone.utc)
        ledger_entries = self._read_ledger()
        cancelled: list[dict[str, Any]] = []

        for entry in ledger_entries:
            order_id = str(entry.get("order_id") or "")
            if not order_id:
                continue
            if not bool(entry.get("is_resting", False)):
                continue
            recorded_at = self._parse_iso_timestamp(entry.get("recorded_at"))
            if recorded_at is None:
                continue
            age_seconds = (now - recorded_at).total_seconds()
            if age_seconds < max_age:
                continue

            if self.config.dry_run:
                cancel_response = {"status": "dry_run_cancel", "order_id": order_id}
                cancel_state = "cancel_dry_run"
            else:
                cancel_response = self.kalshi_client.cancel_order(order_id)
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
            "stale_order_timeout_seconds": max_age,
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
        same_market_resting = [order for order in resting_orders if str(order.get("ticker", order.get("market_ticker", "")) or "") == request.market_ticker]
        same_market_positions = [
            position
            for position in open_positions
            if str(position.get("market_ticker", position.get("ticker", "")) or "") == request.market_ticker
        ]

        context = {
            "series_ticker": self.config.series_ticker,
            "todays_recorded_orders": len(todays_entries),
            "resting_orders": len(resting_orders),
            "open_positions": len(open_positions),
            "same_market_resting_orders": len(same_market_resting),
            "same_market_open_positions": len(same_market_positions),
            "dry_run": self.config.dry_run,
        }
        if self.config.max_daily_orders > 0 and len(todays_entries) >= self.config.max_daily_orders:
            return RealPreflightResult(False, "Daily real-order cap reached.", context)
        if self.config.max_open_orders > 0 and len(resting_orders) >= self.config.max_open_orders:
            return RealPreflightResult(False, "Open resting-order cap reached.", context)
        if self.config.max_open_positions > 0 and len(open_positions) >= self.config.max_open_positions:
            return RealPreflightResult(False, "Open position cap reached on exchange.", context)
        if same_market_resting:
            return RealPreflightResult(False, "Existing resting order already on this market.", context)
        if same_market_positions:
            return RealPreflightResult(False, "Existing exchange position already open on this market.", context)
        return RealPreflightResult(True, "allowed", context)

    def fetch_exchange_state(self, *, order_limit: int = 200, fill_limit: int = 200) -> dict[str, Any]:
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
        return {
            "orders": orders,
            "fills": fills,
            "positions": positions,
            "resting_orders": resting_orders,
            "open_positions": open_positions,
        }

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

    @staticmethod
    def _average_fill_price(fills: list[dict[str, Any]]) -> int | None:
        weighted_total = 0.0
        total_count = 0
        for fill in fills:
            price = RealOrderExecutor._extract_numeric(fill, ("yes_price", "no_price", "price", "price_cents"))
            count = RealOrderExecutor._extract_numeric(fill, ("count", "quantity", "filled_count"))
            if price is None or count is None or count <= 0:
                continue
            weighted_total += float(price) * float(count)
            total_count += count
        if total_count <= 0:
            return None
        return int(round(weighted_total / total_count))

    @staticmethod
    def _extract_position_count(position: dict[str, Any]) -> float:
        raw_position = position.get("position", position.get("count", 0))
        try:
            return float(raw_position or 0)
        except (TypeError, ValueError):
            return 0.0

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

        if cancel_status == "cancel_dry_run":
            return "cancel_dry_run"
        if cancel_status == "cancel_requested":
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
