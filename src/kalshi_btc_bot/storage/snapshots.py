from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import psycopg2
import psycopg2.extras

from kalshi_btc_bot.types import MarketSnapshot
from kalshi_btc_bot.utils.time import ensure_utc

# ---------------------------------------------------------------------------
# Backward-compat shim: if a caller still passes a file path (old SQLite API)
# we derive a DSN from a well-known env var or raise a clear error.
# ---------------------------------------------------------------------------
import os as _os
_DEFAULT_DSN = _os.getenv(
    "KALSHI_BTC_COLLECTOR_POSTGRES_DSN",
    "postgresql://cryptobot:cryptobot_secure_2024@localhost/cryptobot_snapshots",
)

EFFECTIVE_EXPIRY_SQL = """
    COALESCE(
        metadata_json->>'close_time',
        metadata_json->>'expected_expiration_time',
        metadata_json->>'expiry',
        metadata_json->>'settlement_time',
        metadata_json->>'expiration_time',
        expiry::text
    )
"""


class SnapshotStore:
    _INIT_LOCK = threading.Lock()
    _INITIALIZED_DSNS: set[str] = set()

    # Accept either a DSN string ("postgresql://...") or a legacy Path/str for
    # the SQLite file.  When a Path or non-DSN string is passed we log a
    # deprecation note and fall back to the default DSN.
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        path_str = str(path)
        if path_str.startswith("postgresql://") or path_str.startswith("postgres://"):
            self.dsn = path_str
        else:
            # Legacy SQLite path — ignore and use default DSN
            import logging
            logging.getLogger(__name__).warning(
                "SnapshotStore received SQLite path %r; using default PostgreSQL DSN instead.",
                path_str,
            )
            self.dsn = _DEFAULT_DSN

        self.read_only = read_only

        normalized = self.dsn
        if normalized not in self._INITIALIZED_DSNS:
            with self._INIT_LOCK:
                if normalized not in self._INITIALIZED_DSNS:
                    self._init_db()
                    self._INITIALIZED_DSNS.add(normalized)

    @contextmanager
    def _connect(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(self.dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_snapshots (
                        id BIGSERIAL PRIMARY KEY,
                        market_ticker TEXT NOT NULL,
                        series_ticker TEXT NOT NULL,
                        observed_at TIMESTAMPTZ NOT NULL,
                        expiry TIMESTAMPTZ NOT NULL,
                        contract_type TEXT NOT NULL,
                        underlying_symbol TEXT NOT NULL,
                        spot_price DOUBLE PRECISION NOT NULL,
                        threshold DOUBLE PRECISION,
                        range_low DOUBLE PRECISION,
                        range_high DOUBLE PRECISION,
                        direction TEXT,
                        yes_bid DOUBLE PRECISION,
                        yes_ask DOUBLE PRECISION,
                        no_bid DOUBLE PRECISION,
                        no_ask DOUBLE PRECISION,
                        mid_price DOUBLE PRECISION,
                        implied_probability DOUBLE PRECISION,
                        volume DOUBLE PRECISION,
                        open_interest DOUBLE PRECISION,
                        settlement_price DOUBLE PRECISION,
                        source TEXT NOT NULL,
                        metadata_json JSONB NOT NULL DEFAULT '{}'
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ms_ticker_observed
                    ON market_snapshots(market_ticker, observed_at)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ms_series_observed
                    ON market_snapshots(series_ticker, observed_at)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ms_series_market_observed
                    ON market_snapshots(series_ticker, market_ticker, observed_at)
                    """
                )

    def insert_snapshot(self, snapshot: MarketSnapshot) -> None:
        payload = asdict(snapshot)
        metadata_json = json.dumps(payload.pop("metadata", {}), sort_keys=True, default=str)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO market_snapshots (
                        market_ticker, series_ticker, observed_at, expiry, contract_type,
                        underlying_symbol, spot_price, threshold, range_low, range_high,
                        direction, yes_bid, yes_ask, no_bid, no_ask, mid_price,
                        implied_probability, volume, open_interest, settlement_price,
                        source, metadata_json
                    ) VALUES (
                        %(market_ticker)s, %(series_ticker)s, %(observed_at)s, %(expiry)s,
                        %(contract_type)s, %(underlying_symbol)s, %(spot_price)s,
                        %(threshold)s, %(range_low)s, %(range_high)s, %(direction)s,
                        %(yes_bid)s, %(yes_ask)s, %(no_bid)s, %(no_ask)s, %(mid_price)s,
                        %(implied_probability)s, %(volume)s, %(open_interest)s,
                        %(settlement_price)s, %(source)s, %(metadata_json)s
                    )
                    """,
                    {**payload, "metadata_json": metadata_json},
                )

    def snapshot_count(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM market_snapshots")
                row = cur.fetchone()
        return int(row["count"])

    def load_snapshots(
        self,
        *,
        series_ticker: str | None = None,
        market_ticker: str | None = None,
        observed_from: datetime | None = None,
        observed_to: datetime | None = None,
        limit: int | None = None,
        replay_ready_only: bool = False,
        reference_time: datetime | None = None,
        descending: bool = False,
    ) -> list[MarketSnapshot]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = %(series_ticker)s")
            params["series_ticker"] = series_ticker
        if market_ticker is not None:
            conditions.append("market_ticker = %(market_ticker)s")
            params["market_ticker"] = market_ticker
        if observed_from is not None:
            conditions.append("observed_at >= %(observed_from)s")
            params["observed_from"] = observed_from.isoformat()
        if observed_to is not None:
            conditions.append("observed_at <= %(observed_to)s")
            params["observed_to"] = observed_to.isoformat()
        if replay_ready_only:
            reference = (reference_time or datetime.now().astimezone()).isoformat()
            conditions.append(
                f"""
                market_ticker IN (
                    SELECT market_ticker
                    FROM market_snapshots
                    GROUP BY market_ticker
                    HAVING MAX(settlement_price) IS NOT NULL
                       AND MIN({EFFECTIVE_EXPIRY_SQL}) <= %(reference_time)s
                )
                """
            )
            params["reference_time"] = reference

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = "LIMIT %(limit)s" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit
        order_direction = "DESC" if descending else "ASC"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH latest AS (
                        SELECT MAX(id) AS id
                        FROM market_snapshots
                        {where_clause}
                        GROUP BY market_ticker, observed_at
                    )
                    SELECT s.*, {EFFECTIVE_EXPIRY_SQL} AS effective_expiry
                    FROM market_snapshots s
                    JOIN latest l ON s.id = l.id
                    ORDER BY s.observed_at {order_direction}, s.market_ticker {order_direction}
                    {limit_clause}
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def latest_snapshot_for_market(self, market_ticker: str) -> MarketSnapshot | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT s.*, {EFFECTIVE_EXPIRY_SQL} AS effective_expiry
                    FROM market_snapshots s
                    WHERE market_ticker = %(market_ticker)s
                    ORDER BY observed_at DESC, id DESC
                    LIMIT 1
                    """,
                    {"market_ticker": market_ticker},
                )
                row = cur.fetchone()
        return self._row_to_snapshot(row) if row is not None else None

    def load_recent_rows(
        self,
        *,
        series_ticker: str | None = None,
        market_ticker: str | None = None,
        observed_from: datetime | None = None,
        observed_to: datetime | None = None,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[MarketSnapshot]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = %(series_ticker)s")
            params["series_ticker"] = series_ticker
        if market_ticker is not None:
            conditions.append("market_ticker = %(market_ticker)s")
            params["market_ticker"] = market_ticker
        if observed_from is not None:
            conditions.append("observed_at >= %(observed_from)s")
            params["observed_from"] = observed_from.isoformat()
        if observed_to is not None:
            conditions.append("observed_at <= %(observed_to)s")
            params["observed_to"] = observed_to.isoformat()

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = "LIMIT %(limit)s" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit
        order_direction = "DESC" if descending else "ASC"

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT s.*, {EFFECTIVE_EXPIRY_SQL} AS effective_expiry
                    FROM market_snapshots s
                    {where_clause}
                    ORDER BY s.observed_at {order_direction}, s.id {order_direction}
                    {limit_clause}
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def dataset_summary(
        self,
        *,
        series_ticker: str | None = None,
        reference_time: datetime | None = None,
    ) -> dict[str, object]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = %(series_ticker)s")
            params["series_ticker"] = series_ticker
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        reference_iso = (reference_time or datetime.now().astimezone()).isoformat()
        params["reference_time"] = reference_iso

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        COUNT(*) AS snapshot_count,
                        COUNT(DISTINCT market_ticker) AS unique_markets,
                        MIN(observed_at) AS first_observed_at,
                        MAX(observed_at) AS last_observed_at,
                        COUNT(DISTINCT CASE WHEN settlement_price IS NOT NULL THEN market_ticker END) AS settled_markets,
                        COUNT(DISTINCT CASE WHEN {EFFECTIVE_EXPIRY_SQL} <= %(reference_time)s THEN market_ticker END) AS expired_markets,
                        COUNT(DISTINCT CASE WHEN {EFFECTIVE_EXPIRY_SQL} <= %(reference_time)s AND settlement_price IS NOT NULL THEN market_ticker END) AS replay_ready_markets,
                        COUNT(DISTINCT CASE WHEN {EFFECTIVE_EXPIRY_SQL} <= %(reference_time)s AND settlement_price IS NULL THEN market_ticker END) AS unresolved_expired_markets
                    FROM market_snapshots
                    {where_clause}
                    """,
                    params,
                )
                overall = cur.fetchone()
                cur.execute(
                    f"""
                    SELECT contract_type, COUNT(*) AS snapshot_count, COUNT(DISTINCT market_ticker) AS market_count
                    FROM market_snapshots
                    {where_clause}
                    GROUP BY contract_type
                    ORDER BY contract_type
                    """,
                    {k: v for k, v in params.items() if k != "reference_time"},
                )
                per_contract_type = cur.fetchall()

        return {
            "snapshot_count": int(overall["snapshot_count"] or 0),
            "unique_markets": int(overall["unique_markets"] or 0),
            "first_observed_at": overall["first_observed_at"],
            "last_observed_at": overall["last_observed_at"],
            "settled_markets": int(overall["settled_markets"] or 0),
            "expired_markets": int(overall["expired_markets"] or 0),
            "replay_ready_markets": int(overall["replay_ready_markets"] or 0),
            "unresolved_expired_markets": int(overall["unresolved_expired_markets"] or 0),
            "contract_types": [
                {
                    "contract_type": row["contract_type"],
                    "snapshot_count": int(row["snapshot_count"]),
                    "market_count": int(row["market_count"]),
                }
                for row in per_contract_type
            ],
        }

    def remap_series_ticker(self, old_value: str, new_value: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE market_snapshots
                    SET series_ticker = %(new_value)s
                    WHERE series_ticker = %(old_value)s
                    """,
                    {"old_value": old_value, "new_value": new_value},
                )
                return int(cur.rowcount)

    def list_market_tickers(self, *, series_ticker: str | None = None) -> list[str]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = %(series_ticker)s")
            params["series_ticker"] = series_ticker
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT DISTINCT market_ticker
                    FROM market_snapshots
                    {where_clause}
                    ORDER BY market_ticker
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [str(row["market_ticker"]) for row in rows]

    def list_market_records(
        self,
        *,
        series_ticker: str | None = None,
        expired_before: datetime | None = None,
        expired_after: datetime | None = None,
        unsettled_only: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = %(series_ticker)s")
            params["series_ticker"] = series_ticker
        if expired_before is not None:
            conditions.append(f"{EFFECTIVE_EXPIRY_SQL} <= %(expired_before)s")
            params["expired_before"] = expired_before.isoformat()
        if expired_after is not None:
            conditions.append(f"{EFFECTIVE_EXPIRY_SQL} >= %(expired_after)s")
            params["expired_after"] = expired_after.isoformat()

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        having_clause = "HAVING MAX(settlement_price) IS NULL" if unsettled_only else ""
        limit_clause = "LIMIT %(limit)s" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        market_ticker,
                        series_ticker,
                        MIN({EFFECTIVE_EXPIRY_SQL}) AS expiry,
                        MAX(settlement_price) AS settlement_price,
                        COUNT(*) AS snapshot_count
                    FROM market_snapshots
                    {where_clause}
                    GROUP BY market_ticker, series_ticker
                    {having_clause}
                    ORDER BY expiry, market_ticker
                    {limit_clause}
                    """,
                    params,
                )
                rows = cur.fetchall()
        return [
            {
                "market_ticker": str(row["market_ticker"]),
                "series_ticker": str(row["series_ticker"]),
                "expiry": str(row["expiry"]),
                "settlement_price": row["settlement_price"],
                "snapshot_count": int(row["snapshot_count"]),
            }
            for row in rows
        ]

    def prune_old_snapshots(self, *, retain_days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM market_snapshots
                    WHERE observed_at < %(cutoff)s
                      AND settlement_price IS NOT NULL
                    """,
                    {"cutoff": cutoff.isoformat()},
                )
                return int(cur.rowcount)

    def update_market_settlement(self, market_ticker: str, settlement_price: float) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE market_snapshots
                    SET settlement_price = %(settlement_price)s
                    WHERE market_ticker = %(market_ticker)s
                    """,
                    {"market_ticker": market_ticker, "settlement_price": settlement_price},
                )
                return int(cur.rowcount)

    @staticmethod
    def _row_to_snapshot(row: dict) -> MarketSnapshot:
        metadata_raw = row["metadata_json"]
        if isinstance(metadata_raw, str):
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        elif isinstance(metadata_raw, dict):
            metadata = metadata_raw
        else:
            metadata = {}

        effective_expiry = row.get("effective_expiry") or row["expiry"]

        return MarketSnapshot(
            source=row["source"],
            series_ticker=row["series_ticker"],
            market_ticker=row["market_ticker"],
            contract_type=row["contract_type"],
            underlying_symbol=row["underlying_symbol"],
            observed_at=ensure_utc(row["observed_at"]).to_pydatetime(),
            expiry=ensure_utc(effective_expiry).to_pydatetime(),
            spot_price=float(row["spot_price"]),
            threshold=row["threshold"],
            range_low=row["range_low"],
            range_high=row["range_high"],
            direction=row["direction"],
            yes_bid=row["yes_bid"],
            yes_ask=row["yes_ask"],
            no_bid=row["no_bid"],
            no_ask=row["no_ask"],
            mid_price=row["mid_price"],
            implied_probability=row["implied_probability"],
            volume=row["volume"],
            open_interest=row["open_interest"],
            settlement_price=row["settlement_price"],
            metadata=metadata,
        )
