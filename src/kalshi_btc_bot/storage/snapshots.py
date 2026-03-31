from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from kalshi_btc_bot.types import MarketSnapshot


class SnapshotStore:
    SQLITE_TIMEOUT_SECONDS = 30.0
    EFFECTIVE_EXPIRY_SQL = """
        COALESCE(
            json_extract(metadata_json, '$.close_time'),
            json_extract(metadata_json, '$.expected_expiration_time'),
            json_extract(metadata_json, '$.expiry'),
            json_extract(metadata_json, '$.settlement_time'),
            json_extract(metadata_json, '$.expiration_time'),
            expiry
        )
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=self.SQLITE_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_ticker TEXT NOT NULL,
                    series_ticker TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    expiry TEXT NOT NULL,
                    contract_type TEXT NOT NULL,
                    underlying_symbol TEXT NOT NULL,
                    spot_price REAL NOT NULL,
                    threshold REAL,
                    range_low REAL,
                    range_high REAL,
                    direction TEXT,
                    yes_bid REAL,
                    yes_ask REAL,
                    no_bid REAL,
                    no_ask REAL,
                    mid_price REAL,
                    implied_probability REAL,
                    volume REAL,
                    open_interest REAL,
                    settlement_price REAL,
                    source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker_observed
                ON market_snapshots(market_ticker, observed_at)
                """
            )

    def insert_snapshot(self, snapshot: MarketSnapshot) -> None:
        payload = asdict(snapshot)
        metadata_json = json.dumps(payload.pop("metadata", {}), sort_keys=True, default=str)
        payload["observed_at"] = snapshot.observed_at.isoformat()
        payload["expiry"] = snapshot.expiry.isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO market_snapshots (
                    market_ticker, series_ticker, observed_at, expiry, contract_type,
                    underlying_symbol, spot_price, threshold, range_low, range_high,
                    direction, yes_bid, yes_ask, no_bid, no_ask, mid_price,
                    implied_probability, volume, open_interest, settlement_price,
                    source, metadata_json
                ) VALUES (
                    :market_ticker, :series_ticker, :observed_at, :expiry, :contract_type,
                    :underlying_symbol, :spot_price, :threshold, :range_low, :range_high,
                    :direction, :yes_bid, :yes_ask, :no_bid, :no_ask, :mid_price,
                    :implied_probability, :volume, :open_interest, :settlement_price,
                    :source, :metadata_json
                )
                """,
                {**payload, "metadata_json": metadata_json},
            )

    def snapshot_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM market_snapshots").fetchone()
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
    ) -> list[MarketSnapshot]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = :series_ticker")
            params["series_ticker"] = series_ticker
        if market_ticker is not None:
            conditions.append("market_ticker = :market_ticker")
            params["market_ticker"] = market_ticker
        if observed_from is not None:
            conditions.append("observed_at >= :observed_from")
            params["observed_from"] = observed_from.isoformat()
        if observed_to is not None:
            conditions.append("observed_at <= :observed_to")
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
                       AND MIN({self.EFFECTIVE_EXPIRY_SQL}) <= :reference_time
                )
                """
            )
            params["reference_time"] = reference

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = "LIMIT :limit" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                WITH latest AS (
                    SELECT MAX(id) AS id
                    FROM market_snapshots
                    GROUP BY market_ticker, observed_at
                )
                SELECT s.*, {self.EFFECTIVE_EXPIRY_SQL} AS effective_expiry
                FROM market_snapshots s
                JOIN latest l ON s.id = l.id
                {where_clause.replace("WHERE", "WHERE", 1)}
                ORDER BY s.observed_at, s.market_ticker
                {limit_clause}
                """,
                params,
            ).fetchall()
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
            conditions.append("series_ticker = :series_ticker")
            params["series_ticker"] = series_ticker
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        reference_iso = (reference_time or datetime.now().astimezone()).isoformat()

        with self._connect() as conn:
            overall = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS snapshot_count,
                    COUNT(DISTINCT market_ticker) AS unique_markets,
                    MIN(observed_at) AS first_observed_at,
                    MAX(observed_at) AS last_observed_at,
                    COUNT(DISTINCT CASE WHEN settlement_price IS NOT NULL THEN market_ticker END) AS settled_markets,
                    COUNT(DISTINCT CASE WHEN {self.EFFECTIVE_EXPIRY_SQL} <= :reference_time THEN market_ticker END) AS expired_markets,
                    COUNT(DISTINCT CASE WHEN {self.EFFECTIVE_EXPIRY_SQL} <= :reference_time AND settlement_price IS NOT NULL THEN market_ticker END) AS replay_ready_markets,
                    COUNT(DISTINCT CASE WHEN {self.EFFECTIVE_EXPIRY_SQL} <= :reference_time AND settlement_price IS NULL THEN market_ticker END) AS unresolved_expired_markets
                FROM market_snapshots
                {where_clause}
                """,
                {**params, "reference_time": reference_iso},
            ).fetchone()
            per_contract_type = conn.execute(
                f"""
                SELECT contract_type, COUNT(*) AS snapshot_count, COUNT(DISTINCT market_ticker) AS market_count
                FROM market_snapshots
                {where_clause}
                GROUP BY contract_type
                ORDER BY contract_type
                """,
                params,
            ).fetchall()

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
            cursor = conn.execute(
                """
                UPDATE market_snapshots
                SET series_ticker = :new_value
                WHERE series_ticker = :old_value
                """,
                {"old_value": old_value, "new_value": new_value},
            )
            return int(cursor.rowcount)

    def list_market_tickers(self, *, series_ticker: str | None = None) -> list[str]:
        conditions: list[str] = []
        params: dict[str, object] = {}
        if series_ticker is not None:
            conditions.append("series_ticker = :series_ticker")
            params["series_ticker"] = series_ticker
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT market_ticker
                FROM market_snapshots
                {where_clause}
                ORDER BY market_ticker
                """,
                params,
            ).fetchall()
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
            conditions.append("series_ticker = :series_ticker")
            params["series_ticker"] = series_ticker
        if expired_before is not None:
            conditions.append(f"{self.EFFECTIVE_EXPIRY_SQL} <= :expired_before")
            params["expired_before"] = expired_before.isoformat()
        if expired_after is not None:
            conditions.append(f"{self.EFFECTIVE_EXPIRY_SQL} >= :expired_after")
            params["expired_after"] = expired_after.isoformat()

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        having_clause = "HAVING MAX(settlement_price) IS NULL" if unsettled_only else ""
        limit_clause = "LIMIT :limit" if limit is not None else ""
        if limit is not None:
            params["limit"] = limit

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    market_ticker,
                    series_ticker,
                    MIN({self.EFFECTIVE_EXPIRY_SQL}) AS expiry,
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
            ).fetchall()
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

    def update_market_settlement(self, market_ticker: str, settlement_price: float) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE market_snapshots
                SET settlement_price = :settlement_price
                WHERE market_ticker = :market_ticker
                """,
                {"market_ticker": market_ticker, "settlement_price": settlement_price},
            )
            return int(cursor.rowcount)

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> MarketSnapshot:
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return MarketSnapshot(
            source=row["source"],
            series_ticker=row["series_ticker"],
            market_ticker=row["market_ticker"],
            contract_type=row["contract_type"],
            underlying_symbol=row["underlying_symbol"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
            expiry=datetime.fromisoformat(row["effective_expiry"] if "effective_expiry" in row.keys() and row["effective_expiry"] else row["expiry"]),
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
