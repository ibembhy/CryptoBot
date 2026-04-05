from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from kalshi_btc_bot.settings import Settings, load_settings


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "kalshi_btc_snapshots.sqlite3"
DEFAULT_OUTPUT_DIR = DATA_DIR / "analysis_results"

SETTLED_CSV_BY_SERIES = {
    "KXBTCD": DATA_DIR / "settled_markets_kxbtcd.csv",
    "KXBTC15M": DATA_DIR / "settled_markets_kxbtc15m.csv",
}

LIVE_LEDGER_BY_SERIES = {
    "KXBTCD": DATA_DIR / "real_trading_ledger_kxbtcd.json",
    "KXBTC15M": DATA_DIR / "real_trading_ledger_kxbtc15m.json",
}


@dataclass(frozen=True)
class AnalysisContext:
    root: Path
    data_dir: Path
    output_dir: Path
    db_path: Path
    settings: Settings


def build_context(
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> AnalysisContext:
    resolved_output = Path(output_dir or DEFAULT_OUTPUT_DIR)
    resolved_output.mkdir(parents=True, exist_ok=True)
    return AnalysisContext(
        root=ROOT,
        data_dir=DATA_DIR,
        output_dir=resolved_output,
        db_path=Path(db_path or DEFAULT_DB_PATH),
        settings=load_settings(config_path),
    )


def snapshot_row_count(db_path: str | Path) -> int:
    with sqlite3.connect(Path(db_path)) as conn:
        return int(pd.read_sql_query("SELECT COUNT(*) AS count FROM market_snapshots", conn).iloc[0]["count"])


def load_snapshots(
    db_path: str | Path,
    series: str | None = None,
    columns: Iterable[str] | None = None,
    where_sql: str | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    selected_columns = ", ".join(columns) if columns else "*"
    query = f"SELECT {selected_columns} FROM market_snapshots"
    conditions: list[str] = []
    params: list[Any] = []

    if series:
        conditions.append("series_ticker = ?")
        params.append(series)
    if where_sql:
        conditions.append(where_sql)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    conn = sqlite3.connect(Path(db_path))
    if chunksize:
        def _iter_frames() -> Iterable[pd.DataFrame]:
            try:
                for chunk in pd.read_sql_query(query, conn, params=params, chunksize=chunksize):
                    yield _parse_snapshot_dates(chunk)
            finally:
                conn.close()

        return _iter_frames()

    try:
        frame = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return _parse_snapshot_dates(frame)


def _parse_snapshot_dates(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("observed_at", "expiry"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce", format="ISO8601")
    return frame


def load_settlements(series: str) -> pd.DataFrame:
    path = SETTLED_CSV_BY_SERIES[series]
    frame = pd.read_csv(path)
    for column in ("open_time", "close_time", "settlement_ts"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def load_all_settlements() -> pd.DataFrame:
    frames = [load_settlements(series) for series in SETTLED_CSV_BY_SERIES]
    return pd.concat(frames, ignore_index=True)


def load_live_ledger(series: str) -> pd.DataFrame:
    path = LIVE_LEDGER_BY_SERIES[series]
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("orders") or []
    else:
        entries = payload
    frame = pd.json_normalize(entries)
    for column in ("recorded_at", "created_at", "updated_at"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return load_settings(config_path).raw
