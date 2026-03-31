from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def ensure_utc(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def to_iso_utc(value: datetime | str | pd.Timestamp) -> str:
    return ensure_utc(value).isoformat()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
