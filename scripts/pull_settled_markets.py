"""
Block 3 - Item 11: Pull Kalshi historical settled market metadata.

Uses direct series pagination (list_markets with status=settled) — fast.

Writes two CSV files:
  data/settled_markets_kxbtcd.csv
  data/settled_markets_kxbtc15m.csv

Columns:
  ticker, event_ticker, series, open_time, close_time, settlement_ts,
  floor_strike, cap_strike, expiration_value, result, volume, open_interest,
  last_price_dollars

Run from repo root:
  python scripts/pull_settled_markets.py [--max-pages N]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshi_btc_bot.markets.kalshi import KalshiClient

SERIES = ["KXBTCD", "KXBTC15M"]
OUT_DIR = Path(__file__).parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

FIELDS = [
    "ticker",
    "event_ticker",
    "series",
    "open_time",
    "close_time",
    "settlement_ts",
    "floor_strike",
    "cap_strike",
    "expiration_value",
    "result",
    "volume",
    "open_interest",
    "last_price_dollars",
]


def _dt(value) -> str:
    if not value:
        return ""
    return str(value).replace("Z", "+00:00")[:25]


def _flatten(market: dict, series: str) -> dict:
    return {
        "ticker": market.get("ticker", ""),
        "event_ticker": market.get("event_ticker", ""),
        "series": series,
        "open_time": _dt(market.get("open_time")),
        "close_time": _dt(market.get("close_time")),
        "settlement_ts": _dt(market.get("settlement_ts")),
        "floor_strike": market.get("floor_strike", ""),
        "cap_strike": market.get("cap_strike", ""),
        "expiration_value": market.get("expiration_value", ""),
        "result": market.get("result", ""),
        "volume": market.get("volume_fp") or market.get("volume", ""),
        "open_interest": market.get("open_interest_fp") or market.get("open_interest", ""),
        "last_price_dollars": market.get("last_price_dollars", ""),
    }


def pull_series(
    client: KalshiClient,
    series: str,
    max_pages: int | None,
    since: datetime | None,
) -> list[dict]:
    rows: list[dict] = []
    cursor: str | None = None
    page = 0

    since_label = since.strftime("%Y-%m-%d") if since else "all"
    print(f"\n[{series}] Paginating settled markets (1000/page, since={since_label})...")

    while True:
        for attempt in range(4):
            try:
                payload = client.list_markets_page(
                    series_ticker=series,
                    status="settled",
                    limit=1000,
                    cursor=cursor,
                )
                break
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    wait = 2 ** (attempt + 2)  # 4, 8, 16, 32 sec
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/4)...")
                    time.sleep(wait)
                else:
                    print(f"  ERROR on page {page + 1}: {exc}")
                    payload = None
                    break
        else:
            print(f"  Gave up after 4 rate-limit retries on page {page + 1}.")
            break
        if payload is None:
            break

        markets = payload.get("markets", [])
        if not markets:
            break

        cursor = payload.get("cursor")
        page += 1
        reached_cutoff = False

        for m in markets:
            row = _flatten(m, series)
            if since is not None and row["close_time"]:
                try:
                    ct = datetime.fromisoformat(row["close_time"])
                    if ct < since:
                        reached_cutoff = True
                        continue
                except ValueError:
                    pass
            rows.append(row)

        oldest = min((r["close_time"] for r in rows[-len(markets):] if r["close_time"]), default="?")
        print(f"  Page {page}: +{len(markets)} markets | oldest_close={oldest} | total={len(rows)}")

        time.sleep(0.3)  # stay under rate limit

        if not cursor:
            print(f"  No more pages.")
            break
        if reached_cutoff:
            print(f"  Reached --since cutoff {since_label}.")
            break
        if max_pages is not None and page >= max_pages:
            print(f"  Stopped at --max-pages {max_pages}.")
            break

    print(f"[{series}] Done: {len(rows)} markets collected across {page} pages.")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull settled Kalshi BTC market metadata")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages per series (default: all)")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Stop collecting markets that closed before this date (YYYY-MM-DD, UTC). E.g. 2026-01-04",
    )
    args = parser.parse_args()

    since_dt: datetime | None = None
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    client = KalshiClient()

    for series in SERIES:
        out_path = OUT_DIR / f"settled_markets_{series.lower()}.csv"
        rows = pull_series(client, series, max_pages=args.max_pages, since=since_dt)
        if rows:
            write_csv(rows, out_path)
        else:
            print(f"[{series}] No data collected.")


if __name__ == "__main__":
    main()
