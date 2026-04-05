from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from scripts.analysis.loaders import AnalysisContext, load_all_settlements


def run_data_quality_profile(context: AnalysisContext, series_list: list[str]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    summary_lines = [f"DB: {context.db_path}", f"Series: {', '.join(series_list)}", ""]

    with sqlite3.connect(context.db_path) as conn:
        settlements = load_all_settlements()[["ticker", "series"]].drop_duplicates()
        for series in series_list:
            prefix = series.lower()
            snapshot_keys = pd.read_sql_query(
                """
                SELECT market_ticker, MIN(observed_at) AS first_seen, MAX(observed_at) AS last_seen, COUNT(*) AS snapshot_count
                FROM market_snapshots
                WHERE series_ticker = ?
                GROUP BY market_ticker
                """,
                conn,
                params=[series],
            )

            no_settlement = snapshot_keys.merge(
                settlements[settlements["series"] == series],
                how="left",
                left_on="market_ticker",
                right_on="ticker",
            )
            no_settlement = no_settlement[no_settlement["ticker"].isna()].drop(columns=["ticker", "series"], errors="ignore")
            no_settlement_path = context.output_dir / f"{prefix}_dq_no_settlement_match.csv"
            no_settlement.to_csv(no_settlement_path, index=False)
            outputs[f"{series}_no_settlement"] = no_settlement_path

            duplicates = pd.read_sql_query(
                """
                SELECT market_ticker, observed_at, COUNT(*) AS duplicate_count
                FROM market_snapshots
                WHERE series_ticker = ?
                GROUP BY market_ticker, observed_at
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC, market_ticker, observed_at
                """,
                conn,
                params=[series],
            )
            duplicates_path = context.output_dir / f"{prefix}_dq_duplicate_rows.csv"
            duplicates.to_csv(duplicates_path, index=False)
            outputs[f"{series}_duplicates"] = duplicates_path

            frequency = pd.read_sql_query(
                """
                WITH ordered AS (
                    SELECT
                        market_ticker,
                        observed_at,
                        LAG(observed_at) OVER (PARTITION BY market_ticker ORDER BY observed_at) AS prev_observed_at
                    FROM market_snapshots
                    WHERE series_ticker = ?
                )
                SELECT
                    market_ticker,
                    COUNT(*) AS snapshot_count,
                    AVG((julianday(observed_at) - julianday(prev_observed_at)) * 86400.0) AS avg_gap_seconds,
                    MIN((julianday(observed_at) - julianday(prev_observed_at)) * 86400.0) AS min_gap_seconds,
                    MAX((julianday(observed_at) - julianday(prev_observed_at)) * 86400.0) AS max_gap_seconds
                FROM ordered
                WHERE prev_observed_at IS NOT NULL
                GROUP BY market_ticker
                ORDER BY snapshot_count DESC
                """,
                conn,
                params=[series],
            )
            frequency_path = context.output_dir / f"{prefix}_dq_snapshot_frequency.csv"
            frequency.to_csv(frequency_path, index=False)
            outputs[f"{series}_frequency"] = frequency_path

            stale_quotes = pd.read_sql_query(
                """
                WITH ordered AS (
                    SELECT
                        market_ticker,
                        observed_at,
                        volume,
                        open_interest,
                        yes_bid,
                        yes_ask,
                        no_bid,
                        no_ask,
                        LAG(yes_bid) OVER (PARTITION BY market_ticker ORDER BY observed_at) AS prev_yes_bid,
                        LAG(yes_ask) OVER (PARTITION BY market_ticker ORDER BY observed_at) AS prev_yes_ask,
                        LAG(no_bid) OVER (PARTITION BY market_ticker ORDER BY observed_at) AS prev_no_bid,
                        LAG(no_ask) OVER (PARTITION BY market_ticker ORDER BY observed_at) AS prev_no_ask
                    FROM market_snapshots
                    WHERE series_ticker = ?
                )
                SELECT
                    market_ticker,
                    observed_at,
                    volume,
                    open_interest,
                    yes_bid,
                    yes_ask,
                    no_bid,
                    no_ask,
                    CASE
                        WHEN COALESCE(volume, 0) = 0
                         AND prev_yes_bid IS NOT NULL
                         AND yes_bid = prev_yes_bid
                         AND yes_ask = prev_yes_ask
                         AND no_bid = prev_no_bid
                         AND no_ask = prev_no_ask
                        THEN 1 ELSE 0
                    END AS is_stale_zero_volume_quote,
                    CASE
                        WHEN yes_bid IS NOT NULL AND yes_ask IS NOT NULL AND yes_bid = yes_ask THEN 1 ELSE 0
                    END AS is_yes_locked,
                    CASE
                        WHEN no_bid IS NOT NULL AND no_ask IS NOT NULL AND no_bid = no_ask THEN 1 ELSE 0
                    END AS is_no_locked
                FROM ordered
                WHERE COALESCE(volume, 0) = 0
                  AND prev_yes_bid IS NOT NULL
                  AND yes_bid = prev_yes_bid
                  AND yes_ask = prev_yes_ask
                  AND no_bid = prev_no_bid
                  AND no_ask = prev_no_ask
                ORDER BY observed_at
                """,
                conn,
                params=[series],
            )
            stale_quotes_path = context.output_dir / f"{prefix}_dq_stale_quotes.csv"
            stale_quotes.to_csv(stale_quotes_path, index=False)
            outputs[f"{series}_stale_quotes"] = stale_quotes_path

            source_distribution = pd.read_sql_query(
                """
                SELECT source, COUNT(*) AS snapshot_count
                FROM market_snapshots
                WHERE series_ticker = ?
                GROUP BY source
                ORDER BY snapshot_count DESC, source
                """,
                conn,
                params=[series],
            )
            source_distribution_path = context.output_dir / f"{prefix}_dq_source_distribution.csv"
            source_distribution.to_csv(source_distribution_path, index=False)
            outputs[f"{series}_source_distribution"] = source_distribution_path

            summary_lines.extend(
                [
                    f"[{series}]",
                    f"markets_with_no_settlement_match={len(no_settlement):,}",
                    f"duplicate_market_timestamp_rows={len(duplicates):,}",
                    f"stale_quote_rows_refined={len(stale_quotes):,}",
                    f"sources={', '.join(source_distribution['source'].astype(str).tolist()) or 'none'}",
                    "",
                ]
            )

    summary_path = context.output_dir / "data_quality_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs
