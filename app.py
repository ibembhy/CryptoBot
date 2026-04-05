from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kalshi_btc_bot.dashboard import (
    build_live_signal_table,
    build_fills_table,
    build_latest_snapshot_table,
    format_dashboard_time,
    build_positions_table,
    load_paper_trading_state,
    load_recent_snapshots,
)
from kalshi_btc_bot.cli import build_engine
from kalshi_btc_bot.settings import load_settings
from kalshi_btc_bot.storage.snapshots import SnapshotStore


def snapshot_store_dsn(settings) -> str:
    dsn = settings.collector.get("postgres_dsn")
    if not dsn:
        raise RuntimeError("collector.postgres_dsn must be configured; SQLite snapshot fallback is no longer supported in runtime paths.")
    return str(dsn)


st.set_page_config(page_title="Kalshi BTC Bot", layout="wide")


@st.cache_resource
def load_runtime():
    settings = load_settings()
    store = SnapshotStore(snapshot_store_dsn(settings), read_only=True)
    _, engine = build_engine(attach_calibration=False, attach_repricing_profile=False)
    return settings, store, engine


@st.cache_data(ttl=30, show_spinner=False)
def load_recent_snapshots_cached(
    snapshot_store_path: str,
    *,
    series_ticker: str,
    lookback_hours: int,
    limit: int | None,
):
    store = SnapshotStore(snapshot_store_path, read_only=True)
    return load_recent_snapshots(store, series_ticker=series_ticker, lookback_hours=lookback_hours, limit=limit)


@st.cache_data(ttl=10, show_spinner=False)
def load_paper_state_cached(path: str):
    return load_paper_trading_state(path)


@st.cache_data(ttl=30, show_spinner=False)
def build_latest_snapshot_table_cached(snapshots):
    return build_latest_snapshot_table(snapshots, reference_time=datetime.now(timezone.utc))


def build_live_signal_table_cached(
    snapshots,
    *,
    settings_raw,
    engine,
):
    return build_live_signal_table(
        engine=engine,
        snapshots=snapshots,
        volatility_window=int(settings_raw["data"]["volatility_window"]),
        annualization_factor=float(settings_raw["data"]["annualization_factor"]),
    )


def summarize_paper_state(paper_state: dict[str, object], *, starting_capital: float) -> dict[str, object]:
    closed_positions = list(paper_state.get("closed_positions", []) or [])
    native_closed_positions = [
        position
        for position in closed_positions
        if str(position.get("strategy_mode", "")) != "synthetic_from_fills"
        and str(position.get("exit_trigger", "")) != "reconstructed_from_fills"
    ]
    realized_pnl = float(paper_state.get("realized_pnl", 0.0) or 0.0)
    closed_count = len(native_closed_positions)
    winning_closed = 0
    for position in native_closed_positions:
        pnl = float(position.get("realized_pnl", 0.0) or 0.0)
        if pnl > 0:
            winning_closed += 1
    avg_pnl = realized_pnl / closed_count if closed_count else 0.0
    win_rate = (winning_closed / closed_count * 100.0) if closed_count else 0.0
    status = "Winning" if realized_pnl > 0 else "Losing" if realized_pnl < 0 else "Flat"
    current_equity = starting_capital + realized_pnl
    return_pct = (realized_pnl / starting_capital * 100.0) if starting_capital else 0.0
    return {
        "status": status,
        "starting_capital": starting_capital,
        "realized_pnl": realized_pnl,
        "current_equity": current_equity,
        "return_pct": return_pct,
        "avg_pnl_per_trade": avg_pnl,
        "win_rate": win_rate,
        "open_positions": len(paper_state.get("open_positions", [])),
        "closed_positions": closed_count,
        "reconstructed_closed_positions": len(closed_positions) - closed_count,
        "session_notional": float(paper_state.get("session_notional", 0.0) or 0.0),
        "updated_at": format_dashboard_time(paper_state.get("updated_at") or "-"),
    }


def build_series_overview_rows(settings) -> list[dict[str, object]]:
    configured_series = list(
        dict.fromkeys(settings.collector.get("series_tickers") or [str(settings.collector["series_ticker"])]),
    )
    rows: list[dict[str, object]] = []
    for series_ticker in configured_series:
        paper_ledger_path = paper_ledger_path_for_series(settings, series_ticker)
        if paper_ledger_path is None:
            continue
        paper_state = load_paper_state_cached(paper_ledger_path)
        summary = summarize_paper_state(
            paper_state,
            starting_capital=paper_starting_capital_for_series(settings, series_ticker),
        )
        rows.append(
            {
                "series": series_ticker,
                "status": summary["status"],
                "realized_pnl": round(float(summary["realized_pnl"]), 2),
                "equity": round(float(summary["current_equity"]), 2),
                "return_pct": round(float(summary["return_pct"]), 2),
                "win_rate": f'{summary["win_rate"]:.1f}%',
                "open_closed": f'{summary["open_positions"]} / {summary["closed_positions"]}',
                "fills": len(paper_state.get("fills", [])),
                "updated_at": summary["updated_at"],
            }
        )
    return rows


def paper_ledger_path_for_series(settings, series_ticker: str) -> str | None:
    ledgers = settings.raw.get("paper", {}).get("series_ledgers", {})
    if isinstance(ledgers, dict):
        path = ledgers.get(series_ticker)
        if path:
            return str(path)
    default_series = str(settings.paper.get("series_ticker", settings.collector["series_ticker"]))
    if series_ticker == default_series:
        return str(settings.paper["ledger_path"])
    return None


def paper_starting_capital_for_series(settings, series_ticker: str) -> float:
    capitals = settings.raw.get("paper", {}).get("series_starting_capitals", {})
    if isinstance(capitals, dict) and capitals.get(series_ticker) is not None:
        return float(capitals[series_ticker])
    return float(settings.paper.get("starting_capital", 100.0))


def render_series_view(
    *,
    series_ticker: str,
    settings,
    engine,
    snapshot_store_path: str,
    lookback_hours: int,
    snapshot_limit: int,
) -> None:
    quick_limit = max(snapshot_limit * 5, 100)
    recent_snapshots = load_recent_snapshots_cached(
        snapshot_store_path,
        series_ticker=series_ticker,
        lookback_hours=lookback_hours,
        limit=quick_limit,
    )
    latest_snapshot_table = build_latest_snapshot_table_cached(recent_snapshots)
    active_markets_seen = len(latest_snapshot_table)
    latest_seen = "-"
    if not latest_snapshot_table.empty and "observed_at" in latest_snapshot_table.columns:
        latest_seen = str(latest_snapshot_table.iloc[0].get("observed_at", "-"))

    st.subheader("Market Feed Health")
    feed_cols = st.columns(3)
    feed_cols[0].metric("Series", series_ticker)
    feed_cols[1].metric("Active Markets Seen", active_markets_seen)
    feed_cols[2].metric("Latest Snapshot", latest_seen)
    if latest_snapshot_table.empty:
        st.info("No recent market snapshots available.")
    else:
        st.dataframe(latest_snapshot_table.head(min(snapshot_limit, 5)), use_container_width=True, hide_index=True)

    st.subheader("Live Edge")
    signal_table = build_live_signal_table_cached(
        recent_snapshots,
        settings_raw=settings.raw,
        engine=engine,
    )
    if signal_table.empty:
        st.info("No live edge rows available right now.")
    else:
        top_row = signal_table.iloc[0]
        edge_cols = st.columns(5)
        edge_cols[0].metric("Market", str(top_row.get("market_ticker", "-")))
        edge_cols[1].metric("Action", str(top_row.get("action", "-")))
        edge_cols[2].metric("Side", str(top_row.get("side", "-")))
        edge_value = top_row.get("edge")
        edge_cols[3].metric("Edge", "-" if pd.isna(edge_value) else f"{float(edge_value):.4f}")
        edge_cols[4].metric("Reason", str(top_row.get("reason_bucket", "-")))
        full_reason = str(top_row.get("reason", "-"))
        if str(top_row.get("action", "")) == "no_action":
            st.warning(f"Current live reject reason: {full_reason}")
        else:
            st.success(f"Current live entry reason: {full_reason}")
        live_edge_columns = [
            "market_ticker",
            "observed_at",
            "minutes_to_expiry",
            "yes_ask_cents",
            "side",
            "action",
            "edge",
            "model_probability",
            "market_probability",
            "quality_score",
            "reason",
        ]
        visible_columns = [column for column in live_edge_columns if column in signal_table.columns]
        st.dataframe(signal_table[visible_columns].head(min(snapshot_limit, 10)), use_container_width=True, hide_index=True)

    paper_ledger_path = paper_ledger_path_for_series(settings, series_ticker)
    if paper_ledger_path is not None:
        paper_state = load_paper_state_cached(paper_ledger_path)
        open_positions_table = build_positions_table(paper_state.get("open_positions", []))
        closed_positions_table = build_positions_table(paper_state.get("closed_positions", []))
        fills_table = build_fills_table(paper_state.get("fills", []))
        paper_summary = summarize_paper_state(
            paper_state,
            starting_capital=paper_starting_capital_for_series(settings, series_ticker),
        )

        st.subheader("Paper Trading State")
        paper_cols = st.columns(6)
        paper_cols[0].metric("Paper Status", paper_summary["status"])
        paper_cols[1].metric("Realized PnL", round(float(paper_summary["realized_pnl"]), 2))
        paper_cols[2].metric("Paper Equity", round(float(paper_summary["current_equity"]), 2))
        paper_cols[3].metric("Win Rate", f'{paper_summary["win_rate"]:.1f}%')
        paper_cols[4].metric("Open / Closed", f'{paper_summary["open_positions"]} / {paper_summary["closed_positions"]}')
        paper_cols[5].metric("Ledger Updated", paper_summary["updated_at"])

        if str(paper_state.get("data_feed_status") or "unknown").strip().lower() == "degraded":
            last_success = format_dashboard_time(paper_state.get("last_successful_feature_build_at") or "-")
            error_text = str(paper_state.get("last_coinbase_error") or "Coinbase data unavailable.")
            st.error(
                "Coinbase data issue detected. Paper trading is paused until data returns.\n\n"
                f"Last successful Coinbase feature build: {last_success}\n\n"
                f"Latest error: {error_text}"
            )

        extra_paper_cols = st.columns(5)
        extra_paper_cols[0].metric("Starting Paper Capital", round(float(paper_summary["starting_capital"]), 2))
        extra_paper_cols[1].metric("Paper Return", f'{paper_summary["return_pct"]:.2f}%')
        extra_paper_cols[2].metric("Avg PnL / Closed Trade", round(float(paper_summary["avg_pnl_per_trade"]), 2))
        extra_paper_cols[3].metric(
            "Calibration",
            "Enabled" if bool(settings.raw.get("calibration", {}).get("enabled", False)) else "Off",
        )
        extra_paper_cols[4].metric("Recorded Fills", len(paper_state.get("fills", [])))

        st.caption(
            "Paper equity is simulated from starting capital plus realized paper PnL. "
            "This keeps the live paper logic the same while making results easier to read."
        )
        st.caption(f"Session Notional: {round(float(paper_summary['session_notional']), 2)}")

        paper_left, paper_right = st.columns(2)
        with paper_left:
            st.caption("Open Paper Positions")
            if open_positions_table.empty:
                st.info("No open paper positions yet.")
            else:
                st.dataframe(open_positions_table, use_container_width=True, hide_index=True)
        with paper_right:
            st.caption("Recent Closed Paper Positions")
            if closed_positions_table.empty:
                st.info("No closed paper positions yet.")
            else:
                st.dataframe(closed_positions_table.head(20), use_container_width=True, hide_index=True)

        st.caption("Recent Paper Fills")
        if fills_table.empty:
            st.info("No paper fills recorded yet.")
        else:
            st.dataframe(fills_table.head(30), use_container_width=True, hide_index=True)
    else:
        st.subheader("Paper Trading State")
        st.info(
            f"Live paper trading is not enabled for `{series_ticker}` yet. "
            "This tab shows research and signal data only."
        )
    st.caption("Detailed research tables are temporarily disabled to keep the dashboard stable and fast.")


def set_query_param_if_changed(query_params, key: str, value: str) -> None:
    current = query_params.get(key)
    if isinstance(current, list):
        current = current[0] if current else None
    current_text = None if current is None else str(current)
    if current_text != str(value):
        query_params[key] = str(value)


def main() -> None:
    settings, store, engine = load_runtime()
    snapshot_store_path = snapshot_store_dsn(settings)
    query_params = st.query_params
    dashboard_username = os.getenv("KALSHI_BTC_DASHBOARD_USERNAME", "").strip()
    dashboard_password = os.getenv("KALSHI_BTC_DASHBOARD_PASSWORD", "").strip()
    if dashboard_username and dashboard_password:
        auth_token = hashlib.sha256(f"{dashboard_username}:{dashboard_password}".encode("utf-8")).hexdigest()
        authenticated = (
            st.session_state.get("dashboard_authenticated", False)
            or str(query_params.get("auth_token", "")) == auth_token
        )
        if not authenticated:
            st.title("Kalshi BTC Bot Login")
            with st.form("dashboard_login"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In")
            if submitted:
                if username == dashboard_username and password == dashboard_password:
                    st.session_state["dashboard_authenticated"] = True
                    query_params["auth_token"] = auth_token
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            return

    auto_refresh_default = str(query_params.get("auto_refresh", "false")).lower() == "true"
    try:
        refresh_seconds_default = int(query_params.get("refresh_seconds", "10"))
    except (TypeError, ValueError):
        refresh_seconds_default = 10
    refresh_options = [5, 10, 15, 30, 60, 120, 300]
    if refresh_seconds_default not in refresh_options:
        refresh_seconds_default = 10

    st.title("Kalshi BTC Bot")
    st.caption("Live research dashboard for current BTC Kalshi markets and replay-ready data.")
    st.caption("All timestamps and dates below are shown in Eastern Time (ET).")

    configured_series = list(dict.fromkeys(settings.collector.get("series_tickers") or [str(settings.collector["series_ticker"])]))
    default_series = str(query_params.get("series", configured_series[0] if configured_series else settings.collector["series_ticker"]))
    if default_series not in configured_series and configured_series:
        default_series = configured_series[0]

    with st.sidebar:
        st.header("View")
        lookback_hours = st.slider("Recent snapshot window (hours)", min_value=1, max_value=24, value=6)
        snapshot_limit = st.slider("Market rows", min_value=10, max_value=100, value=30, step=10)
        auto_refresh = st.checkbox("Auto refresh", value=auto_refresh_default)
        refresh_seconds = st.select_slider(
            "Refresh interval",
            options=refresh_options,
            value=refresh_seconds_default,
            format_func=lambda value: f"{value // 60} min" if value >= 60 else f"{value}s",
            disabled=not auto_refresh,
        )
        set_query_param_if_changed(query_params, "auto_refresh", "true" if auto_refresh else "false")
        set_query_param_if_changed(query_params, "refresh_seconds", str(refresh_seconds))
        if auto_refresh:
            st.caption(f"Dashboard reloads every {refresh_seconds} seconds.")
            components.html(
                f"""
                <script>
                setTimeout(function() {{
                    window.parent.location.reload();
                }}, {int(refresh_seconds * 1000)});
                </script>
                """,
                height=0,
                width=0,
            )

        dashboard_limit = max(snapshot_limit * 10, 500)
    st.caption(f"Dashboard refreshed at {format_dashboard_time(datetime.now(timezone.utc))}")

    st.subheader("Current Default Setup")
    setup_cols = st.columns(4)
    setup_cols[0].metric("Model", str(settings.model["name"]))
    setup_cols[1].metric("Price Band", f'{settings.signal["min_contract_price_cents"]}-{settings.signal["max_contract_price_cents"]}c')
    setup_cols[2].metric("Near Money", f'{settings.signal["max_near_money_bps"]} bps')
    setup_cols[3].metric("Max Minutes To Expiry", settings.collector["max_minutes_to_expiry"])

    overview_rows = build_series_overview_rows(settings)
    if overview_rows:
        st.subheader("Series At A Glance")
        st.dataframe(overview_rows, use_container_width=True, hide_index=True)

    selected_series = st.radio(
        "Series View",
        configured_series,
        index=configured_series.index(default_series) if configured_series else 0,
        horizontal=True,
        key="series_view_selector",
    )
    set_query_param_if_changed(query_params, "series", selected_series)
    render_series_view(
        series_ticker=selected_series,
        settings=settings,
        engine=engine,
        snapshot_store_path=snapshot_store_path,
        lookback_hours=lookback_hours,
        snapshot_limit=snapshot_limit,
    )


if __name__ == "__main__":
    main()
