from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

from kalshi_btc_bot.cli import build_engine
from kalshi_btc_bot.dashboard import (
    build_fills_table,
    build_latest_snapshot_table,
    build_live_signal_table,
    build_positions_table,
    load_paper_trading_state,
    load_recent_snapshots,
)
from kalshi_btc_bot.storage.snapshots import SnapshotStore


st.set_page_config(page_title="Kalshi BTC Bot", layout="wide")


@st.cache_resource
def load_runtime():
    settings, engine = build_engine(attach_calibration=False)
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    return settings, engine, store


@st.cache_data(ttl=5, show_spinner=False)
def load_summary_cached(sqlite_path: str, series_ticker: str) -> dict[str, object]:
    store = SnapshotStore(sqlite_path)
    return store.dataset_summary(series_ticker=series_ticker, reference_time=datetime.now(timezone.utc))


@st.cache_data(ttl=5, show_spinner=False)
def load_recent_snapshots_cached(
    sqlite_path: str,
    *,
    series_ticker: str,
    lookback_hours: int,
    limit: int | None,
):
    store = SnapshotStore(sqlite_path)
    return load_recent_snapshots(store, series_ticker=series_ticker, lookback_hours=lookback_hours, limit=limit)


@st.cache_data(ttl=5, show_spinner=False)
def load_paper_state_cached(path: str):
    return load_paper_trading_state(path)


@st.cache_data(ttl=5, show_spinner=False)
def build_signal_table_cached(*, _engine, snapshots, volatility_window: int, annualization_factor: float):
    return build_live_signal_table(
        engine=_engine,
        snapshots=snapshots,
        volatility_window=volatility_window,
        annualization_factor=annualization_factor,
    )


@st.cache_data(ttl=5, show_spinner=False)
def build_latest_snapshot_table_cached(snapshots):
    return build_latest_snapshot_table(snapshots)


def summarize_paper_state(paper_state: dict[str, object]) -> dict[str, object]:
    closed_positions = paper_state.get("closed_positions", [])
    realized_pnl = float(paper_state.get("realized_pnl", 0.0) or 0.0)
    closed_count = len(closed_positions)
    winning_closed = 0
    for position in closed_positions:
        pnl = float(position.get("realized_pnl", 0.0) or 0.0)
        if pnl > 0:
            winning_closed += 1
    avg_pnl = realized_pnl / closed_count if closed_count else 0.0
    win_rate = (winning_closed / closed_count * 100.0) if closed_count else 0.0
    status = "Winning" if realized_pnl > 0 else "Losing" if realized_pnl < 0 else "Flat"
    return {
        "status": status,
        "realized_pnl": realized_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "win_rate": win_rate,
        "open_positions": len(paper_state.get("open_positions", [])),
        "closed_positions": closed_count,
        "session_notional": float(paper_state.get("session_notional", 0.0) or 0.0),
        "updated_at": paper_state.get("updated_at") or "-",
    }


def summarize_signal_table(signal_table) -> dict[str, object]:
    if signal_table.empty:
        return {
            "candidate_count": 0,
            "tradeable_count": 0,
            "best_market_ticker": "-",
            "best_side": "-",
            "best_edge": 0.0,
            "best_quality_score": 0.0,
        }

    candidate_count = len(signal_table)
    tradeable = signal_table[signal_table["action"] != "no_action"] if "action" in signal_table.columns else signal_table
    if tradeable.empty:
        return {
            "candidate_count": candidate_count,
            "tradeable_count": 0,
            "best_market_ticker": "-",
            "best_side": "-",
            "best_edge": 0.0,
            "best_quality_score": 0.0,
        }

    best = tradeable.sort_values(by=["quality_score", "edge"], ascending=[False, False]).iloc[0]
    return {
        "candidate_count": candidate_count,
        "tradeable_count": len(tradeable),
        "best_market_ticker": best.get("market_ticker", "-"),
        "best_side": best.get("side", "-") or "-",
        "best_edge": float(best.get("edge", 0.0) or 0.0),
        "best_quality_score": float(best.get("quality_score", 0.0) or 0.0),
    }


def main() -> None:
    settings, engine, store = load_runtime()
    sqlite_path = str(settings.collector["sqlite_path"])
    paper_ledger_path = str(settings.paper["ledger_path"])
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

    st.title("Kalshi BTC Bot")
    st.caption("Live research dashboard for current BTC Kalshi markets and replay-ready data.")

    with st.sidebar:
        st.header("View")
        series_ticker = st.text_input("Series", value=str(settings.collector["series_ticker"]))
        lookback_hours = st.slider("Recent snapshot window (hours)", min_value=1, max_value=24, value=6)
        signal_limit = st.slider("Signal rows", min_value=10, max_value=100, value=30, step=10)
        snapshot_limit = st.slider("Market rows", min_value=10, max_value=100, value=30, step=10)
        auto_refresh = st.checkbox("Auto refresh", value=auto_refresh_default)
        refresh_seconds = st.slider(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=60,
            value=refresh_seconds_default,
            step=5,
            disabled=not auto_refresh,
        )
        query_params["auto_refresh"] = "true" if auto_refresh else "false"
        query_params["refresh_seconds"] = str(refresh_seconds)
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

    dashboard_limit = max(signal_limit * 25, snapshot_limit * 25, 2000)
    summary = load_summary_cached(sqlite_path, series_ticker)
    recent_snapshots = load_recent_snapshots_cached(
        sqlite_path,
        series_ticker=series_ticker,
        lookback_hours=lookback_hours,
        limit=dashboard_limit,
    )
    latest_snapshot_table = build_latest_snapshot_table_cached(recent_snapshots)
    paper_state = load_paper_state_cached(paper_ledger_path)
    open_positions_table = build_positions_table(paper_state.get("open_positions", []))
    closed_positions_table = build_positions_table(paper_state.get("closed_positions", []))
    fills_table = build_fills_table(paper_state.get("fills", []))
    signal_table = build_signal_table_cached(
        _engine=engine,
        snapshots=recent_snapshots,
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
    )
    paper_summary = summarize_paper_state(paper_state)
    signal_summary = summarize_signal_table(signal_table)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Snapshots", summary["snapshot_count"])
    metric_cols[1].metric("Markets", summary["unique_markets"])
    metric_cols[2].metric("Settled", summary["settled_markets"])
    metric_cols[3].metric("Replay Ready", summary["replay_ready_markets"])
    metric_cols[4].metric("Expired", summary["expired_markets"])
    metric_cols[5].metric("Last Seen", summary["last_observed_at"] or "-")
    st.caption(f"Dashboard refreshed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    st.subheader("Current Default Setup")
    setup_cols = st.columns(4)
    setup_cols[0].metric("Model", str(settings.model["name"]))
    setup_cols[1].metric("Price Band", f'{settings.signal["min_contract_price_cents"]}-{settings.signal["max_contract_price_cents"]}c')
    setup_cols[2].metric("Near Money", f'{settings.signal["max_near_money_bps"]} bps')
    setup_cols[3].metric("Max Minutes To Expiry", settings.collector["max_minutes_to_expiry"])

    st.subheader("Paper Trading State")
    paper_cols = st.columns(6)
    paper_cols[0].metric("Paper Status", paper_summary["status"])
    paper_cols[1].metric("Realized PnL", round(float(paper_summary["realized_pnl"]), 2))
    paper_cols[2].metric("Avg PnL / Closed Trade", round(float(paper_summary["avg_pnl_per_trade"]), 2))
    paper_cols[3].metric("Win Rate", f'{paper_summary["win_rate"]:.1f}%')
    paper_cols[4].metric("Open / Closed", f'{paper_summary["open_positions"]} / {paper_summary["closed_positions"]}')
    paper_cols[5].metric("Ledger Updated", paper_summary["updated_at"])

    extra_paper_cols = st.columns(3)
    extra_paper_cols[0].metric("Session Notional", round(float(paper_summary["session_notional"]), 2))
    extra_paper_cols[1].metric(
        "Calibration",
        "Enabled" if bool(settings.raw.get("calibration", {}).get("enabled", False)) else "Off",
    )
    extra_paper_cols[2].metric("Recorded Fills", len(paper_state.get("fills", [])))

    st.subheader("Signal Summary")
    signal_cols = st.columns(5)
    signal_cols[0].metric("Candidates", signal_summary["candidate_count"])
    signal_cols[1].metric("Tradeable Now", signal_summary["tradeable_count"])
    signal_cols[2].metric("Top Candidate", signal_summary["best_market_ticker"])
    signal_cols[3].metric("Top Side", str(signal_summary["best_side"]).upper())
    signal_cols[4].metric(
        "Top Edge / Quality",
        f'{signal_summary["best_edge"]:.4f} / {signal_summary["best_quality_score"]:.3f}',
    )

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
            st.dataframe(closed_positions_table.tail(20), use_container_width=True, hide_index=True)

    st.caption("Recent Paper Fills")
    if fills_table.empty:
        st.info("No paper fills recorded yet.")
    else:
        st.dataframe(fills_table.head(30), use_container_width=True, hide_index=True)

    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("Current Candidate Signals")
        if signal_table.empty:
            st.info("No live candidates in the selected recent window.")
        else:
            st.dataframe(signal_table.head(signal_limit), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Latest Market Snapshots")
        if latest_snapshot_table.empty:
            st.info("No recent market snapshots available.")
        else:
            st.dataframe(latest_snapshot_table.head(snapshot_limit), use_container_width=True, hide_index=True)

    st.subheader("Contract Type Breakdown")
    st.dataframe(summary["contract_types"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
