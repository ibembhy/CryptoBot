from __future__ import annotations

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
    settings, engine = build_engine()
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    return settings, engine, store


def main() -> None:
    settings, engine, store = load_runtime()
    query_params = st.query_params
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

    summary = store.dataset_summary(series_ticker=series_ticker, reference_time=datetime.now(timezone.utc))
    recent_snapshots = load_recent_snapshots(store, series_ticker=series_ticker, lookback_hours=lookback_hours)
    latest_snapshot_table = build_latest_snapshot_table(recent_snapshots)
    paper_state = load_paper_trading_state(str(settings.paper["ledger_path"]))
    open_positions_table = build_positions_table(paper_state.get("open_positions", []))
    closed_positions_table = build_positions_table(paper_state.get("closed_positions", []))
    fills_table = build_fills_table(paper_state.get("fills", []))
    signal_table = build_live_signal_table(
        engine=engine,
        snapshots=recent_snapshots,
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
    )

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
    paper_cols = st.columns(5)
    paper_cols[0].metric("Realized PnL", paper_state.get("realized_pnl", 0.0))
    paper_cols[1].metric("Session Notional", paper_state.get("session_notional", 0.0))
    paper_cols[2].metric("Open Positions", len(paper_state.get("open_positions", [])))
    paper_cols[3].metric("Closed Positions", len(paper_state.get("closed_positions", [])))
    paper_cols[4].metric("Ledger Updated", paper_state.get("updated_at") or "-")

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
