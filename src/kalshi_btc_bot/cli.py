from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.backtest.replay import build_replay_dataset
from kalshi_btc_bot.collectors.backfill import BackfillConfig, CandlestickBackfillService
from kalshi_btc_bot.collectors.hybrid import HybridCollector, HybridCollectorConfig
from kalshi_btc_bot.collectors.settlements import SettlementEnricher, SettlementEnrichmentConfig
from kalshi_btc_bot.data.coinbase import CoinbaseClient, default_history_window
from kalshi_btc_bot.markets.auth import KalshiAuthConfig, KalshiAuthSigner
from kalshi_btc_bot.data.features import build_feature_frame
from kalshi_btc_bot.markets.kalshi import KalshiClient
from kalshi_btc_bot.models.gbm_threshold import GBMThresholdModel
from kalshi_btc_bot.models.latency_repricing import LatencyRepricingModel
from kalshi_btc_bot.reports.comparison import (
    build_grid_search_report,
    build_model_comparison_report,
    clone_engine_with_mode,
    filter_snapshots_for_focus,
)
from kalshi_btc_bot.settings import load_settings
from kalshi_btc_bot.signals.calibration import build_engine_calibrators
from kalshi_btc_bot.signals.fusion import FusionConfig
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.live_paper import LivePaperConfig, LivePaperTrader
from kalshi_btc_bot.trading.risk import RiskConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kalshi BTC probability bot MVP")
    parser.add_argument("--log-level", type=str, default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-config", help="Print resolved configuration")
    subparsers.add_parser("spot", help="Fetch current Coinbase BTC spot price")
    demo = subparsers.add_parser("demo-backtest", help="Run a lightweight live-data connectivity check")
    demo.add_argument("--hours", type=int, default=24)
    collect = subparsers.add_parser("collect-once", help="Bootstrap Kalshi BTC markets and persist snapshots once")
    collect.add_argument("--series", type=str, default=None)
    collect.add_argument("--max-minutes-to-expiry", type=int, default=None)
    collect.add_argument("--min-minutes-to-expiry", type=int, default=None)
    run = subparsers.add_parser("collect-forever", help="Run the hybrid Kalshi BTC collector continuously")
    run.add_argument("--series", type=str, default=None)
    run.add_argument("--max-minutes-to-expiry", type=int, default=None)
    run.add_argument("--min-minutes-to-expiry", type=int, default=None)
    paper_once = subparsers.add_parser("paper-trade-once", help="Run one live paper-trading cycle from current market data")
    paper_once.add_argument("--series", type=str, default=None)
    paper_once.add_argument("--max-minutes-to-expiry", type=int, default=None)
    paper_once.add_argument("--min-minutes-to-expiry", type=int, default=None)
    paper_forever = subparsers.add_parser("paper-trade-forever", help="Continuously paper trade live Kalshi BTC markets")
    paper_forever.add_argument("--series", type=str, default=None)
    paper_forever.add_argument("--max-minutes-to-expiry", type=int, default=None)
    paper_forever.add_argument("--min-minutes-to-expiry", type=int, default=None)
    backfill = subparsers.add_parser("backfill-candles", help="Backfill Kalshi candlestick history into SQLite snapshots")
    backfill.add_argument("--series", type=str, default=None)
    backfill.add_argument("--start-ts", type=int, required=True)
    backfill.add_argument("--end-ts", type=int, required=True)
    backfill.add_argument("--period-interval", type=int, default=None)
    backfill.add_argument("--max-markets", type=int, default=None)
    backfill.add_argument("--historical", action="store_true")
    historical_list = subparsers.add_parser("historical-list-markets", help="List settled historical Kalshi BTC markets for a time window")
    historical_list.add_argument("--series", type=str, default=None)
    historical_list.add_argument("--start-ts", type=int, required=True)
    historical_list.add_argument("--end-ts", type=int, required=True)
    historical_list.add_argument("--max-events", type=int, default=50)
    historical_list.add_argument("--max-markets", type=int, default=200)
    subparsers.add_parser("historical-cutoff", help="Show Kalshi historical cutoff timestamps")
    historical_inspect = subparsers.add_parser("historical-inspect-market", help="Inspect one historical Kalshi market payload")
    historical_inspect.add_argument("--ticker", type=str, required=True)
    historical_candles = subparsers.add_parser("historical-fetch-candles", help="Fetch archived candlesticks for one Kalshi market")
    historical_candles.add_argument("--ticker", type=str, required=True)
    historical_candles.add_argument("--start-ts", type=int, required=True)
    historical_candles.add_argument("--end-ts", type=int, required=True)
    historical_candles.add_argument("--period-interval", type=int, default=1)
    historical_candles.add_argument("--include-latest-before-start", action="store_true")
    historical_candles.add_argument("--series", type=str, default=None)
    dataset = subparsers.add_parser("dataset-summary", help="Summarize collected Kalshi BTC snapshot history")
    dataset.add_argument("--series", type=str, default=None)
    remap = subparsers.add_parser("fix-series-ticker", help="Fix stored snapshots that used the wrong series ticker")
    remap.add_argument("--old", type=str, required=True)
    remap.add_argument("--new", type=str, required=True)
    settlements = subparsers.add_parser("enrich-settlements", help="Fetch final market outcomes and store settlement values")
    settlements.add_argument("--series", type=str, default=None)
    settlements.add_argument("--max-markets", type=int, default=None)
    settlements.add_argument("--recent-hours", type=int, default=None)
    diagnostics = subparsers.add_parser("replay-diagnostics", help="Explain why replay produced few or no trades")
    diagnostics.add_argument("--series", type=str, default=None)
    diagnostics.add_argument("--market", type=str, default=None)
    diagnostics.add_argument("--from-ts", type=str, default=None)
    diagnostics.add_argument("--to-ts", type=str, default=None)
    diagnostics.add_argument("--limit", type=int, default=None)
    replay = subparsers.add_parser("replay-backtest", help="Run a backtest from stored SQLite snapshots")
    replay.add_argument("--series", type=str, default=None)
    replay.add_argument("--market", type=str, default=None)
    replay.add_argument("--from-ts", type=str, default=None)
    replay.add_argument("--to-ts", type=str, default=None)
    replay.add_argument("--limit", type=int, default=None)
    compare = subparsers.add_parser("replay-compare-models", help="Compare gbm, latency repricing, and hybrid side by side")
    compare.add_argument("--series", type=str, default=None)
    compare.add_argument("--market", type=str, default=None)
    compare.add_argument("--from-ts", type=str, default=None)
    compare.add_argument("--to-ts", type=str, default=None)
    compare.add_argument("--limit", type=int, default=None)
    compare.add_argument("--near-money-bps", type=float, default=None)
    compare.add_argument("--max-minutes-to-expiry", type=float, default=None)
    compare.add_argument("--min-price-cents", type=int, default=None)
    compare.add_argument("--max-price-cents", type=int, default=None)
    grid = subparsers.add_parser("replay-grid-search", help="Search filter ranges and rank the best replay setups")
    grid.add_argument("--series", type=str, default=None)
    grid.add_argument("--market", type=str, default=None)
    grid.add_argument("--from-ts", type=str, default=None)
    grid.add_argument("--to-ts", type=str, default=None)
    grid.add_argument("--limit", type=int, default=None)
    grid.add_argument("--near-money-bps-values", type=str, default="100,150,200")
    grid.add_argument("--max-minutes-to-expiry-values", type=str, default="30,60,90")
    grid.add_argument("--min-price-cents-values", type=str, default="20,25")
    grid.add_argument("--max-price-cents-values", type=str, default="75,80")
    grid.add_argument("--top", type=int, default=20)
    return parser


def parse_numeric_list(raw: str, *, cast):
    values = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def build_engine():
    settings = load_settings()
    gbm_model = GBMThresholdModel(
        drift=float(settings.model["drift"]),
        volatility_floor=float(settings.model["volatility_floor"]),
    )
    latency_model = LatencyRepricingModel(
        drift=float(settings.model["drift"]),
        volatility_floor=float(settings.model["volatility_floor"]),
        persistence_factor=float(settings.model["repricing_persistence_factor"]),
        min_move_bps=float(settings.model["repricing_min_move_bps"]),
        max_move_bps=float(settings.model["repricing_max_move_bps"]),
    )
    models = {
        gbm_model.model_name: gbm_model,
        latency_model.model_name: latency_model,
    }
    requested_model = str(settings.model["name"])
    model_name = requested_model if requested_model in models else str(settings.strategy["primary_model"])
    model = models[model_name]
    signal_config = SignalConfig(
        min_edge=float(settings.signal["min_edge"]),
        min_confidence=float(settings.signal["min_confidence"]),
        min_contract_price_cents=int(settings.signal["min_contract_price_cents"]),
        max_contract_price_cents=int(settings.signal["max_contract_price_cents"]),
        max_near_money_bps=float(settings.signal["max_near_money_bps"]) if settings.signal["max_near_money_bps"] is not None else None,
        min_liquidity=float(settings.signal["min_liquidity"]),
        uncertainty_penalty=float(settings.signal["uncertainty_penalty"]),
        max_spread_cents=int(settings.signal["max_spread_cents"]),
        liquidity_penalty_per_100_volume=float(settings.signal["liquidity_penalty_per_100_volume"]),
        max_data_age_seconds=int(settings.signal["max_data_age_seconds"]),
    )
    exit_config = ExitConfig(
        take_profit_cents=int(settings.trading["take_profit_cents"]),
        stop_loss_cents=int(settings.trading["stop_loss_cents"]),
        fair_value_buffer_cents=int(settings.trading["fair_value_buffer_cents"]),
        time_exit_minutes_before_expiry=int(settings.trading["time_exit_minutes_before_expiry"]),
        min_hold_edge=float(settings.trading["min_hold_edge"]),
        min_ev_to_hold_cents=int(settings.trading["min_ev_to_hold_cents"]),
        exit_liquidity_floor=float(settings.trading["exit_liquidity_floor"]),
    )
    backtest_config = BacktestConfig(
        default_contracts=int(settings.trading["default_contracts"]),
        entry_slippage_cents=int(settings.backtest["entry_slippage_cents"]),
        exit_slippage_cents=int(settings.backtest["exit_slippage_cents"]),
        fee_rate_bps=float(settings.backtest["fee_rate_bps"]),
        starting_bankroll=float(settings.backtest["starting_bankroll"]),
        allow_reentry=bool(settings.backtest["allow_reentry"]),
    )
    fusion_config = FusionConfig(
        mode=str(settings.strategy["mode"]),
        primary_model=str(settings.strategy["primary_model"]),
        confirm_model=str(settings.strategy["confirm_model"]),
        primary_weight=float(settings.strategy["primary_weight"]),
        confirm_weight=float(settings.strategy["confirm_weight"]),
        require_side_agreement=bool(settings.strategy["require_side_agreement"]),
        min_combined_edge=float(settings.strategy["min_combined_edge"]),
        allow_primary_unconfirmed=bool(settings.strategy["allow_primary_unconfirmed"]),
    )
    risk_config = RiskConfig(
        max_trade_notional=float(settings.risk["max_trade_notional"]),
        max_session_notional=float(settings.risk["max_session_notional"]),
        max_open_positions=int(settings.risk["max_open_positions"]),
        max_positions_per_expiry=int(settings.risk["max_positions_per_expiry"]),
        max_daily_loss=float(settings.risk["max_daily_loss"]),
        max_drawdown=float(settings.risk["max_drawdown"]),
    )
    engine = BacktestEngine(
        model=model,
        models=models,
        signal_config=signal_config,
        exit_config=exit_config,
        backtest_config=backtest_config,
        fusion_config=fusion_config,
        risk_config=risk_config,
    )
    if bool(settings.raw.get("calibration", {}).get("enabled", False)):
        attach_replay_calibrators(settings, engine)
    return settings, engine


def attach_replay_calibrators(settings, engine: BacktestEngine) -> dict[str, int]:
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    snapshots = store.load_snapshots(
        series_ticker=str(settings.collector["series_ticker"]),
        replay_ready_only=True,
        reference_time=datetime.now(timezone.utc),
    )
    if not snapshots:
        engine.calibrators = {}
        return {}
    dataset = build_replay_dataset(
        snapshots,
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
    )
    calibrators = build_engine_calibrators(
        engine,
        snapshots=dataset.snapshots,
        feature_frame=dataset.feature_frame,
        bucket_width=float(settings.raw.get("calibration", {}).get("bucket_width", 0.05)),
        min_samples=int(settings.raw.get("calibration", {}).get("min_samples", 50)),
        min_bucket_count=int(settings.raw.get("calibration", {}).get("min_bucket_count", 3)),
    )
    engine.calibrators = calibrators
    return {name: calibrator.sample_count for name, calibrator in calibrators.items()}


def build_collector(settings):
    coinbase = CoinbaseClient(
        product_id=str(settings.data["coinbase_product_id"]),
        verify_ssl=bool(settings.data["coinbase_verify_ssl"]),
    )
    kalshi = KalshiClient()
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    config = HybridCollectorConfig(
        series_ticker=str(settings.collector["series_ticker"]),
        status=str(settings.collector["status"]),
        market_limit=int(settings.collector["market_limit"]),
        min_minutes_to_expiry=int(settings.collector["min_minutes_to_expiry"]),
        max_minutes_to_expiry=int(settings.collector["max_minutes_to_expiry"]) if settings.collector["max_minutes_to_expiry"] is not None else None,
        reconcile_interval_seconds=int(settings.collector["reconcile_interval_seconds"]),
        spot_refresh_interval_seconds=int(settings.collector["spot_refresh_interval_seconds"]),
        websocket_channels=list(settings.collector["websocket_channels"]),
    )
    websocket_headers_factory = None
    api_key_id = str(settings.kalshi["api_key_id"]).strip()
    private_key_path = str(settings.kalshi["private_key_path"]).strip()
    websocket_path = str(settings.kalshi["websocket_path"]).strip()
    if api_key_id and private_key_path:
        signer = KalshiAuthSigner(
            KalshiAuthConfig(
                api_key_id=api_key_id,
                private_key_path=private_key_path,
            )
        )
        websocket_headers_factory = lambda: signer.websocket_headers(websocket_path)
    return HybridCollector(
        kalshi_client=kalshi,
        coinbase_client=coinbase,
        snapshot_store=store,
        config=config,
        websocket_headers_factory=websocket_headers_factory,
    )


def build_live_paper_trader(settings):
    collector = build_collector(settings)
    coinbase = CoinbaseClient(
        product_id=str(settings.data["coinbase_product_id"]),
        verify_ssl=bool(settings.data["coinbase_verify_ssl"]),
    )
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    _, engine = build_engine()
    settlement_enricher = SettlementEnricher(
        kalshi_client=KalshiClient(),
        snapshot_store=store,
        config=SettlementEnrichmentConfig(
            series_ticker=str(settings.collector["series_ticker"]),
            max_markets=int(settings.paper.get("settlement_max_markets_per_cycle", 100)),
            recent_hours=int(settings.paper.get("settlement_recent_hours", settings.collector["settlement_recent_hours"])),
        ),
    )
    config = LivePaperConfig(
        feature_history_hours=int(settings.paper["feature_history_hours"]),
        poll_interval_seconds=int(settings.paper["poll_interval_seconds"]),
        ledger_path=str(settings.paper["ledger_path"]),
        feature_timeframe=str(settings.data["base_timeframe"]),
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
        settlement_check_interval_seconds=int(settings.paper.get("settlement_check_interval_seconds", 300)),
        settlement_recent_hours=int(settings.paper.get("settlement_recent_hours", settings.collector["settlement_recent_hours"])),
        settlement_max_markets_per_cycle=int(settings.paper.get("settlement_max_markets_per_cycle", 100)),
    )
    return LivePaperTrader(
        collector=collector,
        engine=engine,
        coinbase_client=coinbase,
        snapshot_store=store,
        config=config,
        settlement_enricher=settlement_enricher,
        calibrator_refresher=lambda: attach_replay_calibrators(settings, engine),
    )


def build_backfill_service(settings, *, series_ticker: str, start_ts: int, end_ts: int, period_interval: int | None):
    coinbase = CoinbaseClient(
        product_id=str(settings.data["coinbase_product_id"]),
        verify_ssl=bool(settings.data["coinbase_verify_ssl"]),
    )
    kalshi = KalshiClient()
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    config = BackfillConfig(
        series_ticker=series_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=period_interval or int(settings.collector["backfill_period_interval"]),
        batch_size=int(settings.collector["backfill_batch_size"]),
        historical_pages=int(settings.collector["backfill_historical_pages"]),
    )
    return CandlestickBackfillService(
        kalshi_client=kalshi,
        coinbase_client=coinbase,
        snapshot_store=store,
        config=config,
    )


def parse_optional_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def parse_api_datetime(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def event_reference_time(event: dict[str, Any]) -> datetime | None:
    for key in ("settlement_ts", "settlement_time", "close_time", "expiration_time", "strike_date", "end_date"):
        parsed = parse_api_datetime(event.get(key))
        if parsed is not None:
            return parsed
    return None


def historical_market_cutoff(payload: dict[str, Any]) -> datetime | None:
    candidate = payload.get("market_settled_ts")
    if candidate is None and isinstance(payload.get("cutoffs"), dict):
        candidate = payload["cutoffs"].get("market_settled_ts")
    if candidate is None:
        return None
    if isinstance(candidate, (int, float)):
        return datetime.fromtimestamp(float(candidate), tz=timezone.utc)
    parsed = parse_api_datetime(candidate)
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(float(candidate), tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def list_historical_markets_for_window(
    *,
    kalshi: KalshiClient,
    series_ticker: str,
    start_ts: int,
    end_ts: int,
    max_events: int,
    max_markets: int,
) -> dict[str, Any]:
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    cutoff_payload = kalshi.get_historical_cutoff()
    market_cutoff = historical_market_cutoff(cutoff_payload)
    payload = kalshi.list_events(series_ticker=series_ticker, status="settled", limit=200)
    matching_events: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        ref_time = event_reference_time(event)
        if ref_time is None:
            continue
        if start_dt <= ref_time <= end_dt:
            matching_events.append(
                {
                    "event_ticker": str(event.get("event_ticker") or event.get("ticker") or ""),
                    "reference_time": ref_time.isoformat(),
                    "title": event.get("title"),
                    "subtitle": event.get("subtitle") or event.get("sub_title"),
                }
            )
        if len(matching_events) >= max_events:
            break

    markets: list[dict[str, Any]] = []
    for event in matching_events:
        event_ticker = event["event_ticker"]
        if not event_ticker:
            continue
        event_time = parse_api_datetime(event["reference_time"])
        if market_cutoff is not None and event_time is not None and event_time >= market_cutoff:
            fetched_markets = kalshi.list_markets(event_ticker=event_ticker, status="settled", limit=1000)
            route = "live"
        else:
            fetched_markets = kalshi.list_historical_markets(limit=1000, event_ticker=event_ticker).get("markets", [])
            route = "historical"
        for market in fetched_markets:
            ticker = str(market.get("ticker") or "")
            if not ticker.startswith(f"{series_ticker}-"):
                continue
            markets.append(
                {
                    "event_ticker": event_ticker,
                    "market_ticker": ticker,
                    "expiry": market.get("expiry") or market.get("expiration_time") or market.get("close_time"),
                    "status": market.get("status"),
                    "yes_sub_title": market.get("yes_sub_title"),
                    "no_sub_title": market.get("no_sub_title"),
                    "route": route,
                }
            )
            if len(markets) >= max_markets:
                break
        if len(markets) >= max_markets:
            break

    return {
        "series_ticker": series_ticker,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "market_settled_cutoff": market_cutoff.isoformat() if market_cutoff is not None else None,
        "matched_event_count": len(matching_events),
        "matched_market_count": len(markets),
        "events": matching_events,
        "markets": markets,
    }


def inspect_market_with_cutoff(kalshi: KalshiClient, ticker: str) -> dict[str, Any]:
    cutoff_payload = kalshi.get_historical_cutoff()
    market_cutoff = historical_market_cutoff(cutoff_payload)
    live_error = None
    historical_error = None
    live_payload = None
    historical_payload = None
    try:
        live_payload = kalshi.get_market(ticker)
    except Exception as exc:
        live_error = str(exc)
    try:
        historical_payload = kalshi.get_historical_market(ticker)
    except Exception as exc:
        historical_error = str(exc)

    recommended_source = "unknown"
    market_payload = None
    if isinstance(live_payload, dict) and live_payload.get("market"):
        market_payload = live_payload["market"]
        expiry = parse_api_datetime(market_payload.get("settlement_time") or market_payload.get("close_time") or market_payload.get("expiry"))
        if market_cutoff is not None and expiry is not None and expiry < market_cutoff:
            recommended_source = "historical"
        else:
            recommended_source = "live"
    elif isinstance(historical_payload, dict) and historical_payload.get("market"):
        market_payload = historical_payload["market"]
        recommended_source = "historical"

    return {
        "ticker": ticker,
        "market_settled_cutoff": market_cutoff.isoformat() if market_cutoff is not None else None,
        "recommended_source": recommended_source,
        "live_error": live_error,
        "historical_error": historical_error,
        "live_market": live_payload.get("market") if isinstance(live_payload, dict) else None,
        "historical_market": historical_payload.get("market") if isinstance(historical_payload, dict) else None,
    }


def fetch_candles_with_cutoff(
    *,
    kalshi: KalshiClient,
    ticker: str,
    series_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
    include_latest_before_start: bool,
) -> dict[str, Any]:
    cutoff_payload = kalshi.get_historical_cutoff()
    market_cutoff = historical_market_cutoff(cutoff_payload)
    inspect = inspect_market_with_cutoff(kalshi, ticker)
    recommended_source = inspect["recommended_source"]
    error = None
    payload = None
    try:
        if recommended_source == "historical":
            payload = kalshi.get_historical_market_candlesticks(
                ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
                include_latest_before_start=include_latest_before_start,
            )
        else:
            payload = kalshi.get_market_candlesticks(
                series_ticker=series_ticker,
                market_ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=period_interval,
                include_latest_before_start=include_latest_before_start,
            )
    except Exception as exc:
        error = str(exc)

    return {
        "ticker": ticker,
        "series_ticker": series_ticker,
        "market_settled_cutoff": market_cutoff.isoformat() if market_cutoff is not None else None,
        "recommended_source": recommended_source,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
        "include_latest_before_start": include_latest_before_start,
        "error": error,
        "candlestick_count": len(payload.get("candlesticks", [])) if isinstance(payload, dict) else 0,
        "payload": payload,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings, _ = build_engine()
    if args.command == "show-config":
        print(json.dumps(settings.raw, indent=2))
        return
    if args.command == "spot":
        client = CoinbaseClient(product_id=str(settings.data["coinbase_product_id"]))
        print(client.get_spot_price())
        return
    if args.command == "demo-backtest":
        client = CoinbaseClient(product_id=str(settings.data["coinbase_product_id"]))
        start, end = default_history_window(args.hours)
        candles = client.fetch_candles(start, end, timeframe=str(settings.data["base_timeframe"]))
        features = build_feature_frame(
            candles,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
        )
        kalshi = KalshiClient()
        spot = float(candles["close"].iloc[-1]) if not candles.empty else client.get_spot_price()
        snapshots = kalshi.normalized_snapshots(spot_price=spot, observed_at=datetime.now(timezone.utc))
        print(f"Fetched {len(features)} feature rows and {len(snapshots)} Kalshi markets.")
        return
    if args.command == "collect-once":
        collector = build_collector(settings)
        if args.series:
            collector.config.series_ticker = args.series
        if args.max_minutes_to_expiry is not None:
            collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        snapshots = asyncio.run(collector.bootstrap())
        print(
            json.dumps(
                {
                    "snapshots_written": len(snapshots),
                    "sqlite_path": str(settings.collector["sqlite_path"]),
                    "series_ticker": collector.config.series_ticker,
                    "min_minutes_to_expiry": collector.config.min_minutes_to_expiry,
                    "max_minutes_to_expiry": collector.config.max_minutes_to_expiry,
                },
                indent=2,
            )
        )
        return
    if args.command == "collect-forever":
        collector = build_collector(settings)
        if args.series:
            collector.config.series_ticker = args.series
        if args.max_minutes_to_expiry is not None:
            collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        logging.getLogger(__name__).info(
            "Starting collector for series %s into %s",
            collector.config.series_ticker,
            settings.collector["sqlite_path"],
        )
        try:
            asyncio.run(collector.collect_forever())
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Collector stopped by user.")
        return
    if args.command == "paper-trade-once":
        trader = build_live_paper_trader(settings)
        if args.series:
            trader.collector.config.series_ticker = args.series
        if args.max_minutes_to_expiry is not None:
            trader.collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            trader.collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        result = asyncio.run(trader.run_once())
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == "paper-trade-forever":
        trader = build_live_paper_trader(settings)
        if args.series:
            trader.collector.config.series_ticker = args.series
        if args.max_minutes_to_expiry is not None:
            trader.collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            trader.collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        logging.getLogger(__name__).info(
            "Starting live paper trader for series %s with ledger %s",
            trader.collector.config.series_ticker,
            settings.paper["ledger_path"],
        )
        try:
            asyncio.run(trader.run_forever())
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Paper trader stopped by user.")
        return
    if args.command == "backfill-candles":
        series_ticker = args.series or str(settings.collector["series_ticker"])
        service = build_backfill_service(
            settings,
            series_ticker=series_ticker,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            period_interval=args.period_interval,
        )
        if args.max_markets is not None:
            service.config.max_markets = args.max_markets
        if args.historical:
            service.config.use_historical_markets = True
        snapshots = service.backfill()
        print(
            json.dumps(
                {
                    "snapshots_written": len(snapshots),
                    "sqlite_path": str(settings.collector["sqlite_path"]),
                    "series_ticker": series_ticker,
                    "start_ts": args.start_ts,
                    "end_ts": args.end_ts,
                    "historical": bool(args.historical),
                },
                indent=2,
            )
        )
        return
    if args.command == "historical-cutoff":
        kalshi = KalshiClient()
        payload = kalshi.get_historical_cutoff()
        print(json.dumps(payload, indent=2, default=str))
        return
    if args.command == "historical-list-markets":
        kalshi = KalshiClient()
        series_ticker = args.series or str(settings.collector["series_ticker"])
        result = list_historical_markets_for_window(
            kalshi=kalshi,
            series_ticker=series_ticker,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            max_events=args.max_events,
            max_markets=args.max_markets,
        )
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == "historical-inspect-market":
        kalshi = KalshiClient()
        print(json.dumps(inspect_market_with_cutoff(kalshi, args.ticker), indent=2, default=str))
        return
    if args.command == "historical-fetch-candles":
        kalshi = KalshiClient()
        series_ticker = args.series or str(settings.collector["series_ticker"])
        result = fetch_candles_with_cutoff(
            kalshi=kalshi,
            ticker=args.ticker,
            series_ticker=series_ticker,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            period_interval=args.period_interval,
            include_latest_before_start=bool(args.include_latest_before_start),
        )
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == "dataset-summary":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        series_ticker = args.series or str(settings.collector["series_ticker"])
        print(json.dumps(store.dataset_summary(series_ticker=series_ticker, reference_time=datetime.now(timezone.utc)), indent=2))
        return
    if args.command == "fix-series-ticker":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        updated = store.remap_series_ticker(args.old, args.new)
        print(json.dumps({"rows_updated": updated, "old": args.old, "new": args.new}, indent=2))
        return
    if args.command == "enrich-settlements":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        enricher = SettlementEnricher(
            kalshi_client=KalshiClient(),
            snapshot_store=store,
            config=SettlementEnrichmentConfig(
                series_ticker=args.series or str(settings.collector["series_ticker"]),
                max_markets=args.max_markets,
                recent_hours=args.recent_hours if args.recent_hours is not None else int(settings.collector["settlement_recent_hours"]),
            ),
        )
        result = enricher.enrich()
        print(json.dumps(result, indent=2))
        return
    if args.command == "replay-diagnostics":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        series_ticker = args.series or str(settings.collector["series_ticker"])
        snapshots = store.load_snapshots(
            series_ticker=series_ticker,
            market_ticker=args.market,
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
            limit=args.limit,
            replay_ready_only=True,
            reference_time=datetime.now(timezone.utc),
        )
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
        )
        _, engine = build_engine()
        diagnostics = engine.diagnose_replay(dataset.snapshots, dataset.feature_frame)
        print(
            json.dumps(
                {
                    "summary": diagnostics.summary,
                    "reason_counts": diagnostics.reason_counts,
                    "per_market_preview": diagnostics.per_market.head(10).to_dict(orient="records") if not diagnostics.per_market.empty else [],
                },
                indent=2,
                default=str,
            )
        )
        return
    if args.command == "replay-backtest":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        series_ticker = args.series or str(settings.collector["series_ticker"])
        snapshots = store.load_snapshots(
            series_ticker=series_ticker,
            market_ticker=args.market,
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
            limit=args.limit,
            replay_ready_only=True,
            reference_time=datetime.now(timezone.utc),
        )
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
        )
        _, engine = build_engine()
        comparison = engine.compare_strategies(dataset.snapshots, dataset.feature_frame)
        output = {
            "snapshot_count": len(dataset.snapshots),
            "feature_rows": len(dataset.feature_frame),
            "summary": comparison["summary"],
            "hold_to_settlement": comparison["hold_to_settlement"].summary,
            "early_exit": comparison["early_exit"].summary,
        }
        print(json.dumps(output, indent=2, default=str))
        return
    if args.command == "replay-compare-models":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        series_ticker = args.series or str(settings.collector["series_ticker"])
        snapshots = store.load_snapshots(
            series_ticker=series_ticker,
            market_ticker=args.market,
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
            limit=args.limit,
            replay_ready_only=True,
            reference_time=datetime.now(timezone.utc),
        )
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
        )
        focused_snapshots = filter_snapshots_for_focus(
            dataset.snapshots,
            near_money_bps=args.near_money_bps,
            max_minutes_to_expiry=args.max_minutes_to_expiry,
            min_price_cents=args.min_price_cents,
            max_price_cents=args.max_price_cents,
        )
        dataset = build_replay_dataset(
            focused_snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
        )
        _, engine = build_engine()
        engines = {
            "gbm_threshold": clone_engine_with_mode(engine, fusion_mode="single", primary_model="gbm_threshold"),
            "latency_repricing": clone_engine_with_mode(engine, fusion_mode="single", primary_model="latency_repricing"),
            "hybrid": clone_engine_with_mode(engine, fusion_mode="hybrid", primary_model=str(settings.strategy["primary_model"])),
        }
        report = build_model_comparison_report(engines, snapshots=dataset.snapshots, feature_frame=dataset.feature_frame)
        report["snapshot_count"] = len(dataset.snapshots)
        report["feature_rows"] = len(dataset.feature_frame)
        report["filters"] = {
            "near_money_bps": args.near_money_bps,
            "max_minutes_to_expiry": args.max_minutes_to_expiry,
            "min_price_cents": args.min_price_cents,
            "max_price_cents": args.max_price_cents,
        }
        print(json.dumps(report, indent=2, default=str))
        return
    if args.command == "replay-grid-search":
        store = SnapshotStore(str(settings.collector["sqlite_path"]))
        series_ticker = args.series or str(settings.collector["series_ticker"])
        snapshots = store.load_snapshots(
            series_ticker=series_ticker,
            market_ticker=args.market,
            observed_from=parse_optional_timestamp(args.from_ts),
            observed_to=parse_optional_timestamp(args.to_ts),
            limit=args.limit,
            replay_ready_only=True,
            reference_time=datetime.now(timezone.utc),
        )
        _, engine = build_engine()
        engines = {
            "gbm_threshold": clone_engine_with_mode(engine, fusion_mode="single", primary_model="gbm_threshold"),
            "latency_repricing": clone_engine_with_mode(engine, fusion_mode="single", primary_model="latency_repricing"),
            "hybrid": clone_engine_with_mode(engine, fusion_mode="hybrid", primary_model=str(settings.strategy["primary_model"])),
        }
        report = build_grid_search_report(
            engines,
            snapshots=snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            near_money_bps_values=parse_numeric_list(args.near_money_bps_values, cast=float),
            max_minutes_to_expiry_values=parse_numeric_list(args.max_minutes_to_expiry_values, cast=float),
            min_price_cents_values=parse_numeric_list(args.min_price_cents_values, cast=int),
            max_price_cents_values=parse_numeric_list(args.max_price_cents_values, cast=int),
            top_n=int(args.top),
        )
        report["input_snapshot_count"] = len(snapshots)
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
