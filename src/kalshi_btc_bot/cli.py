from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from kalshi_btc_bot.backtest.engine import BacktestConfig, BacktestEngine
from kalshi_btc_bot.backtest.engine import MakerSimulationConfig
from kalshi_btc_bot.backtest.bankroll import BankrollSizingConfig, simulate_bankroll_constrained_compounding
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
from kalshi_btc_bot.models.repricing_target import (
    RepricingTargetModel,
    load_repricing_profile,
    save_repricing_profile,
    train_repricing_profile,
)
from kalshi_btc_bot.reports.comparison import (
    build_failure_analysis,
    build_grid_search_report,
    build_model_comparison_report,
    clone_engine_with_mode,
    filter_snapshots_for_focus,
)
from kalshi_btc_bot.settings import load_settings
from kalshi_btc_bot.signals.calibration import build_engine_calibrators, load_engine_calibrators, save_engine_calibrators
from kalshi_btc_bot.signals.fusion import FusionConfig
from kalshi_btc_bot.signals.engine import SignalConfig
from kalshi_btc_bot.storage.snapshots import SnapshotStore
from kalshi_btc_bot.trading.exits import ExitConfig
from kalshi_btc_bot.trading.live_paper import LivePaperConfig, LivePaperTrader
from kalshi_btc_bot.trading.real_execution import RealExecutionConfig, RealOrderExecutor, RealOrderRequest
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
    replicate_once = subparsers.add_parser("replicate-snapshot-db-once", help="Copy the writer snapshot SQLite DB into the read-replica DB")
    replicate_once.add_argument("--source", type=str, default=None)
    replicate_once.add_argument("--target", type=str, default=None)
    replicate_forever = subparsers.add_parser("replicate-snapshot-db-forever", help="Continuously refresh the read-replica SQLite DB from the writer DB")
    replicate_forever.add_argument("--source", type=str, default=None)
    replicate_forever.add_argument("--target", type=str, default=None)
    replicate_forever.add_argument("--interval-seconds", type=int, default=None)
    paper_once = subparsers.add_parser("paper-trade-once", help="Run one live paper-trading cycle from current market data")
    paper_once.add_argument("--series", type=str, default=None)
    paper_once.add_argument("--max-minutes-to-expiry", type=int, default=None)
    paper_once.add_argument("--min-minutes-to-expiry", type=int, default=None)
    paper_forever = subparsers.add_parser("paper-trade-forever", help="Continuously paper trade live Kalshi BTC markets")
    paper_forever.add_argument("--series", type=str, default=None)
    paper_forever.add_argument("--max-minutes-to-expiry", type=int, default=None)
    paper_forever.add_argument("--min-minutes-to-expiry", type=int, default=None)
    real_preview = subparsers.add_parser("real-order-preview", help="Preview a real-money Kalshi order without sending it")
    real_preview.add_argument("--market", type=str, required=True)
    real_preview.add_argument("--side", type=str, required=True, choices=("yes", "no"))
    real_preview.add_argument("--action", type=str, default="buy", choices=("buy", "sell"))
    real_preview.add_argument("--count", type=int, default=1)
    real_preview.add_argument("--yes-price-cents", type=int, default=None)
    real_preview.add_argument("--no-price-cents", type=int, default=None)
    real_preview.add_argument("--client-order-id", type=str, default=None)
    real_preview.add_argument("--series", type=str, default=None)
    real_submit = subparsers.add_parser("real-order-submit", help="Submit a Kalshi order; dry-run by default unless --live is passed")
    real_submit.add_argument("--market", type=str, required=True)
    real_submit.add_argument("--side", type=str, required=True, choices=("yes", "no"))
    real_submit.add_argument("--action", type=str, default="buy", choices=("buy", "sell"))
    real_submit.add_argument("--count", type=int, default=1)
    real_submit.add_argument("--yes-price-cents", type=int, default=None)
    real_submit.add_argument("--no-price-cents", type=int, default=None)
    real_submit.add_argument("--client-order-id", type=str, default=None)
    real_submit.add_argument("--series", type=str, default=None)
    real_submit.add_argument("--live", action="store_true")
    real_trade_once = subparsers.add_parser("real-trade-once", help="Use the live strategy to select one real-money candidate; dry-run by default")
    real_trade_once.add_argument("--series", type=str, default=None)
    real_trade_once.add_argument("--max-minutes-to-expiry", type=int, default=None)
    real_trade_once.add_argument("--min-minutes-to-expiry", type=int, default=None)
    real_trade_once.add_argument("--live", action="store_true")
    real_trade_forever = subparsers.add_parser("real-trade-forever", help="Continuously mirror the live strategy into the real-order path; dry-run by default")
    real_trade_forever.add_argument("--series", type=str, default=None)
    real_trade_forever.add_argument("--max-minutes-to-expiry", type=int, default=None)
    real_trade_forever.add_argument("--min-minutes-to-expiry", type=int, default=None)
    real_trade_forever.add_argument("--live", action="store_true")
    real_status = subparsers.add_parser("real-trading-status", help="Show kill switch, resting orders, positions, and recent real-order ledger info")
    real_status.add_argument("--series", type=str, default=None)
    real_kill = subparsers.add_parser("real-kill-switch", help="Enable or disable the real-trading kill switch")
    real_kill.add_argument("--series", type=str, default=None)
    real_kill.add_argument("--enable", action="store_true")
    real_kill.add_argument("--disable", action="store_true")
    real_kill.add_argument("--reason", type=str, default="")
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
    failure = subparsers.add_parser("replay-failure-analysis", help="Explain where the current replay setup wins and loses")
    failure.add_argument("--series", type=str, default=None)
    failure.add_argument("--market", type=str, default=None)
    failure.add_argument("--from-ts", type=str, default=None)
    failure.add_argument("--to-ts", type=str, default=None)
    failure.add_argument("--limit", type=int, default=None)
    failure.add_argument("--near-money-bps", type=float, default=None)
    failure.add_argument("--max-minutes-to-expiry", type=float, default=None)
    failure.add_argument("--min-price-cents", type=int, default=None)
    failure.add_argument("--max-price-cents", type=int, default=None)
    failure.add_argument("--model", type=str, default="gbm_threshold")
    failure.add_argument("--mode", type=str, default="early_exit", choices=("early_exit", "hold_to_settlement"))
    failure.add_argument("--top", type=int, default=3)
    maker_proxy = subparsers.add_parser("replay-maker-proxy", help="Test a simple maker-entry proxy against the current taker-entry backtest")
    maker_proxy.add_argument("--series", type=str, default=None)
    maker_proxy.add_argument("--market", type=str, default=None)
    maker_proxy.add_argument("--from-ts", type=str, default=None)
    maker_proxy.add_argument("--to-ts", type=str, default=None)
    maker_proxy.add_argument("--limit", type=int, default=None)
    maker_proxy.add_argument("--near-money-bps", type=float, default=None)
    maker_proxy.add_argument("--max-minutes-to-expiry", type=float, default=None)
    maker_proxy.add_argument("--min-price-cents", type=int, default=None)
    maker_proxy.add_argument("--max-price-cents", type=int, default=None)
    maker_proxy.add_argument("--model", type=str, default="gbm_threshold")
    maker_proxy.add_argument("--mode", type=str, default="early_exit", choices=("early_exit", "hold_to_settlement"))
    maker_sim = subparsers.add_parser("replay-maker-sim", help="Compare taker, optimistic maker, and conservative maker-sim entries")
    maker_sim.add_argument("--series", type=str, default=None)
    maker_sim.add_argument("--market", type=str, default=None)
    maker_sim.add_argument("--from-ts", type=str, default=None)
    maker_sim.add_argument("--to-ts", type=str, default=None)
    maker_sim.add_argument("--limit", type=int, default=None)
    maker_sim.add_argument("--near-money-bps", type=float, default=None)
    maker_sim.add_argument("--max-minutes-to-expiry", type=float, default=None)
    maker_sim.add_argument("--min-price-cents", type=int, default=None)
    maker_sim.add_argument("--max-price-cents", type=int, default=None)
    maker_sim.add_argument("--model", type=str, default="gbm_threshold")
    maker_sim.add_argument("--mode", type=str, default="early_exit", choices=("early_exit", "hold_to_settlement"))
    maker_sim.add_argument("--maker-max-wait-seconds", type=int, default=90)
    maker_sim.add_argument("--maker-min-fill-probability", type=float, default=0.55)
    maker_sim.add_argument("--maker-stale-quote-age-seconds", type=int, default=20)
    maker_sim.add_argument("--maker-max-posted-spread-cents", type=int, default=6)
    maker_sim.add_argument("--maker-min-liquidity-score", type=float, default=80.0)
    maker_sim.add_argument("--maker-max-concurrent-positions-per-side", type=int, default=1)
    rebuild_repricing = subparsers.add_parser("rebuild-repricing-profile", help="Train and cache the short-horizon repricing model from replay-ready history")
    rebuild_repricing.add_argument("--series", type=str, default=None)
    rebuild_repricing.add_argument("--from-ts", type=str, default=None)
    rebuild_repricing.add_argument("--to-ts", type=str, default=None)
    rebuild_repricing.add_argument("--limit", type=int, default=None)
    grid = subparsers.add_parser("replay-grid-search", help="Search filter ranges and rank the best replay setups")
    grid.add_argument("--series", type=str, default=None)
    grid.add_argument("--market", type=str, default=None)
    grid.add_argument("--from-ts", type=str, default=None)
    grid.add_argument("--to-ts", type=str, default=None)
    grid.add_argument("--limit", type=int, default=None)
    grid.add_argument("--near-money-bps-values", type=str, default="100,150,200")
    grid.add_argument("--max-minutes-to-expiry-values", type=str, default="15,30,60")
    grid.add_argument("--min-price-cents-values", type=str, default="20,25")
    grid.add_argument("--max-price-cents-values", type=str, default="60,75")
    grid.add_argument("--top", type=int, default=20)
    bankroll = subparsers.add_parser("replay-bankroll-sim", help="Run a bankroll-constrained compounding simulation on replay trades")
    bankroll.add_argument("--sqlite-path", type=str, required=True)
    bankroll.add_argument("--series", type=str, required=True)
    bankroll.add_argument("--mode", type=str, default="early_exit", choices=("early_exit", "hold_to_settlement"))
    bankroll.add_argument("--starting-bankroll", type=float, default=100.0)
    bankroll.add_argument("--bankroll-fraction-per-trade", type=float, default=1.0)
    bankroll.add_argument("--min-cash-buffer", type=float, default=0.0)
    bankroll.add_argument("--max-contracts-per-trade", type=int, default=1)
    bankroll.add_argument("--allow-fractional-contracts", action="store_true")
    walkforward = subparsers.add_parser("replay-walkforward-calibration", help="Train calibrators on the first part of history and test on the later part")
    walkforward.add_argument("--sqlite-path", type=str, required=True)
    walkforward.add_argument("--series", type=str, required=True)
    walkforward.add_argument("--mode", type=str, default="early_exit", choices=("early_exit", "hold_to_settlement"))
    walkforward.add_argument("--train-fraction", type=float, default=0.6)
    return parser


def parse_numeric_list(raw: str, *, cast):
    values = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def replay_cache_dir(settings) -> str | None:
    replay_cache = settings.raw.get("replay_cache", {})
    if not bool(replay_cache.get("enabled", True)):
        return None
    return str(replay_cache.get("directory", "data/replay_cache"))


def calibration_cache_path(settings) -> str | None:
    calibration_cache = settings.raw.get("calibration_cache", {})
    if not bool(calibration_cache.get("enabled", True)):
        return None
    return str(calibration_cache.get("path", "data/calibration_cache.json"))


def configured_collector_series(settings) -> list[str]:
    raw_series = settings.collector.get("series_tickers")
    series = list(raw_series) if isinstance(raw_series, list) else []
    default_series = str(settings.collector["series_ticker"])
    if default_series and default_series not in series:
        series.insert(0, default_series)
    return [item for item in series if item]


def snapshot_write_path(settings) -> str:
    return str(settings.collector["sqlite_path"])


def snapshot_read_path(settings) -> str:
    return str(settings.collector.get("read_sqlite_path", settings.collector["sqlite_path"]))


def snapshot_replica_interval_seconds(settings) -> int:
    return int(settings.collector.get("replica_sync_interval_seconds", 5))


def replicate_snapshot_db_once(*, source_path: str, target_path: str) -> dict[str, object]:
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True, timeout=30)
    target = sqlite3.connect(target_path, timeout=30)
    try:
        source.backup(target)
        target.commit()
    finally:
        target.close()
        source.close()
    return {
        "source": source_path,
        "target": target_path,
        "replicated_at": datetime.now(timezone.utc).isoformat(),
    }


def replicate_snapshot_db_forever(*, source_path: str, target_path: str, interval_seconds: int) -> None:
    log = logging.getLogger(__name__)
    while True:
        try:
            result = replicate_snapshot_db_once(source_path=source_path, target_path=target_path)
            log.info("Replicated snapshot DB from %s to %s at %s", result["source"], result["target"], result["replicated_at"])
        except Exception as exc:
            log.warning("Snapshot DB replication failed: %s", exc)
        time.sleep(interval_seconds)


def split_snapshots_by_fraction(snapshots, train_fraction: float):
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.observed_at)
    if not ordered:
        return [], [], None
    fraction = min(max(float(train_fraction), 0.05), 0.95)
    cutoff_index = max(1, min(len(ordered) - 1, int(len(ordered) * fraction)))
    cutoff = ordered[cutoff_index].observed_at
    train_snapshots = [snapshot for snapshot in ordered if snapshot.observed_at < cutoff]
    test_snapshots = [snapshot for snapshot in ordered if snapshot.observed_at >= cutoff]
    if not train_snapshots:
        train_snapshots = ordered[:1]
        test_snapshots = ordered[1:]
        cutoff = test_snapshots[0].observed_at if test_snapshots else ordered[-1].observed_at
    if not test_snapshots:
        train_snapshots = ordered[:-1]
        test_snapshots = ordered[-1:]
        cutoff = test_snapshots[0].observed_at
    return train_snapshots, test_snapshots, cutoff


def build_engine(*, attach_calibration: bool = True, attach_repricing_profile: bool = True):
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
        series_tier_profiles=dict(settings.raw.get("signal_tiers", {})),
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
        ranking_support_model=str(settings.strategy.get("ranking_support_model") or "") or None,
        ranking_support_weight=float(settings.strategy.get("ranking_support_weight", 0.0)),
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
    if attach_repricing_profile:
        attach_replay_repricing_model(settings, engine)
        requested_model = str(settings.model["name"])
        if requested_model in engine.models:
            engine.model = engine.models[requested_model]
        elif engine.fusion_config is not None and engine.fusion_config.primary_model in engine.models:
            engine.model = engine.models[engine.fusion_config.primary_model]
    if attach_calibration and bool(settings.raw.get("calibration", {}).get("enabled", False)):
        attach_replay_calibrators(settings, engine)
    return settings, engine


def build_authenticated_kalshi(settings) -> KalshiClient:
    auth = KalshiAuthConfig(
        api_key_id=str(settings.kalshi["api_key_id"]),
        private_key_path=str(settings.kalshi["private_key_path"]),
    )
    signer = KalshiAuthSigner(auth)
    return KalshiClient(auth_signer=signer)


def infer_series_from_market_ticker(market_ticker: str) -> str:
    ticker = str(market_ticker or "")
    return ticker.split("-", 1)[0] if "-" in ticker else ticker


def configured_real_series(settings, explicit_series: str | None = None, market_ticker: str | None = None) -> str:
    if explicit_series:
        return str(explicit_series)
    if market_ticker:
        inferred = infer_series_from_market_ticker(str(market_ticker))
        if inferred:
            return inferred
    return str(settings.paper.get("series_ticker", settings.collector["series_ticker"]))


def real_series_value(settings, series_ticker: str, key: str, default: Any) -> Any:
    per_series_key = f"series_{key}s"
    per_series = settings.real.get(per_series_key)
    if isinstance(per_series, dict) and series_ticker in per_series:
        return per_series[series_ticker]
    return settings.real.get(key, default)


def real_series_live_value(settings, series_ticker: str, key: str, default: Any, *, live: bool) -> Any:
    if live:
        live_key = f"series_live_{key}s"
        per_series_live = settings.real.get(live_key)
        if isinstance(per_series_live, dict) and series_ticker in per_series_live:
            return per_series_live[series_ticker]
    return real_series_value(settings, series_ticker, key, default)


def configured_paper_series(settings, explicit_series: str | None = None) -> str:
    if explicit_series:
        return str(explicit_series)
    return str(settings.paper.get("series_ticker", settings.collector["series_ticker"]))


def paper_series_value(settings, series_ticker: str, key: str, default: Any) -> Any:
    for per_series_key in (f"series_{key}", f"series_{key}s"):
        per_series = settings.paper.get(per_series_key)
        if isinstance(per_series, dict) and series_ticker in per_series:
            return per_series[series_ticker]
    return settings.paper.get(key, default)


def execute_real_trade_once(settings, args) -> dict[str, Any]:
    series_ticker = configured_real_series(settings, explicit_series=getattr(args, "series", None))
    executor = build_real_executor(settings, live=bool(args.live), series_ticker=series_ticker)
    housekeeping = executor.cancel_stale_orders()
    if bool(settings.real.get("skip_cycle_after_cancel", True)) and int(housekeeping.get("cancelled_count", 0)) > 0:
        return {
            "status": "skipped_after_cancel",
            "series_ticker": series_ticker,
            "reason": "Cancelled stale real order(s); skipping new entry this cycle.",
            "housekeeping": housekeeping,
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
    trader = build_live_paper_trader(settings)
    if args.series:
        trader.collector.config.series_ticker = args.series
        trader.collector.config.series_tickers = [args.series]
    if args.max_minutes_to_expiry is not None:
        trader.collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
    if args.min_minutes_to_expiry is not None:
        trader.collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry

    preview = asyncio.run(trader.preview_once())
    top_candidates = list(preview.get("top_candidates", []) or [])
    if not top_candidates:
        return {**preview, "series_ticker": series_ticker, "status": "no_candidate", "housekeeping": housekeeping}

    top = top_candidates[0]
    side = str(top["side"])
    request = RealOrderRequest(
        market_ticker=str(top["market_ticker"]),
        side=side,
        action="buy",
        count=1,
        yes_price_cents=int(top["entry_price_cents"]) if side == "yes" and top.get("entry_price_cents") is not None else None,
        no_price_cents=int(top["entry_price_cents"]) if side == "no" and top.get("entry_price_cents") is not None else None,
        client_order_id=f"cryptobot-{str(top['market_ticker']).lower()}-{int(datetime.now(timezone.utc).timestamp())}",
    )
    result = executor.submit_order(request)
    return {
        **preview,
        "series_ticker": series_ticker,
        "selected_order": request.to_api_payload(),
        "execution": result,
        "housekeeping": housekeeping,
    }


def build_real_executor(settings, *, live: bool, series_ticker: str | None = None, market_ticker: str | None = None) -> RealOrderExecutor:
    resolved_series = configured_real_series(settings, explicit_series=series_ticker, market_ticker=market_ticker)
    kalshi = build_authenticated_kalshi(settings)
    return RealOrderExecutor(
        kalshi_client=kalshi,
        config=RealExecutionConfig(
            ledger_path=str(real_series_live_value(settings, resolved_series, "ledger_path", "data/real_trading_ledger.json", live=live)),
            series_ticker=resolved_series,
            kill_switch_path=str(real_series_value(settings, resolved_series, "kill_switch_path", "data/real_trading_kill_switch.json")),
            dry_run=not live,
            max_daily_orders=int(settings.real.get("max_daily_orders", 5)),
            max_open_orders=int(settings.real.get("max_open_orders", 2)),
            max_open_positions=int(settings.real.get("max_open_positions", 1)),
            stale_order_timeout_seconds=int(settings.real.get("stale_order_timeout_seconds", 60)),
        ),
    )


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
    cache_metadata = {
        "series_ticker": str(settings.collector["series_ticker"]),
        "snapshot_count": len(snapshots),
        "last_observed_at": max(snapshot.observed_at for snapshot in snapshots).isoformat(),
        "bucket_width": float(settings.raw.get("calibration", {}).get("bucket_width", 0.05)),
        "min_samples": int(settings.raw.get("calibration", {}).get("min_samples", 50)),
        "min_bucket_count": int(settings.raw.get("calibration", {}).get("min_bucket_count", 3)),
    }
    cache_path = calibration_cache_path(settings)
    cached = load_engine_calibrators(cache_path, expected_metadata=cache_metadata) if cache_path else None
    if cached is not None:
        engine.calibrators = cached
        return {name: calibrator.sample_count for name, calibrator in cached.items()}
    dataset = build_replay_dataset(
        snapshots,
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
        cache_dir=replay_cache_dir(settings),
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
    if cache_path:
        save_engine_calibrators(cache_path, calibrators, metadata=cache_metadata)
    return {name: calibrator.sample_count for name, calibrator in calibrators.items()}


def attach_replay_repricing_model(settings, engine: BacktestEngine) -> int:
    store = SnapshotStore(str(settings.collector["sqlite_path"]))
    snapshots = store.load_snapshots(
        series_ticker=str(settings.collector["series_ticker"]),
        replay_ready_only=True,
        reference_time=datetime.now(timezone.utc),
    )
    if not snapshots:
        return 0
    repricing_settings = settings.raw.get("repricing_target", {})
    cache_metadata = {
        "series_ticker": str(settings.collector["series_ticker"]),
        "snapshot_count": len(snapshots),
        "last_observed_at": max(snapshot.observed_at for snapshot in snapshots).isoformat(),
        "horizon_minutes": int(repricing_settings.get("horizon_minutes", 5)),
    }
    cache_path = repricing_settings.get("cache_path")
    profile = load_repricing_profile(cache_path, expected_metadata=cache_metadata) if cache_path else None
    if profile is None:
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            cache_dir=replay_cache_dir(settings),
        )
        profile = train_repricing_profile(
            dataset.snapshots,
            feature_frame=dataset.feature_frame,
            horizon_minutes=int(repricing_settings.get("horizon_minutes", 5)),
            min_samples=int(repricing_settings.get("min_samples", 150)),
            min_regime_samples=int(repricing_settings.get("min_regime_samples", 75)),
            ridge_lambda=float(repricing_settings.get("ridge_lambda", 5.0)),
            max_abs_target_cents=float(repricing_settings.get("max_abs_target_cents", 25.0)),
            near_money_regime_bps=float(repricing_settings.get("near_money_regime_bps", 150.0)),
            high_vol_regime_threshold=float(repricing_settings.get("high_vol_regime_threshold", 0.8)),
            profit_cost_cents=float(repricing_settings.get("profit_cost_cents", 2.0)),
        )
        if profile is not None and cache_path:
            save_repricing_profile(cache_path, profile, metadata=cache_metadata)
    if profile is None:
        engine.models.pop("repricing_target", None)
        return 0
    engine.models["repricing_target"] = RepricingTargetModel(
        profile=profile,
        volatility_floor=float(settings.model["volatility_floor"]),
        max_probability_shift=float(repricing_settings.get("max_probability_shift", 0.2)),
    )
    return profile.sample_count


def build_comparison_engines(settings, engine: BacktestEngine) -> dict[str, BacktestEngine]:
    engines = {
        "gbm_threshold": clone_engine_with_mode(engine, fusion_mode="single", primary_model="gbm_threshold"),
        "latency_repricing": clone_engine_with_mode(engine, fusion_mode="single", primary_model="latency_repricing"),
        "hybrid": clone_engine_with_mode(engine, fusion_mode="hybrid", primary_model=str(settings.strategy["primary_model"])),
    }
    if "repricing_target" in engine.models:
        engines["repricing_target"] = clone_engine_with_mode(engine, fusion_mode="single", primary_model="repricing_target")
    return engines


def build_market_data_client(settings) -> CoinbaseClient:
    primary_provider = str(settings.data.get("market_data_primary", "binance")).strip().lower()
    fallback_providers = settings.data.get("market_data_fallbacks", ["coinbase"])
    if not isinstance(fallback_providers, list):
        fallback_providers = [fallback_providers]
    provider_order = tuple(
        item
        for item in [primary_provider, *[str(provider).strip().lower() for provider in fallback_providers]]
        if item
    )
    return CoinbaseClient(
        product_id=str(settings.data["coinbase_product_id"]),
        binance_symbol=str(settings.data.get("binance_symbol", "BTCUSDT")),
        kraken_pair=str(settings.data.get("kraken_pair", "XBTUSD")),
        provider_order=provider_order,
        verify_ssl=bool(settings.data["coinbase_verify_ssl"]),
    )


def build_collector(settings, *, persist_snapshots: bool = True):
    coinbase = build_market_data_client(settings)
    kalshi = KalshiClient()
    store_path = snapshot_write_path(settings) if persist_snapshots else snapshot_read_path(settings)
    store = SnapshotStore(store_path, read_only=not persist_snapshots)
    config = HybridCollectorConfig(
        series_ticker=str(settings.collector["series_ticker"]),
        series_tickers=configured_collector_series(settings),
        status=str(settings.collector["status"]),
        market_limit=int(settings.collector["market_limit"]),
        min_minutes_to_expiry=int(settings.collector["min_minutes_to_expiry"]),
        max_minutes_to_expiry=int(settings.collector["max_minutes_to_expiry"]) if settings.collector["max_minutes_to_expiry"] is not None else None,
        reconcile_interval_seconds=int(settings.collector["reconcile_interval_seconds"]),
        spot_refresh_interval_seconds=int(settings.collector["spot_refresh_interval_seconds"]),
        websocket_channels=list(settings.collector["websocket_channels"]),
        persist_snapshots=persist_snapshots,
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


def build_live_paper_trader(settings, *, series_ticker: str | None = None):
    collector = build_collector(settings, persist_snapshots=False)
    paper_series_ticker = configured_paper_series(settings, explicit_series=series_ticker)
    collector.config.series_ticker = paper_series_ticker
    collector.config.series_tickers = [paper_series_ticker]
    coinbase = build_market_data_client(settings)
    store = SnapshotStore(snapshot_read_path(settings), read_only=True)
    _, engine = build_engine(attach_calibration=False, attach_repricing_profile=False)
    paper_starting_capital = float(paper_series_value(settings, paper_series_ticker, "starting_capital", settings.paper.get("starting_capital", 100.0)))
    engine.backtest_config = replace(engine.backtest_config, starting_bankroll=paper_starting_capital)
    settlement_max_markets = int(settings.paper.get("settlement_max_markets_per_cycle", 0))
    refresh_calibrators_on_settlement = bool(settings.paper.get("refresh_calibrators_on_settlement", False))
    settlement_enricher = None
    calibrator_refresher = None
    if settlement_max_markets > 0 and snapshot_read_path(settings) == snapshot_write_path(settings):
        settlement_enricher = SettlementEnricher(
            kalshi_client=KalshiClient(),
            snapshot_store=store,
            config=SettlementEnrichmentConfig(
                series_ticker=paper_series_ticker,
                max_markets=settlement_max_markets,
                recent_hours=int(settings.paper.get("settlement_recent_hours", settings.collector["settlement_recent_hours"])),
            ),
        )
        if refresh_calibrators_on_settlement:
            calibrator_refresher = lambda: attach_replay_calibrators(settings, engine)
    config = LivePaperConfig(
        feature_history_hours=int(settings.paper["feature_history_hours"]),
        poll_interval_seconds=int(settings.paper["poll_interval_seconds"]),
        ledger_path=str(paper_series_value(settings, paper_series_ticker, "ledger_path", settings.paper["ledger_path"])),
        feature_timeframe=str(settings.data["base_timeframe"]),
        volatility_window=int(settings.data["volatility_window"]),
        annualization_factor=float(settings.data["annualization_factor"]),
        feature_cache_refresh_seconds=int(settings.paper.get("feature_cache_refresh_seconds", 15)),
        enable_bankroll_sizing=bool(paper_series_value(settings, paper_series_ticker, "enable_bankroll_sizing", False)),
        bankroll_fraction_per_trade=float(paper_series_value(settings, paper_series_ticker, "bankroll_fraction_per_trade", 1.0)),
        min_cash_buffer=float(paper_series_value(settings, paper_series_ticker, "min_cash_buffer", 0.0)),
        max_contracts_per_trade=int(paper_series_value(settings, paper_series_ticker, "max_contracts_per_trade", 1)),
        respect_tier_size_multiplier=bool(paper_series_value(settings, paper_series_ticker, "respect_tier_size_multiplier", True)),
        settlement_check_interval_seconds=int(settings.paper.get("settlement_check_interval_seconds", 300)),
        settlement_recent_hours=int(settings.paper.get("settlement_recent_hours", settings.collector["settlement_recent_hours"])),
        settlement_max_markets_per_cycle=int(settings.paper.get("settlement_max_markets_per_cycle", 100)),
        snapshot_store_read_only=True,
    )
    return LivePaperTrader(
        collector=collector,
        engine=engine,
        coinbase_client=coinbase,
        snapshot_store=store,
        config=config,
        settlement_enricher=settlement_enricher,
        calibrator_refresher=calibrator_refresher,
    )


def build_backfill_service(settings, *, series_ticker: str, start_ts: int, end_ts: int, period_interval: int | None):
    coinbase = build_market_data_client(settings)
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
            fetched_markets = kalshi.list_markets(series_ticker=None, event_ticker=event_ticker, status="settled", limit=1000)
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
    settings = load_settings()
    if args.command == "show-config":
        print(json.dumps(settings.raw, indent=2))
        return
    if args.command == "spot":
        client = build_market_data_client(settings)
        print(client.get_spot_price())
        return
    if args.command == "demo-backtest":
        client = build_market_data_client(settings)
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
            collector.config.series_tickers = [args.series]
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
                    "series_tickers": collector.config.series_tickers,
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
            collector.config.series_tickers = [args.series]
        if args.max_minutes_to_expiry is not None:
            collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        logging.getLogger(__name__).info(
            "Starting collector for series %s into %s",
            ",".join(collector.config.series_tickers or [collector.config.series_ticker]),
            settings.collector["sqlite_path"],
        )
        try:
            asyncio.run(collector.collect_forever())
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Collector stopped by user.")
        return
    if args.command == "replicate-snapshot-db-once":
        source = str(args.source or snapshot_write_path(settings))
        target = str(args.target or snapshot_read_path(settings))
        print(json.dumps(replicate_snapshot_db_once(source_path=source, target_path=target), indent=2))
        return
    if args.command == "replicate-snapshot-db-forever":
        source = str(args.source or snapshot_write_path(settings))
        target = str(args.target or snapshot_read_path(settings))
        interval_seconds = int(args.interval_seconds or snapshot_replica_interval_seconds(settings))
        logging.getLogger(__name__).info(
            "Starting snapshot DB replica loop from %s to %s every %ss",
            source,
            target,
            interval_seconds,
        )
        try:
            replicate_snapshot_db_forever(
                source_path=source,
                target_path=target,
                interval_seconds=interval_seconds,
            )
        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Snapshot replica loop stopped by user.")
        return
    if args.command == "paper-trade-once":
        trader = build_live_paper_trader(settings, series_ticker=args.series)
        if args.max_minutes_to_expiry is not None:
            trader.collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            trader.collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        result = asyncio.run(trader.run_once())
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == "paper-trade-forever":
        trader = build_live_paper_trader(settings, series_ticker=args.series)
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
    if args.command in {"real-order-preview", "real-order-submit"}:
        executor = build_real_executor(
            settings,
            live=bool(getattr(args, "live", False)),
            series_ticker=getattr(args, "series", None),
            market_ticker=str(args.market),
        )
        request = RealOrderRequest(
            market_ticker=str(args.market),
            side=str(args.side),
            action=str(args.action),
            count=int(args.count),
            yes_price_cents=args.yes_price_cents,
            no_price_cents=args.no_price_cents,
            client_order_id=args.client_order_id,
        )
        if args.command == "real-order-preview":
            print(json.dumps(executor.preview_order(request), indent=2, default=str))
        else:
            print(json.dumps(executor.submit_order(request), indent=2, default=str))
        return
    if args.command == "real-trade-once":
        print(json.dumps(execute_real_trade_once(settings, args), indent=2, default=str))
        return
    if args.command == "real-trade-forever":
        _log = logging.getLogger(__name__)
        series_ticker = configured_real_series(settings, explicit_series=args.series)
        _log.info("Starting real trader mirror for series %s in %s mode", series_ticker, "live" if bool(args.live) else "dry-run")
        poll_interval = int(settings.paper["poll_interval_seconds"])
        trader = build_live_paper_trader(settings, series_ticker=series_ticker)
        if args.max_minutes_to_expiry is not None:
            trader.collector.config.max_minutes_to_expiry = args.max_minutes_to_expiry
        if args.min_minutes_to_expiry is not None:
            trader.collector.config.min_minutes_to_expiry = args.min_minutes_to_expiry
        executor = build_real_executor(settings, live=bool(args.live), series_ticker=series_ticker)
        while True:
            _current_market_ticker: str | None = None
            try:
                housekeeping = executor.cancel_stale_orders()
                if bool(settings.real.get("skip_cycle_after_cancel", True)) and int(housekeeping.get("cancelled_count", 0)) > 0:
                    result = {
                        "status": "skipped_after_cancel",
                        "series_ticker": series_ticker,
                        "reason": "Cancelled stale real order(s); skipping new entry this cycle.",
                        "housekeeping": housekeeping,
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    preview = asyncio.run(trader.preview_once())
                    top_candidates = list(preview.get("top_candidates", []) or [])
                    if not top_candidates:
                        result = {**preview, "series_ticker": series_ticker, "status": "no_candidate", "housekeeping": housekeeping}
                    else:
                        executions = []
                        try:
                            balance_cents = int(executor.kalshi_client.get_balance().get("balance", 0))
                        except Exception:
                            balance_cents = None
                        for top in top_candidates:
                            _current_market_ticker = str(top["market_ticker"])
                            side = str(top["side"])
                            market_ticker = str(top["market_ticker"])
                            entry_price_cents = int(top["entry_price_cents"]) if top.get("entry_price_cents") is not None else 50
                            max_notional = float(settings.real.get("max_trade_notional", 10.0))
                            bankroll_fraction = float(settings.real.get("bankroll_fraction_per_trade", 0.5))
                            if balance_cents is not None:
                                dynamic_notional = (balance_cents / 100.0) * bankroll_fraction
                                max_notional = min(max_notional, dynamic_notional)
                            size_multiplier = max(float(top.get("size_multiplier") or 1.0), 0.0)
                            max_notional *= size_multiplier
                            max_notional = max(max_notional, entry_price_cents / 100.0)
                            count = max(1, int(max_notional / (entry_price_cents / 100.0)))
                            request = RealOrderRequest(
                                market_ticker=market_ticker,
                                side=side,
                                action="buy",
                                count=count,
                                yes_price_cents=entry_price_cents if side == "yes" else None,
                                no_price_cents=entry_price_cents if side == "no" else None,
                                client_order_id=f"cryptobot-{market_ticker.lower().replace('.', '-')}-{int(datetime.now(timezone.utc).timestamp())}",
                            )
                            execution = executor.submit_order(request)
                            executions.append({"order": request.to_api_payload(), "execution": execution})
                            exec_status = str(execution.get("exchange_status", execution.get("status", "")))
                            if exec_status in ("blocked",):
                                break
                        result = {**preview, "series_ticker": series_ticker, "executions": executions, "housekeeping": housekeeping}
                print(json.dumps(result, indent=2, default=str))
            except KeyboardInterrupt:
                _log.info("Real trader stopped by user.")
                return
            except Exception as exc:
                _log.warning("Real trade cycle error (will retry next cycle): %s | market=%s count=%s price=%s",
                    exc, _current_market_ticker,
                    locals().get("count"), locals().get("entry_price_cents"))
            asyncio.run(asyncio.sleep(poll_interval))
        return
    if args.command == "real-trading-status":
        series_ticker = configured_real_series(settings, explicit_series=args.series)
        executor = build_real_executor(settings, live=False, series_ticker=series_ticker)
        reconcile_summary = executor.reconcile_exchange_state()
        exchange_state = executor.fetch_exchange_state(order_limit=200, fill_limit=200)
        ledger_entries = executor._read_ledger()
        print(
            json.dumps(
                {
                    "series_ticker": series_ticker,
                    "ledger_path": str(executor.ledger_path),
                    "kill_switch_enabled": executor.is_kill_switch_enabled(),
                    "reconciliation": reconcile_summary,
                    "resting_orders_count": len(exchange_state["resting_orders"]),
                    "open_positions_count": len(exchange_state["open_positions"]),
                    "ledger_entries": len(ledger_entries),
                    "last_ledger_entry": ledger_entries[-1] if ledger_entries else None,
                },
                indent=2,
                default=str,
            )
        )
        return
    if args.command == "real-kill-switch":
        executor = build_real_executor(settings, live=False, series_ticker=args.series)
        enable = bool(args.enable)
        if not args.enable and not args.disable:
            enable = True
        if args.disable:
            enable = False
        print(json.dumps(executor.set_kill_switch(enable, reason=str(args.reason or "")), indent=2, default=str))
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
    if args.command == "rebuild-repricing-profile":
        _, engine = build_engine(attach_calibration=False, attach_repricing_profile=True)
        profile_model = engine.models.get("repricing_target")
        samples = getattr(getattr(profile_model, "profile", None), "sample_count", 0)
        print(
            json.dumps(
                {
                    "repricing_target_available": profile_model is not None,
                    "training_samples": samples,
                    "cache_path": settings.raw.get("repricing_target", {}).get("cache_path"),
                },
                indent=2,
            )
        )
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
        )
        _, engine = build_engine()
        engines = build_comparison_engines(settings, engine)
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
    if args.command == "replay-failure-analysis":
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
        )
        _, engine = build_engine()
        engines = build_comparison_engines(settings, engine)
        report = build_model_comparison_report(engines, snapshots=dataset.snapshots, feature_frame=dataset.feature_frame)
        requested_model = str(args.model)
        if requested_model not in report["models"]:
            available_models = sorted(report["models"].keys())
            raise SystemExit(f"Unknown model '{requested_model}'. Available: {', '.join(available_models)}")
        analysis = build_failure_analysis(
            report["models"][requested_model],
            strategy_mode=str(args.mode),
            top_n=int(args.top),
        )
        output = {
            "model": requested_model,
            "strategy_mode": str(args.mode),
            "snapshot_count": len(dataset.snapshots),
            "feature_rows": len(dataset.feature_frame),
            "filters": {
                "near_money_bps": args.near_money_bps,
                "max_minutes_to_expiry": args.max_minutes_to_expiry,
                "min_price_cents": args.min_price_cents,
                "max_price_cents": args.max_price_cents,
            },
            **analysis,
        }
        print(json.dumps(output, indent=2, default=str))
        return
    if args.command == "replay-maker-proxy":
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
        )
        _, engine = build_engine()
        engines = build_comparison_engines(settings, engine)
        requested_model = str(args.model)
        if requested_model not in engines:
            available_models = sorted(engines.keys())
            raise SystemExit(f"Unknown model '{requested_model}'. Available: {', '.join(available_models)}")
        selected_engine = engines[requested_model]
        selected_engine.maker_simulation_config = MakerSimulationConfig(
            max_wait_seconds=int(args.maker_max_wait_seconds),
            min_fill_probability=float(args.maker_min_fill_probability),
            stale_quote_age_seconds=int(args.maker_stale_quote_age_seconds),
            max_posted_spread_cents=int(args.maker_max_posted_spread_cents),
            min_liquidity_score=float(args.maker_min_liquidity_score),
            max_concurrent_positions_per_side=int(args.maker_max_concurrent_positions_per_side),
        )
        taker_result = selected_engine.run_strategy_with_entry_style(
            str(args.mode),
            dataset.snapshots,
            dataset.feature_frame,
            entry_style="taker",
        )
        maker_result = selected_engine.run_strategy_with_entry_style(
            str(args.mode),
            dataset.snapshots,
            dataset.feature_frame,
            entry_style="maker",
        )
        output = {
            "model": requested_model,
            "strategy_mode": str(args.mode),
            "snapshot_count": len(dataset.snapshots),
            "feature_rows": len(dataset.feature_frame),
            "filters": {
                "near_money_bps": args.near_money_bps,
                "max_minutes_to_expiry": args.max_minutes_to_expiry,
                "min_price_cents": args.min_price_cents,
                "max_price_cents": args.max_price_cents,
            },
            "maker_simulation": {
                "max_wait_seconds": args.maker_max_wait_seconds,
                "min_fill_probability": args.maker_min_fill_probability,
                "stale_quote_age_seconds": args.maker_stale_quote_age_seconds,
                "max_posted_spread_cents": args.maker_max_posted_spread_cents,
                "min_liquidity_score": args.maker_min_liquidity_score,
                "max_concurrent_positions_per_side": args.maker_max_concurrent_positions_per_side,
            },
            "taker_entry": taker_result.summary,
            "maker_entry_proxy": maker_result.summary,
            "maker_minus_taker_pnl": round(float(maker_result.summary.get("pnl", 0.0)) - float(taker_result.summary.get("pnl", 0.0)), 2),
            "maker_minus_taker_roi": round(float(maker_result.summary.get("roi", 0.0)) - float(taker_result.summary.get("roi", 0.0)), 2),
        }
        print(json.dumps(output, indent=2, default=str))
        return
    if args.command == "replay-maker-sim":
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
            cache_dir=replay_cache_dir(settings),
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
            cache_dir=replay_cache_dir(settings),
        )
        _, engine = build_engine()
        engines = build_comparison_engines(settings, engine)
        requested_model = str(args.model)
        if requested_model not in engines:
            available_models = sorted(engines.keys())
            raise SystemExit(f"Unknown model '{requested_model}'. Available: {', '.join(available_models)}")
        selected_engine = engines[requested_model]
        selected_engine.maker_simulation_config = MakerSimulationConfig(
            max_wait_seconds=int(args.maker_max_wait_seconds),
            min_fill_probability=float(args.maker_min_fill_probability),
            stale_quote_age_seconds=int(args.maker_stale_quote_age_seconds),
            max_posted_spread_cents=int(args.maker_max_posted_spread_cents),
            min_liquidity_score=float(args.maker_min_liquidity_score),
            max_concurrent_positions_per_side=int(args.maker_max_concurrent_positions_per_side),
        )
        taker_result = selected_engine.run_strategy_with_entry_style(
            str(args.mode),
            dataset.snapshots,
            dataset.feature_frame,
            entry_style="taker",
        )
        maker_proxy_result = selected_engine.run_strategy_with_entry_style(
            str(args.mode),
            dataset.snapshots,
            dataset.feature_frame,
            entry_style="maker",
        )
        maker_sim_result = selected_engine.run_strategy_with_entry_style(
            str(args.mode),
            dataset.snapshots,
            dataset.feature_frame,
            entry_style="maker_sim",
        )
        output = {
            "model": requested_model,
            "strategy_mode": str(args.mode),
            "snapshot_count": len(dataset.snapshots),
            "feature_rows": len(dataset.feature_frame),
            "filters": {
                "near_money_bps": args.near_money_bps,
                "max_minutes_to_expiry": args.max_minutes_to_expiry,
                "min_price_cents": args.min_price_cents,
                "max_price_cents": args.max_price_cents,
            },
            "maker_simulation": {
                "max_wait_seconds": args.maker_max_wait_seconds,
                "min_fill_probability": args.maker_min_fill_probability,
                "stale_quote_age_seconds": args.maker_stale_quote_age_seconds,
                "max_posted_spread_cents": args.maker_max_posted_spread_cents,
                "min_liquidity_score": args.maker_min_liquidity_score,
                "max_concurrent_positions_per_side": args.maker_max_concurrent_positions_per_side,
            },
            "taker_entry": taker_result.summary,
            "maker_entry_proxy": maker_proxy_result.summary,
            "maker_entry_conservative": maker_sim_result.summary,
            "maker_proxy_minus_taker_pnl": round(float(maker_proxy_result.summary.get("pnl", 0.0)) - float(taker_result.summary.get("pnl", 0.0)), 2),
            "maker_proxy_minus_taker_roi": round(float(maker_proxy_result.summary.get("roi", 0.0)) - float(taker_result.summary.get("roi", 0.0)), 2),
            "maker_sim_minus_taker_pnl": round(float(maker_sim_result.summary.get("pnl", 0.0)) - float(taker_result.summary.get("pnl", 0.0)), 2),
            "maker_sim_minus_taker_roi": round(float(maker_sim_result.summary.get("roi", 0.0)) - float(taker_result.summary.get("roi", 0.0)), 2),
        }
        print(json.dumps(output, indent=2, default=str))
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
        engines = build_comparison_engines(settings, engine)
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
        return
    if args.command == "replay-bankroll-sim":
        store = SnapshotStore(str(args.sqlite_path))
        snapshots = store.load_snapshots(series_ticker=str(args.series))
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            cache_dir=replay_cache_dir(settings),
        )
        _, engine = build_engine()
        result = engine.run_strategy(str(args.mode), dataset.snapshots, dataset.feature_frame)
        simulated_trades, summary = simulate_bankroll_constrained_compounding(
            result.trades,
            config=BankrollSizingConfig(
                starting_bankroll=float(args.starting_bankroll),
                bankroll_fraction_per_trade=float(args.bankroll_fraction_per_trade),
                min_cash_buffer=float(args.min_cash_buffer),
                max_contracts_per_trade=int(args.max_contracts_per_trade),
                allow_fractional_contracts=bool(args.allow_fractional_contracts),
            ),
        )
        output = {
            "series_ticker": str(args.series),
            "mode": str(args.mode),
            "sqlite_path": str(args.sqlite_path),
            "snapshot_count": len(dataset.snapshots),
            "feature_rows": len(dataset.feature_frame),
            "fixed_size_summary": result.summary,
            "bankroll_simulation": summary,
            "recent_simulated_trades": simulated_trades[
                [
                    column
                    for column in (
                        "entry_time",
                        "market_ticker",
                        "side",
                        "entry_price_cents",
                        "requested_contracts",
                        "simulated_contracts",
                        "simulated_entry_notional",
                        "simulated_realized_pnl",
                        "bankroll_before",
                        "bankroll_after",
                        "simulated_taken",
                        "simulated_skipped_reason",
                    )
                    if column in simulated_trades.columns
                ]
            ]
            .tail(10)
            .to_dict(orient="records")
            if not simulated_trades.empty
            else [],
        }
        print(json.dumps(output, indent=2, default=str))
        return
    if args.command == "replay-walkforward-calibration":
        store = SnapshotStore(str(args.sqlite_path))
        snapshots = store.load_snapshots(series_ticker=str(args.series))
        dataset = build_replay_dataset(
            snapshots,
            volatility_window=int(settings.data["volatility_window"]),
            annualization_factor=float(settings.data["annualization_factor"]),
            cache_dir=replay_cache_dir(settings),
        )
        train_snapshots, test_snapshots, cutoff = split_snapshots_by_fraction(dataset.snapshots, float(args.train_fraction))
        if not train_snapshots or not test_snapshots or cutoff is None:
            raise SystemExit("Need enough snapshots to form both training and test windows.")

        _, calibrated_engine = build_engine(attach_calibration=False)
        calibration_config = settings.raw.get("calibration", {})
        calibrators = build_engine_calibrators(
            calibrated_engine,
            snapshots=train_snapshots,
            feature_frame=dataset.feature_frame,
            bucket_width=float(calibration_config.get("bucket_width", 0.05)),
            min_samples=int(calibration_config.get("min_samples", 50)),
            min_bucket_count=int(calibration_config.get("min_bucket_count", 3)),
        )
        calibrated_engine.calibrators = calibrators
        calibrated_result = calibrated_engine.run_strategy(str(args.mode), test_snapshots, dataset.feature_frame)

        _, uncalibrated_engine = build_engine(attach_calibration=False)
        uncalibrated_result = uncalibrated_engine.run_strategy(str(args.mode), test_snapshots, dataset.feature_frame)

        output = {
            "series_ticker": str(args.series),
            "mode": str(args.mode),
            "sqlite_path": str(args.sqlite_path),
            "train_fraction": round(float(args.train_fraction), 4),
            "cutoff_observed_at": cutoff.isoformat(),
            "train_snapshot_count": len(train_snapshots),
            "test_snapshot_count": len(test_snapshots),
            "feature_rows": len(dataset.feature_frame),
            "calibration_sample_counts": {name: calibrator.sample_count for name, calibrator in calibrators.items()},
            "walkforward_calibrated_summary": calibrated_result.summary,
            "uncalibrated_summary": uncalibrated_result.summary,
            "calibrated_minus_uncalibrated_pnl": round(
                float(calibrated_result.summary.get("pnl", 0.0)) - float(uncalibrated_result.summary.get("pnl", 0.0)),
                2,
            ),
            "calibrated_minus_uncalibrated_win_rate": round(
                float(calibrated_result.summary.get("win_rate", 0.0)) - float(uncalibrated_result.summary.get("win_rate", 0.0)),
                2,
            ),
        }
        print(json.dumps(output, indent=2, default=str))
        return


if __name__ == "__main__":
    main()
