from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import pandas as pd

from kalshi_btc_bot.backtest.fills import apply_entry_fill, apply_exit_fill
from kalshi_btc_bot.backtest.metrics import summarize_trades
from kalshi_btc_bot.models.base import ProbabilityModel
from kalshi_btc_bot.signals.fusion import FusionConfig, fuse_signals
from kalshi_btc_bot.signals.engine import SignalConfig, generate_signal
from kalshi_btc_bot.trading.exits import ExitConfig, evaluate_exit
from kalshi_btc_bot.trading.risk import RiskConfig, RiskManager
from kalshi_btc_bot.types import BacktestResult, MarketSnapshot
from kalshi_btc_bot.utils.math import probability_to_cents
from kalshi_btc_bot.utils.time import ensure_utc


@dataclass(frozen=True)
class BacktestConfig:
    default_contracts: int
    entry_slippage_cents: int
    exit_slippage_cents: int
    fee_rate_bps: float
    starting_bankroll: float
    allow_reentry: bool = False


@dataclass(frozen=True)
class ReplayDiagnostics:
    summary: dict[str, int | float]
    per_market: pd.DataFrame
    reason_counts: dict[str, int]


class BacktestEngine:
    def __init__(
        self,
        *,
        model: ProbabilityModel,
        signal_config: SignalConfig,
        exit_config: ExitConfig,
        backtest_config: BacktestConfig,
        models: dict[str, ProbabilityModel] | None = None,
        fusion_config: FusionConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self.model = model
        self.models = models or {getattr(model, "model_name", "default"): model}
        self.signal_config = signal_config
        self.exit_config = exit_config
        self.backtest_config = backtest_config
        self.fusion_config = fusion_config
        self.risk_config = risk_config or RiskConfig(0.0, 0.0, 0, 0, 0.0, 0.0)

    def compare_strategies(self, snapshots: list[MarketSnapshot], feature_frame: pd.DataFrame) -> dict[str, BacktestResult | dict]:
        hold = self.run_strategy("hold_to_settlement", snapshots, feature_frame)
        early = self.run_strategy("early_exit", snapshots, feature_frame)
        return {
            "hold_to_settlement": hold,
            "early_exit": early,
            "summary": {
                "hold_pnl": hold.summary["pnl"],
                "early_exit_pnl": early.summary["pnl"],
                "early_exit_improvement": round(early.summary["pnl"] - hold.summary["pnl"], 2),
            },
        }

    def diagnose_replay(self, snapshots: list[MarketSnapshot], feature_frame: pd.DataFrame) -> ReplayDiagnostics:
        if not snapshots:
            empty = pd.DataFrame()
            return ReplayDiagnostics(
                summary={
                    "snapshot_count": 0,
                    "market_count": 0,
                    "entries": 0,
                    "skipped_after_expiry": 0,
                    "skipped_missing_volatility": 0,
                    "skipped_no_settlement": 0,
                    "skipped_no_action": 0,
                },
                per_market=empty,
                reason_counts={},
            )

        snapshot_df = pd.DataFrame([self._snapshot_to_row(snapshot) for snapshot in snapshots]).sort_values(["market_ticker", "observed_at"])
        per_market_rows: list[dict] = []
        totals = {
            "snapshot_count": len(snapshots),
            "market_count": int(snapshot_df["market_ticker"].nunique()),
            "entries": 0,
            "skipped_after_expiry": 0,
            "skipped_missing_volatility": 0,
            "skipped_no_settlement": 0,
            "skipped_no_action": 0,
        }
        reason_counts: Counter[str] = Counter()
        for market_ticker, market_group in snapshot_df.groupby("market_ticker", sort=False):
            market_snapshots = [self._row_to_snapshot(row) for _, row in market_group.iterrows()]
            reasons = {
                "market_ticker": market_ticker,
                "snapshot_count": len(market_snapshots),
                "entries": 0,
                "skipped_after_expiry": 0,
                "skipped_missing_volatility": 0,
                "skipped_no_settlement": 0,
                "skipped_no_action": 0,
            }
            has_settlement = any(snapshot.settlement_price is not None for snapshot in market_snapshots)
            if not has_settlement:
                reasons["skipped_no_settlement"] = len(market_snapshots)
                totals["skipped_no_settlement"] += len(market_snapshots)
            for snapshot in market_snapshots:
                if snapshot.observed_at >= snapshot.expiry:
                    reasons["skipped_after_expiry"] += 1
                    totals["skipped_after_expiry"] += 1
                    continue
                volatility = self._lookup_volatility(feature_frame, snapshot.observed_at)
                if volatility is None:
                    reasons["skipped_missing_volatility"] += 1
                    totals["skipped_missing_volatility"] += 1
                    continue
                if not has_settlement:
                    continue
                snapshot.metadata["volatility"] = volatility
                snapshot.metadata["recent_log_return"] = self._lookup_recent_log_return(feature_frame, snapshot.observed_at)
                estimate = self.model.estimate(snapshot, volatility)
                signal = generate_signal(snapshot, estimate, self.signal_config, as_of=snapshot.observed_at)
                if signal.action == "no_action" or signal.entry_price_cents is None or signal.side is None:
                    reasons["skipped_no_action"] += 1
                    totals["skipped_no_action"] += 1
                    reason_counts[signal.reason] += 1
                    continue
                reasons["entries"] += 1
                totals["entries"] += 1
            per_market_rows.append(reasons)
        return ReplayDiagnostics(summary=totals, per_market=pd.DataFrame(per_market_rows), reason_counts=dict(reason_counts))

    def evaluate_snapshot(self, snapshot: MarketSnapshot, feature_frame: pd.DataFrame):
        volatility = self._lookup_volatility(feature_frame, snapshot.observed_at)
        if volatility is None:
            return None, None
        snapshot.metadata["volatility"] = volatility
        snapshot.metadata["recent_log_return"] = self._lookup_recent_log_return(feature_frame, snapshot.observed_at)
        return self._build_signal(snapshot, volatility)

    def evaluate_live_exit(
        self,
        *,
        snapshot: MarketSnapshot,
        side: str,
        entry_price_cents: int,
        contracts: int,
        feature_frame: pd.DataFrame,
    ):
        volatility = self._lookup_volatility(feature_frame, snapshot.observed_at)
        if volatility is None:
            return None
        snapshot.metadata["volatility"] = volatility
        snapshot.metadata["recent_log_return"] = self._lookup_recent_log_return(feature_frame, snapshot.observed_at)
        estimate = self.model.estimate(snapshot, volatility)
        fair_value = probability_to_cents(estimate.probability if side == "yes" else 1.0 - estimate.probability)
        return evaluate_exit(
            snapshot=snapshot,
            side=side,
            entry_price_cents=entry_price_cents,
            contracts=contracts,
            fair_value_cents=fair_value,
            model_probability=estimate.probability if side == "yes" else 1.0 - estimate.probability,
            config=self.exit_config,
            as_of=snapshot.observed_at,
        )

    def run_strategy(self, mode: str, snapshots: list[MarketSnapshot], feature_frame: pd.DataFrame) -> BacktestResult:
        if not snapshots:
            empty = pd.DataFrame()
            summary = summarize_trades(empty, starting_bankroll=self.backtest_config.starting_bankroll)
            return BacktestResult(strategy_mode=mode, trades=empty, summary=summary)
        snapshot_df = pd.DataFrame([self._snapshot_to_row(snapshot) for snapshot in snapshots]).sort_values(["market_ticker", "observed_at"])
        rows: list[dict] = []
        risk_manager = RiskManager(self.risk_config)
        risk_manager.initialize(self.backtest_config.starting_bankroll)
        for market_ticker, market_group in snapshot_df.groupby("market_ticker", sort=False):
            rows.extend(self._run_market_path(mode, market_ticker, market_group, feature_frame, rows, risk_manager))
        trades = pd.DataFrame(rows)
        if not trades.empty:
            trades = trades.sort_values("entry_time").reset_index(drop=True)
        summary = summarize_trades(trades, starting_bankroll=self.backtest_config.starting_bankroll)
        return BacktestResult(strategy_mode=mode, trades=trades, summary=summary)

    def _run_market_path(
        self,
        mode: str,
        market_ticker: str,
        market_group: pd.DataFrame,
        feature_frame: pd.DataFrame,
        existing_rows: list[dict],
        risk_manager: RiskManager,
    ) -> list[dict]:
        rows: list[dict] = []
        entered = False
        contracts = self.backtest_config.default_contracts
        market_snapshots = [self._row_to_snapshot(row) for _, row in market_group.iterrows()]
        for index, snapshot in enumerate(market_snapshots):
            if not self.backtest_config.allow_reentry and entered:
                break
            if snapshot.observed_at >= snapshot.expiry:
                continue
            volatility = self._lookup_volatility(feature_frame, snapshot.observed_at)
            if volatility is None:
                continue
            snapshot.metadata["volatility"] = volatility
            snapshot.metadata["recent_log_return"] = self._lookup_recent_log_return(feature_frame, snapshot.observed_at)
            signal, estimate = self._build_signal(snapshot, volatility)
            if signal.action == "no_action" or signal.entry_price_cents is None or signal.side is None:
                continue
            entered = True
            entry_fill = apply_entry_fill(
                signal.entry_price_cents,
                self.backtest_config.entry_slippage_cents,
                self.backtest_config.fee_rate_bps,
                spread_cents=signal.spread_cents,
                volume=snapshot.volume,
                open_interest=snapshot.open_interest,
                contracts=contracts,
            )
            entry_fill_cents = entry_fill.price_cents
            entry_fee = entry_fill.fees_paid
            entry_notional = round((entry_fill_cents * contracts) / 100.0 + entry_fee, 4)
            realized_so_far = sum(float(row["realized_pnl"]) for row in existing_rows + rows)
            current_equity = self.backtest_config.starting_bankroll + realized_so_far
            risk_check = risk_manager.check_entry(
                entry_notional=entry_notional,
                expiry=snapshot.expiry,
                open_position_count=0,
                current_equity=current_equity,
            )
            if not risk_check.allowed:
                continue
            risk_manager.record_entry(entry_notional=entry_notional, expiry=snapshot.expiry)
            exit_snapshot, exit_price_cents, exit_fee, exit_trigger = self._resolve_exit(
                mode=mode,
                entry_index=index,
                snapshots=market_snapshots,
                side=signal.side,
                entry_price_cents=entry_fill_cents,
                contracts=contracts,
                feature_frame=feature_frame,
            )
            realized_pnl = round(((exit_price_cents - entry_fill_cents) * contracts) / 100.0 - entry_fee - exit_fee, 2)
            risk_manager.record_exit(
                realized_pnl=realized_pnl,
                expiry=snapshot.expiry,
                current_equity=current_equity + realized_pnl,
            )
            contract_won = self._contract_wins(market_snapshots[-1], signal.side)
            settlement_price_cents = 100 if contract_won else 0
            hold_to_settlement_pnl = round(((settlement_price_cents - entry_fill_cents) * contracts) / 100.0 - entry_fee, 2)
            hold_minutes = round((exit_snapshot.observed_at - snapshot.observed_at).total_seconds() / 60.0, 2)
            closing_mark = self._pre_expiry_mark(market_snapshots, signal.side)
            clv_cents = round((closing_mark - entry_fill_cents), 2) if closing_mark is not None else 0.0
            rows.append(
                {
                    "market_ticker": market_ticker,
                    "entry_time": snapshot.observed_at,
                    "exit_time": exit_snapshot.observed_at,
                    "side": signal.side,
                    "entry_price_cents": entry_fill_cents,
                    "exit_price_cents": exit_price_cents,
                    "contracts": contracts,
                    "edge": signal.edge,
                    "raw_edge": signal.raw_edge,
                    "raw_model_probability": signal.raw_model_probability,
                    "model_probability": signal.model_probability,
                    "market_probability": signal.market_probability,
                    "entry_notional": entry_notional,
                    "realized_pnl": realized_pnl,
                    "hold_to_settlement_pnl": hold_to_settlement_pnl,
                    "exit_trigger": exit_trigger,
                    "hold_minutes": hold_minutes,
                    "exit_improved_result": realized_pnl > hold_to_settlement_pnl,
                    "clv_cents": clv_cents,
                    "fair_value_cents": signal.fair_value_cents,
                    "settlement_price_cents": settlement_price_cents,
                    "contract_won": contract_won,
                    "contract_type": snapshot.contract_type,
                    "quality_score": signal.quality_score,
                    "spread_cents": signal.spread_cents,
                    "liquidity_penalty": signal.liquidity_penalty,
                    "signal_reason": signal.reason,
                    "model_name": estimate.model_name,
                }
            )
        return rows

    def _resolve_exit(
        self,
        *,
        mode: str,
        entry_index: int,
        snapshots: list[MarketSnapshot],
        side: str,
        entry_price_cents: int,
        contracts: int,
        feature_frame: pd.DataFrame,
    ) -> tuple[MarketSnapshot, int, float, str]:
        if mode == "hold_to_settlement":
            settlement_snapshot = snapshots[-1]
            settlement_price = self._settlement_price_cents(settlement_snapshot, side)
            return settlement_snapshot, settlement_price, 0.0, "settlement"

        for snapshot in snapshots[entry_index + 1 :]:
            volatility = max(float(snapshot.metadata.get("volatility", 0.05)), 0.05)
            snapshot.metadata.setdefault("recent_log_return", self._lookup_recent_log_return(feature_frame, snapshot.observed_at))
            estimate = self.model.estimate(snapshot, volatility)
            fair_value = probability_to_cents(estimate.probability if side == "yes" else 1.0 - estimate.probability)
            decision = evaluate_exit(
                snapshot=snapshot,
                side=side,
                entry_price_cents=entry_price_cents,
                contracts=contracts,
                fair_value_cents=fair_value,
                model_probability=estimate.probability if side == "yes" else 1.0 - estimate.probability,
                config=self.exit_config,
                as_of=snapshot.observed_at,
            )
            if decision.action == "exit" and decision.exit_price_cents is not None:
                exit_fill = apply_exit_fill(
                    decision.exit_price_cents,
                    self.backtest_config.exit_slippage_cents,
                    self.backtest_config.fee_rate_bps,
                    spread_cents=self._spread_cents(snapshot, side),
                    volume=snapshot.volume,
                    open_interest=snapshot.open_interest,
                    contracts=contracts,
                )
                return snapshot, exit_fill.price_cents, exit_fill.fees_paid, str(decision.trigger)

        settlement_snapshot = snapshots[-1]
        settlement_price = self._settlement_price_cents(settlement_snapshot, side)
        return settlement_snapshot, settlement_price, 0.0, "settlement"

    @staticmethod
    def _lookup_volatility(feature_frame: pd.DataFrame, observed_at) -> float | None:
        if feature_frame.empty:
            return None
        eligible = feature_frame.loc[feature_frame.index <= ensure_utc(observed_at)]
        if eligible.empty:
            return None
        latest = eligible.iloc[-1]
        value = latest.get("realized_volatility")
        return None if pd.isna(value) else float(value)

    def _build_signal(self, snapshot: MarketSnapshot, volatility: float):
        estimates = {name: model.estimate(snapshot, volatility) for name, model in self.models.items()}
        signals = {
            name: generate_signal(snapshot, estimate, self.signal_config, as_of=snapshot.observed_at)
            for name, estimate in estimates.items()
        }
        if self.fusion_config is None or self.fusion_config.mode != "hybrid":
            selected_name = self.fusion_config.primary_model if self.fusion_config else next(iter(self.models))
            return signals[selected_name], estimates[selected_name]
        fused = fuse_signals(signals, config=self.fusion_config)
        estimate = estimates[self.fusion_config.primary_model]
        return fused, estimate

    @staticmethod
    def _pre_expiry_mark(snapshots: list[MarketSnapshot], side: str) -> int | None:
        for snapshot in reversed(snapshots):
            if snapshot.observed_at < snapshot.expiry:
                if side == "yes" and snapshot.yes_bid is not None:
                    return int(round(snapshot.yes_bid * 100.0))
                if side == "no" and snapshot.no_bid is not None:
                    return int(round(snapshot.no_bid * 100.0))
        return None

    @staticmethod
    def _spread_cents(snapshot: MarketSnapshot, side: str) -> int | None:
        if side == "yes":
            if snapshot.yes_ask is None or snapshot.yes_bid is None:
                return None
            return int(round(snapshot.yes_ask * 100.0)) - int(round(snapshot.yes_bid * 100.0))
        if snapshot.no_ask is None or snapshot.no_bid is None:
            return None
        return int(round(snapshot.no_ask * 100.0)) - int(round(snapshot.no_bid * 100.0))

    @staticmethod
    def _lookup_recent_log_return(feature_frame: pd.DataFrame, observed_at) -> float:
        if feature_frame.empty:
            return 0.0
        eligible = feature_frame.loc[feature_frame.index <= ensure_utc(observed_at)]
        if eligible.empty:
            return 0.0
        latest = eligible.iloc[-1]
        value = latest.get("log_return")
        return 0.0 if pd.isna(value) else float(value)

    @staticmethod
    def _settlement_price_cents(snapshot: MarketSnapshot, side: str) -> int:
        target = 100 if BacktestEngine._contract_wins(snapshot, side) else 0
        return target

    @staticmethod
    def _contract_wins(snapshot: MarketSnapshot, side: str) -> bool:
        if snapshot.settlement_price is not None and 0.0 <= float(snapshot.settlement_price) <= 1.0:
            outcome = float(snapshot.settlement_price) >= 0.5
            return outcome if side == "yes" else not outcome

        spot = snapshot.settlement_price if snapshot.settlement_price is not None else snapshot.spot_price
        if snapshot.contract_type == "threshold":
            threshold = float(snapshot.threshold)
            above = spot >= threshold
            outcome = not above if snapshot.direction == "below" else above
        elif snapshot.contract_type == "range":
            outcome = float(snapshot.range_low) <= spot < float(snapshot.range_high)
        else:
            reference = snapshot.threshold if snapshot.threshold is not None else snapshot.metadata.get("reference_price", snapshot.spot_price)
            up = spot >= float(reference)
            outcome = not up if snapshot.direction == "down" else up
        return outcome if side == "yes" else not outcome

    @staticmethod
    def _snapshot_to_row(snapshot: MarketSnapshot) -> dict:
        row = snapshot.__dict__.copy()
        row["observed_at"] = pd.Timestamp(snapshot.observed_at)
        row["expiry"] = pd.Timestamp(snapshot.expiry)
        return row

    @staticmethod
    def _row_to_snapshot(row: pd.Series) -> MarketSnapshot:
        payload = row.to_dict()
        payload["observed_at"] = pd.Timestamp(payload["observed_at"]).to_pydatetime()
        payload["expiry"] = pd.Timestamp(payload["expiry"]).to_pydatetime()
        return MarketSnapshot(**payload)
