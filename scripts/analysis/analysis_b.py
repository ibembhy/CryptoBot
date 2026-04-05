from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_btc_bot.models.gbm_threshold import terminal_probability_above
from scripts.analysis.cleaning import (
    SERIES_NEAR_MONEY_BPS,
    add_market_derived_columns,
    assign_series_buckets,
    build_global_spot_features,
    deduplicate_snapshots,
)
from scripts.analysis.loaders import AnalysisContext, load_settlements, load_snapshots


@dataclass(frozen=True)
class EntrySpec:
    tte_min: float
    tte_max: float
    dist_bps_max: float
    min_edge: float
    price_min_cents: int
    price_max_cents: int
    max_spread_cents: int
    min_volume: float
    side_mode: str  # yes_only / no_only / best_edge


@dataclass(frozen=True)
class ExitSpec:
    mode: str  # hold / stop / take_profit / stop_take_profit / time_exit / fair_value
    stop_loss_cents: int | None = None
    take_profit_cents: int | None = None
    time_exit_minutes_before_expiry: int | None = None


@dataclass(frozen=True)
class MarketArrays:
    observed_at_ns: np.ndarray
    tte_min: np.ndarray
    model_probability_yes: np.ndarray
    yes_bid_cents: np.ndarray
    no_bid_cents: np.ndarray
    settlement_result: str


def run_analysis_b(
    context: AnalysisContext,
    series_list: list[str],
    *,
    focused: bool = True,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    summary_lines = ["Analysis B - Entry/exit parameter sweep", ""]

    for series in series_list:
        frame = _prepare_series_frame(context, series)
        market_arrays = _build_market_arrays(frame)
        entry_specs, exit_specs = _build_parameter_grid(series, focused=focused)
        candidate_frame = _build_entry_candidate_frame(frame)

        results: list[dict[str, object]] = []
        for entry_spec in entry_specs:
            selected_entries = _select_entries(candidate_frame, entry_spec)

            for exit_spec in exit_specs:
                trades = _simulate_trades_batch(selected_entries, market_arrays, exit_spec)
                metrics = _summarize_simulation(trades)
                results.append(
                    {
                        "series": series,
                        "tte_window": f"{entry_spec.tte_min:g}-{entry_spec.tte_max:g}",
                        "dist_bps_max": entry_spec.dist_bps_max,
                        "min_edge": entry_spec.min_edge,
                        "price_window": f"{entry_spec.price_min_cents}-{entry_spec.price_max_cents}",
                        "max_spread_cents": entry_spec.max_spread_cents,
                        "min_volume": entry_spec.min_volume,
                        "side_mode": entry_spec.side_mode,
                        "exit_mode": exit_spec.mode,
                        "stop_loss_cents": exit_spec.stop_loss_cents,
                        "take_profit_cents": exit_spec.take_profit_cents,
                        "time_exit_minutes_before_expiry": exit_spec.time_exit_minutes_before_expiry,
                        **metrics,
                    }
                )

        result_frame = pd.DataFrame(results).sort_values(
            by=["sharpe_like", "total_pnl_cents", "win_rate"],
            ascending=[False, False, False],
        )
        out_path = context.output_dir / f"{series.lower()}_analysis_b_sweep.csv"
        result_frame.to_csv(out_path, index=False)
        outputs[f"{series}_sweep"] = out_path

        top20_path = context.output_dir / f"{series.lower()}_analysis_b_top20.csv"
        result_frame.head(20).to_csv(top20_path, index=False)
        outputs[f"{series}_top20"] = top20_path

        best_row = result_frame.iloc[0].to_dict() if not result_frame.empty else {}
        summary_lines.extend(
            [
                f"[{series}]",
                f"rows={len(frame):,}",
                f"entry_specs={len(entry_specs):,}",
                f"exit_specs={len(exit_specs):,}",
                f"combinations={len(result_frame):,}",
                f"best={best_row}",
                "",
            ]
        )

    summary_path = context.output_dir / "analysis_b_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    outputs["summary"] = summary_path
    return outputs


def _prepare_series_frame(context: AnalysisContext, series: str) -> pd.DataFrame:
    cache_path = context.output_dir / f"{series.lower()}_analysis_b_prepared.pkl"
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        required = {
            "market_row_number",
            "model_probability_yes",
            "yes_edge",
            "no_edge",
            "yes_spread_cents",
            "no_spread_cents",
            "volume_filled",
            "yes_bid_cents",
            "no_bid_cents",
        }
        if required.issubset(cached.columns):
            return cached

    snapshots = load_snapshots(
        context.db_path,
        series=series,
        columns=[
            "market_ticker",
            "series_ticker",
            "observed_at",
            "expiry",
            "contract_type",
            "underlying_symbol",
            "spot_price",
            "threshold",
            "range_low",
            "range_high",
            "direction",
            "yes_bid",
            "yes_ask",
            "no_bid",
            "no_ask",
            "mid_price",
            "implied_probability",
            "volume",
            "open_interest",
            "source",
            "metadata_json",
        ],
    )
    deduped, _ = deduplicate_snapshots(snapshots)
    deduped, _ = build_global_spot_features(deduped)
    deduped = assign_series_buckets(add_market_derived_columns(deduped))

    settlements = load_settlements(series)[["ticker", "result", "expiration_value"]].copy()
    settlements["actual_yes"] = (settlements["result"].str.lower() == "yes").astype(int)
    frame = deduped.merge(settlements, left_on="market_ticker", right_on="ticker", how="inner")

    drift = float(context.settings.model.get("drift", 0.0))
    vol_floor = float(context.settings.model.get("volatility_floor", 0.05))
    frame["vol_for_model"] = pd.to_numeric(frame["rolling_vol_20"], errors="coerce").fillna(vol_floor)
    frame["model_probability_yes"] = frame.apply(
        lambda row: _gbm_probability_yes(
            spot_price=row["spot_price"],
            threshold=row["threshold"],
            tte_minutes=row["tte_min"],
            volatility=max(float(row["vol_for_model"]), vol_floor),
            drift=drift,
        ),
        axis=1,
    )
    frame["yes_edge"] = frame["model_probability_yes"] - frame["yes_ask"]
    frame["no_edge"] = (1.0 - frame["model_probability_yes"]) - frame["no_ask"]
    frame["yes_spread_cents"] = (frame["yes_ask"] - frame["yes_bid"]) * 100.0
    frame["no_spread_cents"] = (frame["no_ask"] - frame["no_bid"]) * 100.0
    frame["volume_filled"] = frame["volume"].fillna(0.0).astype(float)
    frame["yes_bid_cents"] = np.round(frame["yes_bid"] * 100.0)
    frame["no_bid_cents"] = np.round(frame["no_bid"] * 100.0)
    frame["settlement_yes_payout_cents"] = np.where(frame["actual_yes"] == 1, 100, 0)
    frame["settlement_no_payout_cents"] = np.where(frame["actual_yes"] == 0, 100, 0)
    frame["market_row_number"] = frame.groupby("market_ticker").cumcount()
    frame = frame.sort_values(["market_ticker", "observed_at"]).reset_index(drop=True)
    frame.to_pickle(cache_path)
    return frame


def _gbm_probability_yes(*, spot_price: float, threshold: float, tte_minutes: float, volatility: float, drift: float) -> float:
    years = max(float(tte_minutes), 0.0) / (365.0 * 24.0 * 60.0)
    return float(
        terminal_probability_above(
            spot_price=float(spot_price),
            target_price=float(threshold),
            time_to_expiry_years=years,
            volatility=float(volatility),
            drift=float(drift),
        )
    )


def _build_parameter_grid(series: str, *, focused: bool) -> tuple[list[EntrySpec], list[ExitSpec]]:
    if series == "KXBTCD":
        tte_windows = [(5, 10), (10, 20), (20, 30), (30, 45)]
        dist_thresholds = [200.0, 350.0, 500.0]
    else:
        tte_windows = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 12)]
        dist_thresholds = [100.0, 200.0, SERIES_NEAR_MONEY_BPS[series]]

    if focused:
        min_edges = [0.02, 0.05, 0.08]
        price_windows = [(30, 70), (40, 60)]
        max_spreads = [4, 8]
        min_volumes = [0]
        side_modes = ["yes_only", "no_only", "best_edge"]
        exit_specs = [
            ExitSpec("hold"),
            ExitSpec("stop", stop_loss_cents=10),
            ExitSpec("take_profit", take_profit_cents=8),
            ExitSpec("stop_take_profit", stop_loss_cents=10, take_profit_cents=8),
            ExitSpec("time_exit", time_exit_minutes_before_expiry=2 if series == "KXBTC15M" else 5),
            ExitSpec("fair_value"),
        ]
    else:
        min_edges = [0.02, 0.05, 0.08, 0.12]
        price_windows = [(30, 70), (40, 60), (25, 75)]
        max_spreads = [2, 4, 8]
        min_volumes = [0, 50, 200]
        side_modes = ["yes_only", "no_only", "best_edge"]
        exit_specs = [ExitSpec("hold")]
        exit_specs += [ExitSpec("stop", stop_loss_cents=value) for value in (5, 10, 15, 20)]
        exit_specs += [ExitSpec("take_profit", take_profit_cents=value) for value in (5, 8, 12, 20)]
        exit_specs += [ExitSpec("stop_take_profit", stop_loss_cents=sl, take_profit_cents=tp) for sl, tp in product((5, 10, 15, 20), (5, 8, 12, 20))]
        exit_specs += [ExitSpec("time_exit", time_exit_minutes_before_expiry=value) for value in (2, 5, 10)]
        exit_specs += [ExitSpec("fair_value")]

    entry_specs = [
        EntrySpec(
            tte_min=tte_lo,
            tte_max=tte_hi,
            dist_bps_max=dist_max,
            min_edge=min_edge,
            price_min_cents=price_lo,
            price_max_cents=price_hi,
            max_spread_cents=max_spread,
            min_volume=min_volume,
            side_mode=side_mode,
        )
        for (tte_lo, tte_hi), dist_max, min_edge, (price_lo, price_hi), max_spread, min_volume, side_mode in product(
            tte_windows,
            dist_thresholds,
            min_edges,
            price_windows,
            max_spreads,
            min_volumes,
            side_modes,
        )
    ]
    return entry_specs, exit_specs


def _build_entry_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    common = [
        "market_ticker",
        "observed_at",
        "market_row_number",
        "tte_min",
        "dist_bps",
        "hour_utc",
        "trend_regime",
        "vol_regime",
        "volume_bucket",
        "volume_filled",
        "result",
    ]
    yes = frame[common + ["yes_ask", "yes_bid", "yes_edge", "yes_spread_cents"]].copy()
    yes["entry_side"] = "yes"
    yes["entry_price_cents"] = np.round(yes["yes_ask"] * 100.0)
    yes["entry_edge_actual"] = yes["yes_edge"]
    yes["spread_cents"] = yes["yes_spread_cents"]

    no = frame[common + ["no_ask", "no_bid", "no_edge", "no_spread_cents"]].copy()
    no["entry_side"] = "no"
    no["entry_price_cents"] = np.round(no["no_ask"] * 100.0)
    no["entry_edge_actual"] = no["no_edge"]
    no["spread_cents"] = no["no_spread_cents"]

    candidate_frame = pd.concat([yes, no], ignore_index=True, sort=False)
    return candidate_frame


def _build_market_arrays(frame: pd.DataFrame) -> dict[str, MarketArrays]:
    arrays: dict[str, MarketArrays] = {}
    for market_ticker, group in frame.groupby("market_ticker", sort=False):
        arrays[str(market_ticker)] = MarketArrays(
            observed_at_ns=group["observed_at"].astype("int64").to_numpy(),
            tte_min=group["tte_min"].to_numpy(dtype=float),
            model_probability_yes=group["model_probability_yes"].to_numpy(dtype=float),
            yes_bid_cents=group["yes_bid_cents"].to_numpy(dtype=float),
            no_bid_cents=group["no_bid_cents"].to_numpy(dtype=float),
            settlement_result=str(group["result"].iloc[0]).lower(),
        )
    return arrays


def _select_entries(candidate_frame: pd.DataFrame, spec: EntrySpec) -> pd.DataFrame:
    eligible = candidate_frame[
        (candidate_frame["tte_min"] >= spec.tte_min)
        & (candidate_frame["tte_min"] <= spec.tte_max)
        & (candidate_frame["dist_bps"] < spec.dist_bps_max)
        & (candidate_frame["volume_filled"] >= spec.min_volume)
        & (candidate_frame["entry_edge_actual"] >= spec.min_edge)
        & (candidate_frame["entry_price_cents"] >= spec.price_min_cents)
        & (candidate_frame["entry_price_cents"] <= spec.price_max_cents)
        & (candidate_frame["spread_cents"] <= spec.max_spread_cents)
    ].copy()
    if spec.side_mode != "best_edge":
        side = "yes" if spec.side_mode == "yes_only" else "no"
        eligible = eligible[eligible["entry_side"] == side]
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "entry_index",
                "side",
                "entry_time",
                "entry_price_cents",
                "entry_edge",
                "tte_min",
                "dist_bps",
                "hour_utc",
                "trend_regime",
                "vol_regime",
                "volume_bucket",
                "settlement_result",
            ]
        )

    ordered = eligible.sort_values(["market_ticker", "observed_at", "entry_edge_actual"], ascending=[True, True, False])
    firsts = ordered.drop_duplicates(subset=["market_ticker"], keep="first")
    selected = firsts.rename(
        columns={
            "market_row_number": "entry_index",
            "entry_side": "side",
            "observed_at": "entry_time",
            "entry_edge_actual": "entry_edge",
            "result": "settlement_result",
        }
    )[
        [
            "market_ticker",
            "entry_index",
            "side",
            "entry_time",
            "entry_price_cents",
            "entry_edge",
            "tte_min",
            "dist_bps",
            "hour_utc",
            "trend_regime",
            "vol_regime",
            "volume_bucket",
            "settlement_result",
        ]
    ].copy()
    selected["entry_index"] = selected["entry_index"].astype(int)
    selected["entry_price_cents"] = selected["entry_price_cents"].astype(int)
    selected["hour_utc"] = selected["hour_utc"].astype(int)
    selected["settlement_result"] = selected["settlement_result"].astype(str).str.lower()
    return selected.reset_index(drop=True)


def _simulate_trades_batch(entries: pd.DataFrame, market_arrays: dict[str, MarketArrays], exit_spec: ExitSpec) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame(
            columns=[
                "market_ticker",
                "side",
                "entry_time",
                "entry_price_cents",
                "entry_edge",
                "exit_time",
                "exit_price_cents",
                "exit_trigger",
                "realized_pnl_cents",
                "won",
                "hold_minutes",
                "entry_tte_min",
                "exit_tte_min",
                "entry_dist_bps",
                "entry_hour_utc",
                "trend_regime",
                "vol_regime",
                "volume_bucket",
            ]
        )

    rows: list[dict[str, object]] = []
    for market_ticker, market_entries in entries.groupby("market_ticker", sort=False):
        market = market_arrays[str(market_ticker)]
        for side, side_entries in market_entries.groupby("side", sort=False):
            entry_indices = side_entries["entry_index"].to_numpy(dtype=int)
            entry_prices = side_entries["entry_price_cents"].to_numpy(dtype=float)
            post_starts = entry_indices + 1

            max_len = int(max(len(market.observed_at_ns) - start for start in post_starts)) if len(post_starts) else 0
            if max_len <= 0:
                trade_rows = _finalize_settlement_only(side_entries, market)
                rows.extend(trade_rows)
                continue

            offsets = np.arange(max_len, dtype=int)
            positions = post_starts[:, None] + offsets[None, :]
            valid = positions < len(market.observed_at_ns)

            if side == "yes":
                base_mark = market.yes_bid_cents
                model_prob_side_full = market.model_probability_yes
            else:
                base_mark = market.no_bid_cents
                model_prob_side_full = 1.0 - market.model_probability_yes

            mark = np.full((len(side_entries), max_len), np.nan, dtype=float)
            tte = np.full((len(side_entries), max_len), np.nan, dtype=float)
            model_prob_side = np.full((len(side_entries), max_len), np.nan, dtype=float)
            observed_ns = np.full((len(side_entries), max_len), -1, dtype=np.int64)

            mark[valid] = base_mark[positions[valid]]
            tte[valid] = market.tte_min[positions[valid]]
            model_prob_side[valid] = model_prob_side_full[positions[valid]]
            observed_ns[valid] = market.observed_at_ns[positions[valid]]

            market_prob_side = mark / 100.0
            price_delta = mark - entry_prices[:, None]
            model_edge = model_prob_side - market_prob_side

            hit_mask = np.zeros_like(valid, dtype=bool)
            trigger_names = np.full(len(side_entries), "settlement", dtype=object)

            if exit_spec.mode == "stop" and exit_spec.stop_loss_cents is not None:
                hit_mask = valid & ~np.isnan(mark) & (price_delta <= -exit_spec.stop_loss_cents)
                trigger_names[:] = "stop_loss"
            elif exit_spec.mode == "take_profit" and exit_spec.take_profit_cents is not None:
                hit_mask = valid & ~np.isnan(mark) & (price_delta >= exit_spec.take_profit_cents)
                trigger_names[:] = "take_profit"
            elif exit_spec.mode == "stop_take_profit":
                stop_mask = valid & ~np.isnan(mark) & (price_delta <= -(exit_spec.stop_loss_cents or 0))
                tp_mask = valid & ~np.isnan(mark) & (price_delta >= (exit_spec.take_profit_cents or 0))
                hit_mask = stop_mask | tp_mask
            elif exit_spec.mode == "time_exit" and exit_spec.time_exit_minutes_before_expiry is not None:
                hit_mask = valid & ~np.isnan(mark) & (tte <= exit_spec.time_exit_minutes_before_expiry)
                trigger_names[:] = "time_exit"
            elif exit_spec.mode == "fair_value":
                hit_mask = valid & ~np.isnan(mark) & (price_delta > 0) & (model_edge <= 0)
                trigger_names[:] = "fair_value"

            has_hit = hit_mask.any(axis=1)
            first_hit_col = np.argmax(hit_mask, axis=1)

            for row_idx, entry in enumerate(side_entries.itertuples(index=False)):
                if has_hit[row_idx]:
                    col = int(first_hit_col[row_idx])
                    exit_price_cents = int(mark[row_idx, col])
                    exit_time = pd.to_datetime(int(observed_ns[row_idx, col]), utc=True)
                    exit_tte_min = float(tte[row_idx, col])
                    trigger = trigger_names[row_idx]
                    if exit_spec.mode == "stop_take_profit":
                        trigger = "stop_loss" if price_delta[row_idx, col] <= -(exit_spec.stop_loss_cents or 0) else "take_profit"
                else:
                    exit_price_cents = 100 if ((side == "yes" and str(entry.settlement_result) == "yes") or (side == "no" and str(entry.settlement_result) == "no")) else 0
                    exit_time = pd.to_datetime(int(market.observed_at_ns[-1]), utc=True)
                    exit_tte_min = 0.0
                    trigger = "settlement"

                realized_pnl_cents = exit_price_cents - int(entry.entry_price_cents)
                rows.append(
                    {
                        "market_ticker": entry.market_ticker,
                        "side": side,
                        "entry_time": entry.entry_time,
                        "entry_price_cents": int(entry.entry_price_cents),
                        "entry_edge": float(entry.entry_edge),
                        "exit_time": exit_time,
                        "exit_price_cents": exit_price_cents,
                        "exit_trigger": trigger,
                        "realized_pnl_cents": realized_pnl_cents,
                        "won": int(realized_pnl_cents > 0),
                        "hold_minutes": max((exit_time - pd.Timestamp(entry.entry_time)).total_seconds() / 60.0, 0.0),
                        "entry_tte_min": float(entry.tte_min),
                        "exit_tte_min": exit_tte_min,
                        "entry_dist_bps": float(entry.dist_bps),
                        "entry_hour_utc": int(entry.hour_utc),
                        "trend_regime": entry.trend_regime,
                        "vol_regime": entry.vol_regime,
                        "volume_bucket": entry.volume_bucket,
                    }
                )

    return pd.DataFrame(rows)


def _finalize_settlement_only(entries: pd.DataFrame, market: MarketArrays) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    final_time = pd.to_datetime(int(market.observed_at_ns[-1]), utc=True)
    for entry in entries.itertuples(index=False):
        exit_price_cents = 100 if ((entry.side == "yes" and str(entry.settlement_result) == "yes") or (entry.side == "no" and str(entry.settlement_result) == "no")) else 0
        realized_pnl_cents = exit_price_cents - int(entry.entry_price_cents)
        rows.append(
            {
                "market_ticker": entry.market_ticker,
                "side": entry.side,
                "entry_time": entry.entry_time,
                "entry_price_cents": int(entry.entry_price_cents),
                "entry_edge": float(entry.entry_edge),
                "exit_time": final_time,
                "exit_price_cents": exit_price_cents,
                "exit_trigger": "settlement",
                "realized_pnl_cents": realized_pnl_cents,
                "won": int(realized_pnl_cents > 0),
                "hold_minutes": max((final_time - pd.Timestamp(entry.entry_time)).total_seconds() / 60.0, 0.0),
                "entry_tte_min": float(entry.tte_min),
                "exit_tte_min": 0.0,
                "entry_dist_bps": float(entry.dist_bps),
                "entry_hour_utc": int(entry.hour_utc),
                "trend_regime": entry.trend_regime,
                "vol_regime": entry.vol_regime,
                "volume_bucket": entry.volume_bucket,
            }
        )
    return rows


def _summarize_simulation(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win_cents": 0.0,
            "avg_loss_cents": 0.0,
            "total_pnl_cents": 0.0,
            "sharpe_like": 0.0,
            "max_consecutive_losses": 0,
            "avg_hold_minutes": 0.0,
            "yes_trade_count": 0,
            "no_trade_count": 0,
        }

    pnl = pd.to_numeric(trades["realized_pnl_cents"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    std = float(pnl.std(ddof=0))
    sharpe_like = float(pnl.mean() / std) if std > 0 else float("inf" if pnl.mean() > 0 else 0.0)

    max_consec_losses = 0
    current = 0
    for value in pnl.tolist():
        if value < 0:
            current += 1
            max_consec_losses = max(max_consec_losses, current)
        else:
            current = 0

    return {
        "trade_count": int(len(trades)),
        "win_rate": round(float((pnl > 0).mean() * 100.0), 2),
        "avg_win_cents": round(float(wins.mean()), 4) if not wins.empty else 0.0,
        "avg_loss_cents": round(float(losses.mean()), 4) if not losses.empty else 0.0,
        "total_pnl_cents": round(float(pnl.sum()), 4),
        "sharpe_like": round(sharpe_like, 6) if np.isfinite(sharpe_like) else 999999.0,
        "max_consecutive_losses": int(max_consec_losses),
        "avg_hold_minutes": round(float(pd.to_numeric(trades["hold_minutes"], errors="coerce").mean()), 4),
        "yes_trade_count": int((trades["side"] == "yes").sum()),
        "no_trade_count": int((trades["side"] == "no").sum()),
    }
