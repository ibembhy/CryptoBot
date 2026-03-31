from __future__ import annotations

import pandas as pd


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return round(abs(float(drawdown.min())), 2)


def summarize_trades(trades: pd.DataFrame, *, starting_bankroll: float) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trade_count": 0,
            "roi": 0.0,
            "win_rate": 0.0,
            "average_edge": 0.0,
            "average_raw_edge": 0.0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "average_hold_minutes": 0.0,
            "trade_frequency": 0.0,
            "exit_improved_rate": 0.0,
            "average_clv_cents": 0.0,
            "entry_vs_exit_attribution": 0.0,
            "regret_exited_too_early": 0.0,
            "regret_held_too_long": 0.0,
        }
    pnl = float(trades["realized_pnl"].sum())
    stakes = float(trades["entry_notional"].sum())
    equity = starting_bankroll + trades["realized_pnl"].cumsum()
    win_rate = float((trades["realized_pnl"] > 0).mean() * 100.0)
    frequency = float(len(trades) / max(trades["entry_time"].nunique(), 1))
    return {
        "trade_count": int(len(trades)),
        "roi": round((pnl / stakes) * 100.0, 2) if stakes else 0.0,
        "win_rate": round(win_rate, 2),
        "average_edge": round(float(trades["edge"].mean()), 4),
        "average_raw_edge": round(float(trades["raw_edge"].mean()), 4) if "raw_edge" in trades else 0.0,
        "pnl": round(pnl, 2),
        "max_drawdown": compute_max_drawdown(equity),
        "average_hold_minutes": round(float(trades["hold_minutes"].mean()), 2),
        "trade_frequency": round(frequency, 4),
        "exit_improved_rate": round(float(trades["exit_improved_result"].mean() * 100.0), 2),
        "average_clv_cents": round(float(trades["clv_cents"].mean()), 2),
        "entry_vs_exit_attribution": round(float((trades["exit_price_cents"] - trades["entry_price_cents"]).mean()), 4),
        "regret_exited_too_early": round(float((trades["hold_to_settlement_pnl"] - trades["realized_pnl"]).clip(lower=0).mean()), 4),
        "regret_held_too_long": round(float((trades["realized_pnl"] - trades["hold_to_settlement_pnl"]).clip(lower=0).mean()), 4),
    }
