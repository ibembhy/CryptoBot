from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kalshi_btc_bot.backtest.metrics import compute_max_drawdown


@dataclass(frozen=True)
class BankrollSizingConfig:
    starting_bankroll: float
    min_cash_buffer: float = 0.0
    bankroll_fraction_per_trade: float = 1.0
    max_contracts_per_trade: int = 1
    allow_fractional_contracts: bool = False


def simulate_bankroll_constrained_compounding(
    trades: pd.DataFrame,
    *,
    config: BankrollSizingConfig,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if trades.empty:
        empty = trades.copy()
        return empty, {
            "trade_count": 0,
            "taken_trade_count": 0,
            "skipped_trade_count": 0,
            "starting_bankroll": round(float(config.starting_bankroll), 2),
            "ending_bankroll": round(float(config.starting_bankroll), 2),
            "pnl": 0.0,
            "return_pct": 0.0,
            "max_drawdown": 0.0,
            "avg_contracts_per_taken_trade": 0.0,
            "max_contracts_used": 0.0,
        }

    trades = trades.sort_values("entry_time").reset_index(drop=True).copy()
    simulated_rows: list[dict] = []
    bankroll = float(config.starting_bankroll)
    max_contracts_used = 0.0

    for _, row in trades.iterrows():
        requested_contracts = float(row.get("contracts", 1) or 1)
        entry_notional = float(row.get("entry_notional", 0.0) or 0.0)
        realized_pnl = float(row.get("realized_pnl", 0.0) or 0.0)
        if requested_contracts <= 0 or entry_notional <= 0:
            continue

        per_contract_notional = entry_notional / requested_contracts
        per_contract_pnl = realized_pnl / requested_contracts
        available_bankroll = max(bankroll - float(config.min_cash_buffer), 0.0)
        sizing_budget = max(available_bankroll * float(config.bankroll_fraction_per_trade), 0.0)
        affordable_contracts = sizing_budget / per_contract_notional if per_contract_notional > 0 else 0.0
        contracts = min(float(config.max_contracts_per_trade), affordable_contracts)
        if not config.allow_fractional_contracts:
            contracts = float(int(contracts))

        taken = contracts > 0
        scaled_entry_notional = round(per_contract_notional * contracts, 4)
        scaled_realized_pnl = round(per_contract_pnl * contracts, 4)
        bankroll_after = round(bankroll + scaled_realized_pnl, 4) if taken else round(bankroll, 4)
        max_contracts_used = max(max_contracts_used, contracts)

        simulated_rows.append(
            {
                **row.to_dict(),
                "bankroll_before": round(bankroll, 4),
                "bankroll_after": bankroll_after,
                "requested_contracts": requested_contracts,
                "simulated_contracts": contracts,
                "simulated_entry_notional": scaled_entry_notional,
                "simulated_realized_pnl": scaled_realized_pnl,
                "simulated_taken": taken,
                "simulated_skipped_reason": "" if taken else "Insufficient bankroll for minimum trade size.",
            }
        )
        bankroll = bankroll_after

    simulated = pd.DataFrame(simulated_rows)
    taken = simulated.loc[simulated["simulated_taken"]].copy()
    equity_curve = simulated["bankroll_after"] if not simulated.empty else pd.Series(dtype=float)
    summary = {
        "trade_count": int(len(simulated)),
        "taken_trade_count": int(len(taken)),
        "skipped_trade_count": int(len(simulated) - len(taken)),
        "starting_bankroll": round(float(config.starting_bankroll), 2),
        "ending_bankroll": round(float(bankroll), 2),
        "pnl": round(float(bankroll - config.starting_bankroll), 2),
        "return_pct": round(((float(bankroll) - float(config.starting_bankroll)) / float(config.starting_bankroll)) * 100.0, 2)
        if config.starting_bankroll
        else 0.0,
        "max_drawdown": compute_max_drawdown(equity_curve),
        "avg_contracts_per_taken_trade": round(float(taken["simulated_contracts"].mean()), 4) if not taken.empty else 0.0,
        "max_contracts_used": round(float(max_contracts_used), 4),
    }
    return simulated, summary
