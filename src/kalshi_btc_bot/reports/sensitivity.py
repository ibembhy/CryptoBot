from __future__ import annotations

import pandas as pd

from kalshi_btc_bot.backtest.engine import BacktestEngine
from kalshi_btc_bot.backtest.scenarios import clone_with_volatility_multiplier


def run_volatility_sensitivity(
    engine: BacktestEngine,
    snapshots,
    feature_frame: pd.DataFrame,
    multipliers: list[float],
) -> pd.DataFrame:
    rows = []
    for multiplier in multipliers:
        scenario = clone_with_volatility_multiplier(feature_frame, multiplier)
        result = engine.run_strategy("early_exit", snapshots, scenario)
        rows.append(
            {
                "volatility_multiplier": multiplier,
                "pnl": result.summary["pnl"],
                "roi": result.summary["roi"],
                "trade_count": result.summary["trade_count"],
            }
        )
    return pd.DataFrame(rows)
