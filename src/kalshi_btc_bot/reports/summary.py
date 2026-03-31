from __future__ import annotations

import pandas as pd

from kalshi_btc_bot.reports.calibration import build_calibration_report
from kalshi_btc_bot.types import BacktestResult


def build_summary_report(result: BacktestResult) -> dict:
    report = dict(result.summary)
    trades = result.trades
    report["edge_distribution"] = (
        trades["edge"].describe().to_dict() if isinstance(trades, pd.DataFrame) and not trades.empty else {}
    )
    report["raw_edge_distribution"] = (
        trades["raw_edge"].describe().to_dict() if isinstance(trades, pd.DataFrame) and not trades.empty and "raw_edge" in trades else {}
    )
    report["calibration"] = build_calibration_report(trades) if isinstance(trades, pd.DataFrame) else {}
    return report
