# Kalshi BTC Bot

This repository contains a live research and paper-trading system for short-horizon Kalshi Bitcoin threshold markets.

The project is no longer just an MVP GBM bot. It now includes:
- live Kalshi snapshot collection into SQLite
- a Streamlit dashboard for live monitoring
- replay-ready market enrichment and settlement tracking
- a hybrid signal stack built around `gbm_threshold`, `latency_repricing`, and `repricing_target`
- regime-aware calibration
- paper trading with position/risk controls
- replay diagnostics, failure analysis, model comparison, grid search, and maker-side research tools
- replay and calibration caches so repeated research runs are much faster

## Current Live Setup

The default live configuration in [config/default.toml](C:/Users/cbemb/Documents/CryptoBot/config/default.toml) is intentionally selective:
- `series_ticker = "KXBTCD"`
- `model = "hybrid"`
- price band: `25-60c`
- near-money filter: `150 bps`
- max minutes to expiry: `15`
- max spread: `6c`
- calibration: enabled

This setup reflects the current best replay regime we found so far. It is designed to favor higher-quality short-expiry trades rather than constant activity.

## What The System Does

At a high level:
1. Collect open Kalshi BTC market snapshots and BTC spot data.
2. Store normalized snapshots in SQLite for both live monitoring and replay.
3. Generate live trading signals from a hybrid model stack.
4. Rank signals using edge, quality, calibration, and repricing support.
5. Execute paper trades only when risk, quality, and market filters are satisfied.
6. Track fills, positions, closed trades, and session PnL in a ledger.
7. Replay the saved history offline to compare models, stress test rules, and diagnose failures.

## Main Documents

- High-level system guide: [docs/SYSTEM_GUIDE.md](C:/Users/cbemb/Documents/CryptoBot/docs/SYSTEM_GUIDE.md)
- Runtime config: [config/default.toml](C:/Users/cbemb/Documents/CryptoBot/config/default.toml)

If you need to explain the full system to someone else, start with the system guide.

## Install

Core dev install:

```bash
py -m pip install -e .[dev]
```

For the Streamlit dashboard:

```bash
py -m pip install -e .[ui]
```

## Useful Commands

Show resolved config:

```bash
py -m kalshi_btc_bot.cli show-config
```

Collect one batch of live snapshots:

```bash
py -m kalshi_btc_bot.cli collect-once
```

Run the continuous collector:

```bash
py -m kalshi_btc_bot.cli collect-forever --log-level INFO
```

Run the continuous paper trader:

```bash
py -m kalshi_btc_bot.cli paper-trade-forever --series KXBTCD --max-minutes-to-expiry 15
```

See dataset health:

```bash
py -m kalshi_btc_bot.cli dataset-summary --series KXBTCD
```

Replay a backtest from stored history:

```bash
py -m kalshi_btc_bot.cli replay-backtest --series KXBTCD
```

Compare models on a replay slice:

```bash
py -m kalshi_btc_bot.cli replay-compare-models --series KXBTCD
```

Diagnose which buckets are helping or hurting:

```bash
py -m kalshi_btc_bot.cli replay-failure-analysis --series KXBTCD
```

Run conservative maker-side research:

```bash
py -m kalshi_btc_bot.cli replay-maker-sim --series KXBTCD
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Live Deployment Notes

The system is designed to run in the cloud. In the current deployment:
- the collector runs as its own service
- the paper trader runs as its own service
- the dashboard runs as its own service

That means the system keeps running even if your laptop is off.

## Known Tradeoffs

- The current live setup is intentionally narrow, so the dashboard can look quiet between trading windows.
- The system is still in the paper-trading proof stage, not the “proven live money machine” stage.
- Maker-side research exists, but realistic passive-fill simulation is still limited by data granularity and execution realism.

## Status

This repo now has enough infrastructure to:
- collect and replay real market state
- paper trade a selective live regime
- inspect why signals are skipped or taken
- compare directional and maker-style research branches

The best current use of the project is:
- run the live paper trader
- monitor the dashboard
- use replay analysis to improve edge without changing too many things at once
