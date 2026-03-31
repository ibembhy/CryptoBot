# Kalshi BTC Probability Bot MVP

This project is a clean-room Bitcoin prediction-market trading bot focused on Kalshi crypto markets. The MVP uses Coinbase BTC-USD price data, a simple GBM/lognormal threshold model, normalized Kalshi market snapshots, signal generation, paper-trading abstractions, and a backtester that compares hold-to-settlement against early-exit behavior.

## How It Works
- Ingest BTC candles from Coinbase.
- Resample and compute log returns plus rolling realized volatility.
- Normalize Kalshi crypto contracts into a shared `MarketSnapshot`.
- Estimate `P(price_T >= threshold)` with a simple GBM terminal-price model.
- Compare model probability against market-implied probability from tradable prices.
- Emit `buy_yes`, `buy_no`, or `no_action`.
- Backtest both hold and early-exit strategies with slippage and fees.

## Project Layout
```text
config/default.toml
src/kalshi_btc_bot/
tests/
```

## Configuration
Runtime thresholds live in [`config/default.toml`](C:/Users/cbemb/Documents/CryptoBot/config/default.toml).

You can override config values with environment variables like:
- `KALSHI_BTC_SIGNAL_MIN_EDGE`
- `KALSHI_BTC_TRADING_TAKE_PROFIT_CENTS`
- `KALSHI_BTC_BACKTEST_ENTRY_SLIPPAGE_CENTS`

You can also point to a custom TOML file with `KALSHI_BTC_CONFIG_PATH`.

## Install
```bash
py -m pip install -e .[dev]
```

For the Streamlit dashboard:
```bash
py -m pip install -e .[ui]
```

## CLI
Show resolved config:
```bash
py -m kalshi_btc_bot.cli show-config
```

Fetch spot:
```bash
py -m kalshi_btc_bot.cli spot
```

Run a lightweight live-data connectivity check:
```bash
py -m kalshi_btc_bot.cli demo-backtest --hours 24
```

Bootstrap and persist one batch of Kalshi BTC snapshots:
```bash
py -m kalshi_btc_bot.cli collect-once
```

This writes normalized snapshots to SQLite at the path configured by `collector.sqlite_path`.

Run the live hybrid collector:
```bash
py -m kalshi_btc_bot.cli collect-forever --log-level INFO
```

Backfill historical Kalshi candlestick data into the same SQLite store:
```bash
py -m kalshi_btc_bot.cli backfill-candles --series KXBTCD --start-ts 1774800000 --end-ts 1774886400
```

For settled-market research, use the historical discovery path so the backfill targets settled BTC events and their archived markets:
```bash
py -m kalshi_btc_bot.cli --log-level INFO backfill-candles --series KXBTCD --historical --start-ts 1774886400 --end-ts 1774915200 --max-markets 5
```

Replay a backtest from stored SQLite history:
```bash
py -m kalshi_btc_bot.cli replay-backtest --series KXBTCD
```

See what dataset you currently have:
```bash
py -m kalshi_btc_bot.cli dataset-summary --series KXBTCD
```

## Streamlit Dashboard
Launch the live research dashboard:
```bash
streamlit run app.py
```

The dashboard shows:
- dataset health and replay-ready counts
- latest BTC Kalshi market snapshots
- current candidate signals using the bot's default strategy settings
- the active default trading filters

For an overnight one-click launcher that starts both the collector and the dashboard in separate PowerShell windows:
```powershell
.\start-night.ps1
```

## Hybrid Collector
- `REST` is used for market discovery, cold start, and periodic reconciliation.
- `WebSocket` is intended for live `ticker` updates after the relevant BTC markets are known.
- The collector stores every normalized snapshot with contemporaneous BTC spot so later backtests can replay actual market evolution.
- Kalshi requires authenticated WebSocket handshakes. Configure these env vars to enable signed socket access:
  - `KALSHI_BTC_KALSHI_API_KEY_ID`
  - `KALSHI_BTC_KALSHI_PRIVATE_KEY_PATH`
  - `KALSHI_BTC_KALSHI_WEBSOCKET_PATH`
- If those auth values are not configured, the collector falls back to REST-only reconciliation.

## Pre-Phase-2 Workflow
1. Run `collect-forever` to accumulate Kalshi BTC market history.
2. Use `backfill-candles` to bootstrap older history from Kalshi candlestick endpoints.
3. Use `replay-backtest` to evaluate hold-to-settlement vs early-exit behavior on stored history.
4. Once enough history is available, inspect calibration, conservative edge, and smarter quality controls.

## Backtesting Notes
- Volatility is computed only from completed bars available at each decision timestamp.
- Resampling uses right-labeled, right-closed bars to avoid future leakage.
- Entry decisions never see settlement or future path information.
- Early exits use only the observed post-entry snapshot path.

## What To Tune First
- `data.resample_timeframe`
- `data.volatility_window`
- `signal.min_edge`
- `trading.take_profit_cents`
- `trading.stop_loss_cents`
- `trading.time_exit_minutes_before_expiry`

## Next Improvements
- Persist historical Kalshi snapshots for richer replay quality.
- Add authenticated live Kalshi order routing behind the paper broker interface.
- Support more exchanges and assets via the same adapter pattern.
- Add richer market-quality filters and capital-allocation rules.
