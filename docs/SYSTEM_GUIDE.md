# Kalshi BTC Bot System Guide

This document explains the current system in detail so you can describe what it does, how it is built, how it is operated, and where the project stands today.

## 1. Purpose

The bot is a live research and paper-trading system for short-horizon Kalshi Bitcoin threshold contracts, mainly `KXBTCD`.

Its job is not just to forecast whether BTC will settle above or below a threshold. The system is designed to:
- collect live market state
- normalize that data into replay-ready history
- generate ranked trade candidates
- paper trade the strongest setups under risk controls
- compare alternative models and rules offline
- diagnose where edge appears to exist and where it breaks

The project is best thought of as a combined:
- live collector
- live paper trader
- replay research lab
- dashboard and monitoring tool

## 2. Core Idea

The bot tries to find situations where Kalshi BTC contracts are priced differently from what the model stack believes is fair, but it does this under strict quality filters.

The current philosophy is:
- trade only near expiry
- trade only in a tighter price band
- avoid wide-spread conditions
- use a hybrid signal rather than a single raw model
- calibrate predictions before trusting them
- rank candidates instead of treating all valid trades equally

This is why the live setup now trades far less often than early versions. The goal is better trade quality, not constant activity.

## 3. Main Components

### 3.1 Collector

The collector discovers relevant Kalshi BTC markets, attaches live BTC spot context, and writes normalized market snapshots to SQLite.

Responsibilities:
- discover open BTC threshold markets
- fetch live market state
- attach contemporaneous BTC spot
- write normalized snapshots to `data/kalshi_btc_snapshots.sqlite3`
- reconcile settled and expired market state over time

Current default collector settings:
- series: `KXBTCD`
- status: `open`
- minutes to expiry: `0-15`
- reconcile interval: `60s`
- spot refresh interval: `10s`

Important tradeoff:
- because the collector is currently limited to the same `<=15m` regime as the trader, the dashboard can look empty between windows even though Kalshi still has open markets further from expiry

### 3.2 Paper Trader

The paper trader evaluates eligible live markets and simulates trades using the current default strategy and risk controls.

Responsibilities:
- build live features from recent snapshots
- compute signals and quality scores
- decide whether to enter
- manage exits
- record fills
- maintain paper positions
- persist a paper ledger to `data/paper_trading_state.json`

Important recent fix:
- expired positions are no longer dependent on the current collector batch to be closed
- positions now carry their own expiry
- the paper trader can fall back to the latest stored snapshot when closing expired positions

This was a real bug fix and matters for paper-ledger correctness.

### 3.3 Dashboard

The Streamlit dashboard is the live monitoring surface.

It currently shows:
- snapshot and market counts
- settled and replay-ready counts
- the active default setup
- current paper-trading state
- recent closed positions and fills
- live signal decisions
- signal-reason summaries
- latest market snapshots

Recent dashboard improvements:
- timestamps are displayed in readable Eastern time
- signal decisions show whether a row was a candidate or skipped
- decision context is visible through reasons, regimes, spread, stale-quote flags, and other quality fields
- a dashboard-side reconciliation step can reconstruct missing closed positions from raw fills if the ledger overwrote repeated trades on the same market

## 4. Data Flow

The live and research path looks like this:

1. Kalshi market data and BTC spot data are collected.
2. Snapshots are normalized into a shared internal format.
3. Snapshots are stored in SQLite.
4. The paper trader builds current features from recent history.
5. The model stack produces signals, fair values, and ranking context.
6. Risk checks and market-quality filters gate execution.
7. Paper fills and positions are written to the ledger.
8. Offline replay tools reuse the same stored snapshots to test models and rules.

This shared data path is one of the project’s biggest strengths: the same historical store supports both dashboarding and replay.

## 5. Model Stack

The bot no longer relies on a single model.

### 5.1 `gbm_threshold`

This is the original directional model.

Idea:
- estimate the probability that BTC will finish above or below a threshold by expiry using a GBM-style threshold probability

Strength:
- simple
- stable
- still the strongest standalone directional baseline in our replay work

### 5.2 `latency_repricing`

This model adds a short-horizon repricing view on top of terminal-outcome thinking.

Idea:
- account for short-latency repricing rather than only final settlement

Role:
- confirmatory model inside the hybrid stack

### 5.3 `repricing_target`

This is the more advanced repricing research model.

Idea:
- learn a short-horizon Kalshi repricing target rather than only final settlement probability

What it now includes:
- a cached repricing profile
- regime-level repricing statistics
- profit-oriented labels
- bounded probability shifts
- microstructure inputs

Important status:
- it is useful as a ranking/support model
- it has not yet beaten the directional baseline as the main live driver

## 6. Hybrid Strategy

The current live default strategy is `hybrid`.

Current config:
- primary model: `gbm_threshold`
- confirm model: `latency_repricing`
- ranking support model: `repricing_target`
- require side agreement: `true`
- allow unconfirmed primary trades: `false`

What this means in practice:
- the primary directional model still carries the decision
- the confirm model must agree on side
- the repricing model helps ranking and quality scoring, not final leadership

This is an important design decision. The project tried to promote repricing into a lead role, but replay did not justify that yet.

## 7. Features

The bot started with simple return and volatility features, but now includes a more serious short-horizon feature set.

### 7.1 BTC features

Examples:
- log returns
- realized volatility over `1m`, `3m`, and `5m`
- vol-of-vol
- BTC acceleration
- micro-jump flags

### 7.2 Kalshi microstructure features

Examples:
- local momentum
- reversion gap
- spread regime
- liquidity regime
- quote age
- stale-quote flag

These were added because repricing logic was too blind without microstructure context.

## 8. Calibration

Calibration is now a real part of the system rather than a side experiment.

### 8.1 What calibration does

It maps raw model outputs toward more trustworthy probabilities or edge estimates.

### 8.2 Current calibration behavior

Calibration is enabled by default.

Current settings:
- bucket width: `0.05`
- min samples: `50`
- min bucket count: `3`

### 8.3 Regime-aware calibration

Calibration is no longer only global. The system now supports calibration by slices such as:
- near-money vs not
- `0-15m` vs `15-30m`
- lower-vol vs higher-vol
- tighter-spread vs wider-spread

This matters because global calibration was too blunt and could collapse tradeable predictions.

### 8.4 Calibration cache

Repeated research runs now use a calibration cache:
- `data/calibration_cache.json`

That speeds up iteration substantially on repeated replay analysis.

## 9. Replay Research Capabilities

The replay path is now much more mature than the original MVP.

Main commands include:
- `replay-backtest`
- `replay-compare-models`
- `replay-failure-analysis`
- `replay-grid-search`
- `replay-maker-proxy`
- `replay-maker-sim`
- `rebuild-repricing-profile`

### 9.1 Replay cache

The feature-frame build is cached to disk:
- `data/replay_cache/`

This means repeated work on the same slice is much faster than rebuilding the feature frame every time.

### 9.2 Failure analysis

This is one of the most important additions.

It lets us break down performance by buckets such as:
- price bands
- time to expiry
- side
- exit type
- edge buckets

This is how we found some of the best improvements so far.

## 10. Current Best Live Regime

The current live defaults reflect the best regime found so far in replay:
- price band: `25-60c`
- near money: `150 bps`
- max minutes to expiry: `15`
- max spread: `6c`

Why this regime:
- replay improved materially after cutting longer-expiry and higher-price trades
- this looks more like a true regime improvement than a random tuning win

Important nuance:
- this is the current best regime, not proof of a final durable edge

## 11. Paper Trading Rules

The paper trader currently uses:
- `default_contracts = 1`
- `take_profit_cents = 8`
- `stop_loss_cents = 10`
- `fair_value_buffer_cents = 3`
- `time_exit_minutes_before_expiry = 5`
- `min_hold_edge = 0.02`
- `min_ev_to_hold_cents = 2`

Risk controls include:
- max trade notional
- max session notional
- max open positions
- max positions per expiry
- max daily loss
- max drawdown

The current stage of the project is still proof-of-edge, so fixed small sizing remains appropriate.

## 12. Recent Improvements That Matter

The system has changed meaningfully from the early version.

### 12.1 Strategy improvements

- moved from a simple single-model setup to a hybrid signal stack
- narrowed live trading into a stronger replay regime
- added ranking support from repricing logic
- added regime-aware calibration
- improved spread and quality filtering

### 12.2 Research improvements

- replay cache
- calibration cache
- failure analysis tooling
- model comparison tooling
- maker-side research tools

### 12.3 Dashboard improvements

- readable ET timestamps
- better signal decision visibility
- skip/take reason summaries
- reconciliation of closed trades from raw fills

### 12.4 Paper-trader correctness improvements

- expired positions can now close correctly even after their market rolls out of the current collector window

## 13. Maker-Side Research

The project also explored whether the real edge may be more structural than predictive.

That means testing whether money is made more reliably by:
- acting like a maker
- assuming better entry mechanics
- capturing spread or stale quotes
- providing better liquidity than takers

Two tools were added:
- `replay-maker-proxy`
- `replay-maker-sim`

### 13.1 What we learned

The optimistic maker proxy looked promising.

But once we added more realistic assumptions such as:
- partial fills
- fill probability
- stale-quote cancellation
- liquidity constraints
- inventory limits

the edge mostly disappeared in the conservative simulation.

This suggests a major current bottleneck:
- data granularity and execution realism

In plain English:
- maker-side may still be a real direction
- but the current stored data is not rich enough to prove it cleanly yet

## 14. Live Deployment Model

The cloud deployment is split into independent services.

Current roles:
- collector service
- paper-trader service
- dashboard service

Why this matters:
- the dashboard can be restarted without resetting the paper session
- collector and paper trader can continue running even if the user’s laptop is off
- the dashboard can be viewed from a phone browser because it is served from the cloud host

## 15. Known Limitations

The project is much stronger than the original MVP, but there are still important limitations.

### 15.1 Edge is not fully proven

The system has shown promising paper behavior, but not enough live evidence exists yet to call it proven for real money.

### 15.2 Activity is intentionally low

Because the current setup is narrow, the system can look quiet for stretches of time.

This is expected when:
- there are open markets
- but none are yet inside the `<=15m` active regime

### 15.3 Maker-side realism is limited

Passive-fill simulation still suffers from limited data detail.

### 15.4 Current collector window affects dashboard feel

Because the collector is also limited to `<=15m`, the dashboard can show no recent snapshots even while longer-dated open markets still exist.

## 16. What The System Is Good For Right Now

Right now the system is good for:
- live paper trading in a selective BTC regime
- replaying real stored market history
- comparing models on the same data
- finding which filters help or hurt
- monitoring why signals are skipped or taken

It is not yet good for claiming:
- fully proven real-money profitability
- production-grade maker-side execution
- fully mature capital scaling

## 17. Recommended Explanation To Others

If you need to explain the project quickly, this is the clean version:

The bot is a cloud-run Kalshi BTC research and paper-trading system. It collects live Bitcoin threshold-market snapshots, stores them for replay, scores near-expiry contracts with a hybrid model stack, calibrates and ranks those signals, paper trades the strongest setups under risk limits, and exposes the whole process through a dashboard and replay diagnostics. The current best regime is a selective short-expiry window, and the project is now in the stage of proving whether that edge is durable enough to justify real capital later.

## 18. Important Files

- Root config: [config/default.toml](C:/Users/cbemb/Documents/CryptoBot/config/default.toml)
- CLI entrypoint: [src/kalshi_btc_bot/cli.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/cli.py)
- Dashboard app: [app.py](C:/Users/cbemb/Documents/CryptoBot/app.py)
- Dashboard logic: [src/kalshi_btc_bot/dashboard.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/dashboard.py)
- Paper trader: [src/kalshi_btc_bot/trading/live_paper.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/trading/live_paper.py)
- Position book: [src/kalshi_btc_bot/trading/positions.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/trading/positions.py)
- Replay engine: [src/kalshi_btc_bot/backtest/engine.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/backtest/engine.py)
- Repricing model: [src/kalshi_btc_bot/models/repricing_target.py](C:/Users/cbemb/Documents/CryptoBot/src/kalshi_btc_bot/models/repricing_target.py)

## 19. Bottom Line

The project is no longer a simple bot with one model and a paper loop.

It is now a fuller trading research system with:
- live collection
- live paper execution
- replay infrastructure
- calibration
- diagnostics
- regime-specific filtering
- maker-side experimentation

The biggest unresolved question is no longer “can the system run?” It can.

The real open question now is:
- whether the current selective edge is strong and durable enough to justify scaling into real money later.
