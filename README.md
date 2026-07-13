# Indian Portfolio Optimization Platform

A complete, production-grade quantitative research platform for NIFTY 50
equities — from raw data ingestion through classical portfolio
optimization, a real backtesting engine with transaction costs, and a
Deep RL portfolio manager. **All 6 phases of the roadmap are built and
tested (69/69 tests passing).**

**See `PROGRESS.md` for exactly what's built, every design decision made,
and honest limitations to know about before using this with real money.**

## Structure
```
config/               # settings, logging — single source of truth for constants
data_layer/            # Phase 1: fetch -> clean -> store pipeline
eda/                   # Phase 2: returns, volatility, drawdown, correlation,
                       #          regime detection, distribution stats, sector
                       #          analysis, and the report generator
feature_engineering/   # Phase 3: trend/momentum/volatility/volume/market
                       #          features -> model-ready feature panel
optimization/          # Phase 4: equal weight, Markowitz (min-var/max-Sharpe/
                       #          efficient frontier), risk parity, evaluation
                       #          metrics, simple static-weight simulator
backtesting/           # Phase 5: configurable engine — pluggable strategies,
                       #          real transaction costs, no-look-ahead by
                       #          construction
rl/                    # Phase 6: Gymnasium env, PPO/SAC training, evaluation
                       #          against classical baselines
tests/                 # unit tests (no network required) — 69/69 passing
data_store/            # raw/, processed/, and features/ parquet files
models/                # saved RL agent checkpoints
reports/               # generated reports & figures for every phase
logs/                  # pipeline.log
```

## Quickstart
```bash
pip install -r requirements.txt --break-system-packages

# Phase 1: pull real data
python -m data_layer.pipeline --tickers RELIANCE.NS,TCS.NS --start 2022-01-01

# Phase 2: EDA report (real data, or --synthetic for a demo)
python -m eda.report
python -m eda.report --synthetic

# Phase 3: build the feature panel
python -m feature_engineering.pipeline
python -m feature_engineering.pipeline --synthetic

# Phase 4: classical optimization comparison (no transaction costs)
python -m optimization.report
python -m optimization.report --synthetic

# Phase 5: full backtest with real transaction costs
python -m backtesting.report
python -m backtesting.report --synthetic

# Phase 6: train and evaluate a Deep RL agent
python -m rl.train --synthetic --timesteps 100000
python -m rl.evaluate --synthetic

# Run the full test suite
python -m pytest tests/ -v
```

## Before using this with real money

Read the "Honest limitations" section at the bottom of `PROGRESS.md` —
in particular, this build has never been run against live market data
(only synthetic demo data, due to the sandbox it was built in), and
Phase 6's RL agent is trained at demonstration scale, not research scale.
