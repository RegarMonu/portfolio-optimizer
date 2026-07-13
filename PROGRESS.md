# Build Progress — Indian Portfolio Optimization Platform

**Project status: COMPLETE.** All 6 phases of the roadmap are built and
tested. This file is kept as the historical build log and design-decision
record — useful if you come back to extend, debug, or re-derive "why is
it built this way" for any part of the codebase.

---

## ✅ Phase 0 — Domain Knowledge
Understand the financial concepts required before implementing any optimization algorithm.

### Topics to Learn 
* Indian Stock Market (NSE, BSE, NIFTY 50, Sensex)
* Stocks, ETFs, Mutual Funds
* Corporate Actions (Dividends, Splits, Bonus Issues)
* Portfolio Theory
* Diversification
* Risk vs Return
* Correlation & Covariance
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown
* Portfolio Volatility 

### Deliverable 
* Personal notes explaining all concepts.
* Mathematical understanding of portfolio risk and return.
* Ability to explain why portfolio optimization is needed.

## ✅ Phase 1 — Data Layer — **DONE**
Location: `config/`, `data_layer/`

Fetches NIFTY 50 + benchmark + India VIX from yfinance with retry/backoff
and batching; cleans via calendar reindexing with bounded forward-fill
(gaps ≤5 days) or ticker drop (longer gaps / >30% missing, always logged,
never silent); flags (never deletes) extreme price moves; stores as
Parquet with a JSON manifest audit trail. `auto_adjust=True` handles
corporate actions. NIFTY 50 list + sector map manually curated as of Jan
2026 — needs periodic refresh, not automated.
**5/5 tests passing.**

## ✅ Phase 2 — Exploratory Data Analysis — **DONE**
Location: `eda/`

Returns (daily/weekly/monthly/CAGR), rolling volatility + vol regimes,
drawdown series + episode detection + Calmar, correlation/covariance
matrices, bull/bear regime detection (±20% peak/trough convention),
distribution stats (skew, kurtosis, VaR/CVaR, Jarque-Bera — flags where
Markowitz's normality assumption is shaky), sector-wise performance.
`eda/report.py` generates a full markdown + chart report; `--synthetic`
mode allows verification without live market data.
**No dedicated test file for eda itself** (covered indirectly via
`test_features.py` and `test_optimization.py`, which both depend on
`eda.returns` / `eda.drawdown` / `eda.correlation` working correctly).

## ✅ Phase 3 — Feature Engineering — **DONE**
Location: `feature_engineering/`

Trend (SMA/EMA/price-to-SMA ratio/crossover signal), momentum (RSI, MACD,
ROC), volatility (ATR + re-exported rolling vol), volume (change/SMA/
relative volume), market-wide (NIFTY return, India VIX, sector relative
strength). `pipeline.py` builds a 22-column long-format feature panel
(MultiIndex ticker/date). Documented feature-by-feature in `FEATURES.md`.
Explicitly tested for no look-ahead leakage.
**13/13 tests passing.**

## ✅ Phase 4 — Classical Portfolio Optimization — **DONE**
Location: `optimization/`

Equal Weight, Min Variance, Max Sharpe, Risk Parity, and a tested
Efficient Frontier function — all implemented directly with
`scipy.optimize` (SLSQP), no PyPortfolioOpt/cvxpy dependency. Sharpe/
Sortino metrics; CAGR/max-drawdown/Calmar reused as-is from Phase 2.
Comparison report vs. Equal Weight and Buy & Hold. Risk parity's sanity
check confirmed all assets converge to *exactly* equal risk contribution.
**15/15 tests passing.**

## ✅ Phase 5 — Backtesting Engine — **DONE**
Location: `backtesting/`

Generalizes Phase 4's minimal simulator into a real, reusable engine:
* `Strategy` ABC — any Phase 4 method (or later, an RL policy) plugs in
  identically via one `compute_weights(returns_history)` method.
* `BacktestEngine` — configurable rebalance frequency, REAL transaction
  costs (`transaction_costs.py`, basis-points-of-turnover model), full
  portfolio tracking (daily returns, weights history, turnover, costs,
  rebalance failures — nothing thrown away).
* No-look-ahead is enforced BY THE ENGINE (`returns_matrix.iloc[:i]`,
  strictly before the rebalance date), not left to each strategy to get
  right independently.
* `report.py` compares all 4 classical methods + Buy & Hold + the NIFTY 50
  index itself, net of transaction costs.

**Design decisions:**
- Cost model: basis points of turnover (default 10 bps ≈ brokerage + STT +
  slippage estimate for liquid NIFTY 50 names). Swappable in
  `transaction_costs.py` if you have better real-world cost data.
- Before the first rebalance (insufficient lookback history), the
  portfolio holds cash (zero return) rather than trading on too little data.
- On a rebalance failure (optimizer non-convergence), the engine logs a
  warning, holds previous weights, and keeps going — a single bad day
  never crashes the whole backtest.

**Verified end-to-end:** `python -m backtesting.report --synthetic` — full
comparison table + equity curve + turnover charts. Confirmed empirically
that higher-turnover strategies (max_sharpe) pay materially more
transaction cost drag than lower-turnover ones (risk_parity, equal_weight).
**12/12 tests passing.**

## ✅ Phase 6 — Deep RL Portfolio Optimization — **DONE**
Location: `rl/`

* `env.py` — Gymnasium-compatible `PortfolioAllocationEnv`. State = rolling
  window of historical returns + current weights (incl. cash) + optional
  market features. Action = raw logits over assets+cash, softmax-normalized
  inside the env into a valid long-only simplex (so a plain continuous Box
  action space always produces valid portfolio weights). Reward =
  portfolio return − transaction cost (reuses Phase 5's cost model) −
  asymmetric downside-risk penalty (Sortino-flavored: only bad days
  penalized).
* `train.py` — trains PPO (primary, per roadmap) or SAC (`--algo sac`, the
  roadmap's "future comparison" algorithm) via stable-baselines3. Both
  work off-the-shelf against the same continuous action space.
* `baselines.py` — reuses Phase 5's `BacktestEngine` for the classical
  comparison set, so the RL-vs-classical comparison is apples-to-apples
  (same date range, same transaction cost assumption), not against a
  separately-coded simulator.
* `evaluate.py` — runs the trained policy deterministically over the full
  evaluation period, computes the roadmap's required metrics (cumulative
  return, Sharpe, max drawdown, annual return, portfolio stability via
  weight-turnover, transaction cost impact), and produces a comparison
  report against every classical baseline + the NIFTY 50 index.

**Design decisions:**
- Softmax-over-logits action parameterization was chosen specifically so
  the agent can't produce invalid (negative or non-summing-to-1) weights —
  this is a common, well-tested trick in portfolio RL rather than a novel
  choice, and it means no action-clipping/rejection logic was needed.
- `train.py`'s default 20,000 timesteps and short 120-day episodes are
  **demonstration-scale**, chosen to make the whole pipeline runnable and
  verifiable quickly. `rl_report.md`'s "Honest Assessment" section says
  this explicitly: a real research run needs far more timesteps,
  hyperparameter search, and critically, a **time-based train/test split**
  (this build's `--synthetic` smoke test evaluates over the same
  distribution it trained on, which is fine for verifying the MDP wiring,
  but is NOT how you'd validate real out-of-sample performance).
- `evaluate.py` explicitly reports portfolio stability (weight turnover
  over time) alongside performance, since the roadmap calls it out as an
  evaluation criterion distinct from raw returns.

**Verified end-to-end:** trained a PPO agent for 3,000 timesteps on 5
synthetic tickers (`python -m rl.train --synthetic --n-synthetic-tickers 5
--timesteps 3000 --episode-length 60`), then evaluated it against all 4
classical baselines + NIFTY 50 (`python -m rl.evaluate --synthetic
--n-synthetic-tickers 5`). The under-trained agent landed mid-pack on
Sharpe ratio, below min_variance/risk_parity/equal_weight but above
max_sharpe and the raw index — a sane result for a smoke test, not a
claim of RL superiority (see the report's honest-assessment section).
**7/7 environment tests passing** (training itself isn't unit-tested
directly — training is stochastic and slow; the environment's MDP
mechanics are what's tested, plus the smoke-test run above confirms the
full train→save→load→evaluate pipeline actually works).

---

## Full test suite: 69/69 passing across all 6 phases

```
tests/test_cleaner.py        5 passed   (Phase 1)
tests/test_eda.py           17 passed   (Phase 2, and used by Phase 3/4/5/6)
tests/test_features.py      13 passed   (Phase 3)
tests/test_optimization.py  15 passed   (Phase 4)
tests/test_backtesting.py   12 passed   (Phase 5)
tests/test_rl_env.py         7 passed   (Phase 6)
```

## How to run the whole project

```bash
pip install -r requirements.txt --break-system-packages

# Phase 1: fetch + clean + store real NIFTY 50 + benchmark + VIX data
python -m data_layer.pipeline

# Phase 2: EDA report
python -m eda.report                    # real data
python -m eda.report --synthetic        # demo, no network needed

# Phase 3: feature panel
python -m feature_engineering.pipeline
python -m feature_engineering.pipeline --synthetic

# Phase 4: classical optimization comparison (no transaction costs)
python -m optimization.report
python -m optimization.report --synthetic

# Phase 5: full backtest with transaction costs (the "real" comparison)
python -m backtesting.report
python -m backtesting.report --synthetic

# Phase 6: train and evaluate an RL agent
python -m rl.train --synthetic --timesteps 100000          # scale up for real use
python -m rl.evaluate --synthetic

# Run everything (no network needed) — 69/69 passing
python -m pytest tests/ -v
```

## Honest limitations of this build (read before using with real money)

1. **Never run against live Yahoo Finance data in this build environment**
   — the sandbox this was built in has a domain-restricted network. Every
   phase was verified with the `--synthetic` correlated-GBM demo data
   generator instead. Everything downstream of
   `data_layer.storage.load_universe()` is identical whether the data came
   from real or synthetic frames, so Phase 1 running for real on your
   machine should make every later phase work unchanged — but this hasn't
   been directly confirmed against real NSE data.
2. **Phase 6's RL training is demonstration-scale** (thousands, not
   hundreds of thousands, of timesteps). Treat the RL results as "the
   pipeline works end-to-end," not "RL beats classical optimization."
3. **NIFTY 50 constituent list and sector map are manually curated**
   (Jan 2026 snapshot) and will drift out of date as the index rebalances.
4. **No fundamental (P/E, earnings) data anywhere** — this is a pure
   price/volume platform.
5. **Transaction cost model is a simple bps-of-turnover estimate**, not a
   real broker's fee schedule, slippage model, or market-impact model.
