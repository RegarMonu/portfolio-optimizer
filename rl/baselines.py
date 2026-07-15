"""
Phase 6 — Baseline comparisons.

Rather than reimplementing Equal Weight / Markowitz / Risk Parity a THIRD
time, this module runs them through Phase 5's `BacktestEngine` over
exactly the same date range the RL agent is evaluated on, using the same
transaction cost assumption — so the RL-vs-classical comparison in
`rl/evaluate.py` is apples-to-apples, not apples-to-a-different-simulator.
"""

from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestEngine
from backtesting.strategy import (
    EqualWeightStrategy,
    MaxSharpeStrategy,
    MinVarianceStrategy,
    RiskParityStrategy,
)


def run_baselines(
    returns_matrix: pd.DataFrame,
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = 10.0,
    min_lookback_days: int = 60,
) -> dict:
    """Returns {strategy_name: BacktestResult} for every classical Phase 4 method."""
    engine = BacktestEngine(
        returns_matrix, rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps, min_lookback_days=min_lookback_days,
    )
    strategies = [
        EqualWeightStrategy(),
        MinVarianceStrategy(),
        MaxSharpeStrategy(),
        RiskParityStrategy(),
    ]
    return engine.run_many(strategies)
