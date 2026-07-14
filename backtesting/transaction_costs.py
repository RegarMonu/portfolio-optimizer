"""
Phase 5 — Transaction cost modeling.

Kept as its own tiny module (rather than inlined in the engine) so the
cost assumption is a single, obvious, swappable place — and independently
testable without spinning up a full backtest.
"""

from __future__ import annotations

import pandas as pd


def turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """
    Portfolio turnover = sum of absolute weight changes across all assets.
    A turnover of 0.30 means 30% of the portfolio's value was traded.
    """
    aligned_old = old_weights.reindex(new_weights.index).fillna(0)
    return float((new_weights - aligned_old).abs().sum())


def transaction_cost(turnover_amount: float, cost_bps: float = 10.0) -> float:
    """
    Cost as a fraction of portfolio value, given turnover and a cost rate
    in basis points (bps) of value traded. Default 10 bps (0.10%) is a
    reasonable round-trip estimate for liquid NIFTY 50 names including
    brokerage + STT + slippage — override for a more precise cost model.
    """
    return turnover_amount * (cost_bps / 10_000)
