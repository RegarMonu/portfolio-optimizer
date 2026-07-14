"""
Phase 5 — Strategy interface.

Every strategy implements ONE method: given a window of historical daily
returns (up to but NOT including the rebalance date — enforced by the
engine, not here), produce target portfolio weights. This is the seam
that lets `backtesting.engine.BacktestEngine` run any of Phase 4's
classical methods, and later a Phase 6 RL policy, through identical code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "unnamed_strategy"

    @abstractmethod
    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        """
        returns_history: DataFrame of daily returns, index=dates, columns=tickers,
            containing ONLY data strictly before the current rebalance date.
        Returns: target weight Series, index=tickers, summing to 1.
        """
        raise NotImplementedError


class EqualWeightStrategy(Strategy):
    name = "equal_weight"

    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        from optimization.equal_weight import equal_weights
        return equal_weights(returns_history.columns.tolist())


class MinVarianceStrategy(Strategy):
    name = "min_variance"

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days

    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        from optimization.markowitz import min_variance_weights
        window = returns_history.tail(self.lookback_days)
        cov = window.cov() * 252
        return min_variance_weights(cov)


class MaxSharpeStrategy(Strategy):
    name = "max_sharpe"

    def __init__(self, lookback_days: int = 252, risk_free_rate: float = 0.065):
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        from optimization.markowitz import max_sharpe_weights
        window = returns_history.tail(self.lookback_days)
        mean_returns = window.mean() * 252
        cov = window.cov() * 252
        return max_sharpe_weights(mean_returns, cov, self.risk_free_rate)


class RiskParityStrategy(Strategy):
    name = "risk_parity"

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days

    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        from optimization.risk_parity import risk_parity_weights
        window = returns_history.tail(self.lookback_days)
        cov = window.cov() * 252
        return risk_parity_weights(cov)


class FixedWeightStrategy(Strategy):
    """
    Wraps a pre-computed, unchanging weight vector as a Strategy — used for
    Buy & Hold (compute once, never re-optimize) and for dropping in a
    Phase 6 RL policy's output as a one-off comparison point.
    """
    name = "fixed_weight"

    def __init__(self, weights: pd.Series, name: str | None = None):
        self.weights = weights
        if name:
            self.name = name

    def compute_weights(self, returns_history: pd.DataFrame) -> pd.Series:
        return self.weights.reindex(returns_history.columns).fillna(0)
