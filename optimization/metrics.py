"""
Phase 4 — Portfolio evaluation metrics.

Sharpe and Sortino live here because they're specifically PORTFOLIO-level
evaluation metrics (need a risk-free rate assumption); CAGR, max drawdown,
and Calmar already exist from Phase 2 (`eda.returns`, `eda.drawdown`) and
are reused as-is — no need to re-derive return/drawdown math that Phase 2
already tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eda.drawdown import calmar_ratio, max_drawdown
from eda.returns import cagr


def sharpe_ratio(
    returns: pd.Series, risk_free_rate_annual: float = 0.065, periods_per_year: int = 252
) -> float:
    """
    Annualized Sharpe ratio. Default risk-free rate 6.5% approximates the
    Indian 10-year G-Sec yield as a reasonable long-run default — override
    with the actual prevailing rate for precise work.
    """
    excess_daily_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = returns - excess_daily_rf
    if excess_returns.std() == 0:
        return float("nan")
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series, risk_free_rate_annual: float = 0.065, periods_per_year: int = 252
) -> float:
    """
    Like Sharpe, but only penalizes DOWNSIDE volatility (returns below the
    risk-free rate), not all volatility. More appropriate when returns are
    skewed — which Phase 2's distribution analysis showed is common here.
    """
    excess_daily_rf = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = returns - excess_daily_rf
    downside = excess_returns[excess_returns < 0]
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return float("nan")
    return (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)


def performance_summary(
    portfolio_returns: pd.Series, risk_free_rate_annual: float = 0.065
) -> dict:
    """One consolidated dict of every metric required by the Phase 4 roadmap."""
    cumulative_value = (1 + portfolio_returns).cumprod()
    return {
        "cagr": cagr(cumulative_value),
        "annual_return": portfolio_returns.mean() * 252,
        "annual_volatility": portfolio_returns.std() * np.sqrt(252),
        "sharpe_ratio": sharpe_ratio(portfolio_returns, risk_free_rate_annual),
        "sortino_ratio": sortino_ratio(portfolio_returns, risk_free_rate_annual),
        "max_drawdown": max_drawdown(cumulative_value),
        "calmar_ratio": calmar_ratio(cumulative_value),
    }
