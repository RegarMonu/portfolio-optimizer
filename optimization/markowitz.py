"""
Phase 4 — Classical Mean-Variance (Markowitz) Portfolio Optimization.

Implemented directly with `scipy.optimize.minimize` (SLSQP) rather than a
dedicated library like PyPortfolioOpt or cvxpy — this keeps the dependency
footprint identical to what Phase 2 already required (scipy was already in
requirements.txt) and makes the constrained-optimization mechanics fully
transparent and testable in this codebase, rather than opaque inside a
third-party solver.

Constraints used throughout (can be relaxed later if short-selling is
wanted): weights sum to 1, weights >= 0 (long-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return weights @ cov_matrix @ weights


def _portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    return weights @ mean_returns


def _base_constraints(n_assets: int) -> tuple[dict, tuple]:
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))  # long-only
    return constraints, bounds


def min_variance_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """The Minimum Variance Portfolio: lowest possible risk, ignoring expected return entirely."""
    tickers = cov_matrix.columns.tolist()
    n = len(tickers)
    cov = cov_matrix.values

    constraints, bounds = _base_constraints(n)
    x0 = np.repeat(1 / n, n)

    result = minimize(
        _portfolio_variance, x0, args=(cov,), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Min-variance optimization failed to converge: {result.message}")

    return pd.Series(result.x, index=tickers, name="weight").clip(lower=0)


def max_sharpe_weights(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float = 0.065
) -> pd.Series:
    """
    The Maximum Sharpe (Tangency) Portfolio: best risk-adjusted return.
    Maximizing Sharpe = minimizing its negative (scipy only minimizes).
    """
    tickers = mean_returns.index.tolist()
    n = len(tickers)
    mu = mean_returns.values
    cov = cov_matrix.loc[tickers, tickers].values

    def _neg_sharpe(w):
        ret = _portfolio_return(w, mu)
        vol = np.sqrt(_portfolio_variance(w, cov))
        if vol == 0:
            return 1e6
        return -(ret - risk_free_rate) / vol

    constraints, bounds = _base_constraints(n)
    x0 = np.repeat(1 / n, n)

    result = minimize(
        _neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed to converge: {result.message}")

    return pd.Series(result.x, index=tickers, name="weight").clip(lower=0)


def target_return_min_variance_weights(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, target_return: float
) -> pd.Series:
    """
    Minimum-variance portfolio subject to achieving AT LEAST `target_return`.
    Used to trace out the efficient frontier point-by-point.
    """
    tickers = mean_returns.index.tolist()
    n = len(tickers)
    mu = mean_returns.values
    cov = cov_matrix.loc[tickers, tickers].values

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: _portfolio_return(w, mu) - target_return},
    )
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.repeat(1 / n, n)

    result = minimize(
        _portfolio_variance, x0, args=(cov,), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return None  # target return infeasible at this risk level -- caller should skip this point

    return pd.Series(result.x, index=tickers, name="weight").clip(lower=0)


def efficient_frontier(
    mean_returns: pd.Series, cov_matrix: pd.DataFrame, n_points: int = 25
) -> pd.DataFrame:
    """
    Traces the efficient frontier by solving target-return-constrained
    min-variance problems across the achievable return range.
    Returns a DataFrame: target_return, volatility, and one weight column per asset.
    """
    min_ret, max_ret = mean_returns.min(), mean_returns.max()
    targets = np.linspace(min_ret, max_ret, n_points)

    rows = []
    for target in targets:
        weights = target_return_min_variance_weights(mean_returns, cov_matrix, target)
        if weights is None:
            continue
        vol = np.sqrt(_portfolio_variance(weights.values, cov_matrix.loc[weights.index, weights.index].values))
        row = {"target_return": target, "volatility": vol}
        row.update(weights.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)
