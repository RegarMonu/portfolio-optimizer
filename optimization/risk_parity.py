"""
Phase 4 — Risk Parity Portfolio.

Goal: every asset contributes EQUAL risk to total portfolio volatility
(as opposed to equal weight, where a high-vol asset dominates portfolio
risk despite having the same capital allocation as a low-vol asset).

Risk contribution of asset i = w_i * (Cov @ w)_i / portfolio_vol
We minimize the sum of squared deviations between each asset's risk
contribution and the equal-risk target (portfolio_vol / n).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights
    return weights * marginal_contrib / portfolio_vol


def risk_parity_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    tickers = cov_matrix.columns.tolist()
    n = len(tickers)
    cov = cov_matrix.values
    target_risk_share = np.repeat(1 / n, n)

    def _objective(w):
        contributions = _risk_contributions(w, cov)
        total_vol = np.sqrt(w @ cov @ w)
        actual_share = contributions / total_vol
        return np.sum((actual_share - target_risk_share) ** 2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = tuple((1e-6, 1.0) for _ in range(n))  # small epsilon lower bound avoids div-by-zero
    x0 = np.repeat(1 / n, n)

    result = minimize(
        _objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-16},
    )
    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed to converge: {result.message}")

    weights = pd.Series(result.x, index=tickers, name="weight")
    return weights / weights.sum()  # renormalize for numerical exactness


def verify_risk_parity(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Sanity-check helper: returns each asset's actual % share of total
    portfolio risk. In a perfect risk-parity solution, every entry equals
    1/n_assets — use this to confirm an optimization result is genuinely
    balanced, not just numerically converged.
    """
    w = weights.reindex(cov_matrix.columns).values
    contributions = _risk_contributions(w, cov_matrix.values)
    total_vol = np.sqrt(w @ cov_matrix.values @ w)
    return pd.Series(contributions / total_vol, index=cov_matrix.columns, name="risk_share")
