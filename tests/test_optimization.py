"""
Unit tests for optimization.* modules — deterministic synthetic data,
no network required. Run with: python -m pytest tests/test_optimization.py -v
"""

import numpy as np
import pandas as pd
import pytest

from optimization.equal_weight import equal_weights
from optimization.markowitz import (
    efficient_frontier,
    max_sharpe_weights,
    min_variance_weights,
    target_return_min_variance_weights,
)
from optimization.risk_parity import risk_parity_weights, verify_risk_parity
from optimization.metrics import performance_summary, sharpe_ratio, sortino_ratio
from optimization.simulate import simulate_portfolio


def make_returns_matrix(n_assets=4, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    # Assets with meaningfully different vols so risk parity / min-var are non-trivial
    vols = np.linspace(0.01, 0.03, n_assets)
    means = np.linspace(0.0002, 0.0008, n_assets)
    data = {
        f"A{i}": rng.normal(means[i], vols[i], n_days) for i in range(n_assets)
    }
    return pd.DataFrame(data, index=dates)


def make_cov(returns_matrix):
    return returns_matrix.cov() * 252


# ---------- equal_weight.py ----------

def test_equal_weights_sum_to_one_and_uniform():
    weights = equal_weights(["A", "B", "C", "D"])
    assert weights.sum() == pytest.approx(1.0)
    assert (weights == 0.25).all()


# ---------- markowitz.py ----------

def test_min_variance_weights_sum_to_one_and_nonnegative():
    rm = make_returns_matrix()
    cov = make_cov(rm)
    weights = min_variance_weights(cov)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()


def test_min_variance_favors_low_vol_asset():
    rm = make_returns_matrix(n_assets=2, seed=1)
    rm["LOW_VOL"] = np.random.default_rng(2).normal(0.0003, 0.005, len(rm))
    rm["HIGH_VOL"] = np.random.default_rng(3).normal(0.0003, 0.04, len(rm))
    cov = make_cov(rm[["LOW_VOL", "HIGH_VOL"]])
    weights = min_variance_weights(cov)
    assert weights["LOW_VOL"] > weights["HIGH_VOL"]


def test_max_sharpe_weights_sum_to_one():
    rm = make_returns_matrix()
    mean_returns = rm.mean() * 252
    cov = make_cov(rm)
    weights = max_sharpe_weights(mean_returns, cov)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()


def test_max_sharpe_favors_high_return_low_vol_asset():
    rng = np.random.default_rng(4)
    n = 300
    good = rng.normal(0.0015, 0.01, n)   # high return, moderate vol
    bad = rng.normal(0.0001, 0.02, n)    # low return, higher vol
    rm = pd.DataFrame({"GOOD": good, "BAD": bad})
    mean_returns = rm.mean() * 252
    cov = make_cov(rm)
    weights = max_sharpe_weights(mean_returns, cov)
    assert weights["GOOD"] > weights["BAD"]


def test_efficient_frontier_produces_increasing_risk_for_increasing_return():
    rm = make_returns_matrix(n_assets=4, seed=5)
    mean_returns = rm.mean() * 252
    cov = make_cov(rm)
    frontier = efficient_frontier(mean_returns, cov, n_points=10)
    assert len(frontier) > 0
    # Volatility should be (weakly) non-decreasing as target return increases
    assert frontier["volatility"].is_monotonic_increasing or \
        (frontier["volatility"].diff().dropna() >= -1e-6).all()


def test_target_return_weights_achieve_at_least_target():
    rm = make_returns_matrix(n_assets=3, seed=6)
    mean_returns = rm.mean() * 252
    cov = make_cov(rm)
    mid_target = mean_returns.mean()
    weights = target_return_min_variance_weights(mean_returns, cov, mid_target)
    assert weights is not None
    achieved = (weights.reindex(mean_returns.index) * mean_returns).sum()
    assert achieved >= mid_target - 1e-6


# ---------- risk_parity.py ----------

def test_risk_parity_weights_sum_to_one():
    rm = make_returns_matrix()
    cov = make_cov(rm)
    weights = risk_parity_weights(cov)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_risk_parity_actually_equalizes_risk_contribution():
    rm = make_returns_matrix(n_assets=4, seed=7)
    cov = make_cov(rm)
    weights = risk_parity_weights(cov)
    risk_shares = verify_risk_parity(weights, cov)
    # All risk shares should be close to 1/n_assets
    target = 1 / len(risk_shares)
    assert (risk_shares - target).abs().max() < 0.02


def test_risk_parity_gives_less_weight_to_high_vol_asset_than_equal_weight():
    rng = np.random.default_rng(8)
    n = 400
    low_vol = rng.normal(0.0003, 0.008, n)
    high_vol = rng.normal(0.0003, 0.035, n)
    rm = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol})
    cov = make_cov(rm)
    weights = risk_parity_weights(cov)
    assert weights["LOW"] > weights["HIGH"]
    assert weights["HIGH"] < 0.5  # less than equal-weight's 0.5


# ---------- metrics.py ----------

def test_sharpe_ratio_higher_for_better_risk_adjusted_returns():
    rng = np.random.default_rng(9)
    good_returns = pd.Series(rng.normal(0.0015, 0.01, 500))
    bad_returns = pd.Series(rng.normal(0.0002, 0.02, 500))
    assert sharpe_ratio(good_returns) > sharpe_ratio(bad_returns)


def test_sortino_only_penalizes_downside():
    # Same mean/std overall, but one has its variance concentrated on the upside
    rng = np.random.default_rng(10)
    symmetric = pd.Series(rng.normal(0.0005, 0.01, 500))
    upside_skewed = symmetric.copy()
    upside_skewed[upside_skewed > 0] *= 2  # amplify only the positive days
    assert sortino_ratio(upside_skewed) >= sortino_ratio(symmetric) - 0.5


def test_performance_summary_has_all_required_keys():
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0004, 0.012, 500))
    summary = performance_summary(returns)
    required = {"cagr", "annual_return", "annual_volatility", "sharpe_ratio",
                "sortino_ratio", "max_drawdown", "calmar_ratio"}
    assert required.issubset(summary.keys())
    assert summary["max_drawdown"] <= 0


# ---------- simulate.py ----------

def test_simulate_portfolio_equal_weight_matches_manual_calc():
    dates = pd.bdate_range("2024-01-01", periods=3)
    rm = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [-0.10, 0.0, 0.0]}, index=dates)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    port_rets = simulate_portfolio(rm, weights, rebalance_freq="never")
    # Day 1: 0.5*0.10 + 0.5*(-0.10) = 0 -> portfolio value stays at 1.0 -> pct_change ~ 0
    assert port_rets.iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_simulate_portfolio_weights_reindexed_and_normalized():
    dates = pd.bdate_range("2024-01-01", periods=5)
    rm = pd.DataFrame({"A": np.repeat(0.01, 5), "B": np.repeat(0.02, 5)}, index=dates)
    weights = pd.Series({"A": 2.0, "B": 2.0})  # unnormalized, should be renormalized to 0.5/0.5
    port_rets = simulate_portfolio(rm, weights, rebalance_freq="ME")
    assert not port_rets.isna().all()
    assert port_rets.iloc[0] == pytest.approx(0.5 * 0.01 + 0.5 * 0.02)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
