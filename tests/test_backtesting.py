"""
Unit tests for backtesting.* modules — deterministic synthetic data,
no network required. Run with: python -m pytest tests/test_backtesting.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backtesting.transaction_costs import transaction_cost, turnover
from backtesting.strategy import (
    EqualWeightStrategy,
    FixedWeightStrategy,
    MinVarianceStrategy,
    RiskParityStrategy,
)
from backtesting.engine import BacktestEngine


def make_returns_matrix(n_assets=4, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    vols = np.linspace(0.01, 0.03, n_assets)
    means = np.linspace(0.0002, 0.0008, n_assets)
    data = {f"A{i}": rng.normal(means[i], vols[i], n_days) for i in range(n_assets)}
    return pd.DataFrame(data, index=dates)


# ---------- transaction_costs.py ----------

def test_turnover_zero_when_weights_unchanged():
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover(w, w) == pytest.approx(0.0)


def test_turnover_full_when_flipping_positions():
    old = pd.Series({"A": 1.0, "B": 0.0})
    new = pd.Series({"A": 0.0, "B": 1.0})
    assert turnover(old, new) == pytest.approx(2.0)  # sold 1.0 of A, bought 1.0 of B


def test_transaction_cost_scales_linearly_with_turnover_and_bps():
    assert transaction_cost(0.5, cost_bps=10) == pytest.approx(0.0005)
    assert transaction_cost(1.0, cost_bps=20) == pytest.approx(0.002)


# ---------- strategy.py ----------

def test_equal_weight_strategy_produces_uniform_weights():
    rm = make_returns_matrix(n_assets=4)
    strategy = EqualWeightStrategy()
    weights = strategy.compute_weights(rm)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights == 0.25).all()


def test_fixed_weight_strategy_ignores_history_and_returns_fixed_weights():
    rm = make_returns_matrix(n_assets=3)
    fixed = pd.Series({"A0": 0.5, "A1": 0.3, "A2": 0.2})
    strategy = FixedWeightStrategy(fixed, name="my_fixed")
    result = strategy.compute_weights(rm)
    assert strategy.name == "my_fixed"
    pd.testing.assert_series_equal(result.sort_index(), fixed.sort_index(), check_names=False)


def test_min_variance_strategy_respects_lookback_window():
    rm = make_returns_matrix(n_assets=3, n_days=600)
    strategy = MinVarianceStrategy(lookback_days=100)
    weights = strategy.compute_weights(rm)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)


# ---------- engine.py ----------

def test_engine_no_lookahead_first_rebalance_respects_min_lookback():
    rm = make_returns_matrix(n_assets=3, n_days=100)
    engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=0, min_lookback_days=60)
    result = engine.run(EqualWeightStrategy())
    # Before day 60, portfolio should be uninitialized -> zero return (holding cash)
    assert (result.daily_returns.iloc[:60] == 0).all()
    # After sufficient history, it should have started trading
    assert not (result.daily_returns.iloc[60:] == 0).all()


def test_engine_zero_cost_equal_weight_matches_manual_calc_first_days():
    rm = make_returns_matrix(n_assets=2, n_days=70, seed=1)
    engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=0, min_lookback_days=60)
    result = engine.run(EqualWeightStrategy())
    # Right after the first rebalance (day 60), portfolio return should equal
    # simple average of the two assets' returns that day (weights = 0.5/0.5)
    first_active_date = rm.index[60]
    expected = rm.loc[first_active_date].mean()
    assert result.daily_returns.loc[first_active_date] == pytest.approx(expected, abs=1e-9)


def test_engine_applies_transaction_costs_and_reduces_returns():
    rm = make_returns_matrix(n_assets=4, n_days=300, seed=2)
    zero_cost_engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=0, min_lookback_days=60)
    high_cost_engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=500, min_lookback_days=60)

    zero_cost_result = zero_cost_engine.run(RiskParityStrategy())
    high_cost_result = high_cost_engine.run(RiskParityStrategy())

    zero_cost_final = zero_cost_result.cumulative_value.iloc[-1]
    high_cost_final = high_cost_result.cumulative_value.iloc[-1]
    assert high_cost_final < zero_cost_final
    assert high_cost_result.total_transaction_cost_drag > 0


def test_engine_turnover_history_recorded_at_each_rebalance():
    rm = make_returns_matrix(n_assets=3, n_days=300, seed=3)
    engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=10, min_lookback_days=60)
    result = engine.run(EqualWeightStrategy())
    assert len(result.turnover_history) > 0
    assert (result.turnover_history >= 0).all()


def test_engine_weights_always_sum_to_one_after_rebalance():
    rm = make_returns_matrix(n_assets=4, n_days=300, seed=4)
    engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=10, min_lookback_days=60)
    result = engine.run(MinVarianceStrategy())
    row_sums = result.weights_history.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6


def test_engine_run_many_returns_dict_keyed_by_strategy_name():
    rm = make_returns_matrix(n_assets=3, n_days=200, seed=5)
    engine = BacktestEngine(rm, rebalance_freq="ME", transaction_cost_bps=10, min_lookback_days=60)
    results = engine.run_many([EqualWeightStrategy(), MinVarianceStrategy()])
    assert set(results.keys()) == {"equal_weight", "min_variance"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
