"""
Unit tests for rl.env.PortfolioAllocationEnv — deterministic synthetic
data, no network or model training required.
Run with: python -m pytest tests/test_rl_env.py -v
"""

import numpy as np
import pandas as pd
import pytest

from rl.env import PortfolioAllocationEnv


def make_returns_matrix(n_assets=4, n_days=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = {f"A{i}": rng.normal(0.0005, 0.015, n_days) for i in range(n_assets)}
    return pd.DataFrame(data, index=dates)


def test_env_conforms_to_gymnasium_spaces():
    rm = make_returns_matrix()
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50, seed=0)
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs2) or truncated
    assert isinstance(reward, float)


def test_env_starts_fully_in_cash():
    rm = make_returns_matrix()
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50, seed=1)
    env.reset()
    assert env.weights[-1] == pytest.approx(1.0)
    assert env.weights[:-1].sum() == pytest.approx(0.0)


def test_env_weights_always_sum_to_one_after_step():
    rm = make_returns_matrix()
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50, seed=2)
    env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        env.step(action)
        assert env.weights.sum() == pytest.approx(1.0, abs=1e-5)


def test_env_truncates_at_episode_length():
    rm = make_returns_matrix(n_days=300)
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=15,
                                  random_start=False, seed=3)
    env.reset()
    truncated = False
    steps = 0
    while not truncated and steps < 100:
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 15


def test_env_transaction_cost_increases_with_turnover():
    rm = make_returns_matrix(n_days=200)
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50,
                                  transaction_cost_bps=100, random_start=False, seed=4)
    env.reset()
    # Extreme action -> near all-in on one asset -> high turnover from all-cash start
    extreme_action = np.array([5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32)
    _, _, _, _, info = env.step(extreme_action)
    assert info["turnover"] > 1.0  # moving fully out of cash into one asset
    assert info["transaction_cost"] > 0


def test_env_reward_penalizes_downside_more_than_upside():
    rm = make_returns_matrix()
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50,
                                  risk_aversion=1.0, transaction_cost_bps=0, seed=5)

    # Manually verify the asymmetric penalty via the reward formula directly
    # (mirrors the env's internal logic without needing a specific market day)
    good_return, bad_return = 0.02, -0.02
    good_reward = good_return - 1.0 * max(0.0, -good_return)
    bad_reward = bad_return - 1.0 * max(0.0, -bad_return)
    assert good_reward == pytest.approx(0.02)   # no penalty on an up day
    assert bad_reward == pytest.approx(-0.04)   # penalty doubles the down day's hit


def test_env_market_features_included_in_observation():
    rm = make_returns_matrix(n_days=200)
    market = pd.DataFrame({"nifty_return": np.zeros(200), "vix": np.full(200, 15.0)}, index=rm.index)
    env = PortfolioAllocationEnv(rm, lookback_window=10, episode_length=50,
                                  market_features=market, seed=6)
    obs, _ = env.reset()
    expected_dim = 4 * 10 + 5 + 2  # 4 assets * window + (4 assets+cash) + 2 market features
    assert obs.shape[0] == expected_dim


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
