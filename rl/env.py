"""
Phase 6 — Gymnasium-compatible RL environment for portfolio allocation.

MDP design (per the project roadmap):
    State  : a rolling window of historical daily returns for every asset,
             the agent's CURRENT portfolio weights (including a cash slot),
             and optional market indicators (e.g. benchmark return, VIX).
    Action : a raw logit vector over (assets + cash), softmax-normalized
             inside the environment into a valid long-only simplex — this
             is what makes a plain continuous Box action space produce
             valid portfolio weights without the agent needing to learn
             the simplex constraint itself.
    Reward : portfolio return, minus a transaction-cost penalty (reusing
             Phase 5's cost model so training and backtesting agree on what
             a trade costs), minus a downside-risk penalty (Sortino-flavored:
             only bad days are penalized, not volatility in general).

Episode structure: each episode is a contiguous slice of the returns
history of length `episode_length`. By default the start point is
randomized on `reset()` (standard practice — improves sample diversity
during training); pass `random_start=False` for deterministic evaluation
episodes that always start right after the lookback window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioAllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_matrix: pd.DataFrame,
        lookback_window: int = 30,
        episode_length: int | None = None,
        transaction_cost_bps: float = 10.0,
        risk_aversion: float = 0.5,
        market_features: pd.DataFrame | None = None,
        random_start: bool = True,
        seed: int | None = None,
    ):
        """
        returns_matrix: wide DataFrame, index=dates, columns=tickers, daily returns.
        lookback_window: how many past days of returns form the state.
        episode_length: trading days per episode. Defaults to using nearly
            the whole series (good for evaluation); set smaller for
            training so many diverse episodes fit in a training run.
        market_features: optional extra DataFrame (same date index) of
            broadcastable market context, e.g. columns=['nifty_return', 'vix_level'].
        random_start: True for training (sample diverse start points),
            False for evaluation (always walk the same, full period).
        """
        super().__init__()
        self.returns_matrix = returns_matrix
        self.tickers = returns_matrix.columns.tolist()
        self.n_assets = len(self.tickers)
        self.lookback_window = lookback_window
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_aversion = risk_aversion
        self.market_features = market_features
        self.n_market_features = 0 if market_features is None else market_features.shape[1]
        self.random_start = random_start
        self.episode_length = episode_length or (len(returns_matrix) - lookback_window - 1)

        # Action: raw logits over [assets..., cash] -> softmax-normalized in step()
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        obs_dim = (
            self.n_assets * self.lookback_window   # flattened return window
            + (self.n_assets + 1)                  # current weights incl. cash
            + self.n_market_features
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._np_random = np.random.default_rng(seed)
        self.current_idx = 0
        self.current_step = 0
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        window = self.returns_matrix.iloc[
            self.current_idx - self.lookback_window : self.current_idx
        ].to_numpy().flatten()
        market = (
            np.array([], dtype=np.float32) if self.market_features is None
            else self.market_features.iloc[self.current_idx].to_numpy()
        )
        return np.concatenate([window, self.weights, market]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        min_start = self.lookback_window
        max_start = max(min_start, len(self.returns_matrix) - self.episode_length - 1)

        self.current_idx = (
            int(self._np_random.integers(min_start, max_start))
            if (self.random_start and max_start > min_start) else min_start
        )
        self.current_step = 0
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # start fully in cash

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        exp_a = np.exp(action - np.max(action))  # numerically stable softmax
        target_weights = (exp_a / exp_a.sum()).astype(np.float32)

        turnover = float(np.abs(target_weights - self.weights).sum())
        cost = turnover * (self.transaction_cost_bps / 10_000)

        day_returns = self.returns_matrix.iloc[self.current_idx].to_numpy()
        day_returns_with_cash = np.append(day_returns, 0.0)  # cash earns 0

        port_return = float((target_weights * day_returns_with_cash).sum()) - cost
        downside_penalty = self.risk_aversion * max(0.0, -port_return)
        reward = port_return - downside_penalty

        grown = target_weights * (1 + day_returns_with_cash)
        total = grown.sum()
        self.weights = (grown / total).astype(np.float32) if total > 0 else target_weights

        self.current_idx += 1
        self.current_step += 1

        terminated = False
        truncated = (
            self.current_step >= self.episode_length
            or self.current_idx >= len(self.returns_matrix) - 1
        )

        obs = (
            self._get_obs() if not truncated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info = {
            "portfolio_return": port_return,
            "transaction_cost": cost,
            "turnover": turnover,
            "weights": target_weights.copy(),
        }
        return obs, reward, terminated, truncated, info
