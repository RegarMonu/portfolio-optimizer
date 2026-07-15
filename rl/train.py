"""
Phase 6 — Train a Deep RL portfolio allocation agent.

PPO is the primary algorithm (per the roadmap); SAC is available via
--algo sac as the "future comparison" algorithm the roadmap calls for.
Both work off-the-shelf with `PortfolioAllocationEnv`'s continuous Box
action space — no environment changes needed to switch algorithms.

Run via:
    python -m rl.train --synthetic --timesteps 20000
    python -m rl.train --synthetic --algo sac --timesteps 20000
"""

from __future__ import annotations

import argparse

import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config.logging_config import get_logger
from config.settings import BENCHMARK_TICKER, NIFTY_50_TICKERS, PROJECT_ROOT
from data_layer.storage import load_universe
from eda.correlation import build_returns_matrix
from rl.env import PortfolioAllocationEnv

logger = get_logger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

ALGOS = {"ppo": PPO, "sac": SAC}


def make_env_fn(returns_matrix: pd.DataFrame, **env_kwargs):
    """Factory so stable-baselines3's vec-env wrapper can construct fresh env instances."""
    def _make():
        env = PortfolioAllocationEnv(returns_matrix, **env_kwargs)
        return Monitor(env)
    return _make


def train_agent(
    returns_matrix: pd.DataFrame,
    algo: str = "ppo",
    total_timesteps: int = 20_000,
    lookback_window: int = 30,
    episode_length: int = 120,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 0.5,
    seed: int = 42,
):
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}', choose from {list(ALGOS)}")

    env_kwargs = dict(
        lookback_window=lookback_window, episode_length=episode_length,
        transaction_cost_bps=transaction_cost_bps, risk_aversion=risk_aversion,
        random_start=True, seed=seed,
    )
    vec_env = make_vec_env(make_env_fn(returns_matrix, **env_kwargs), n_envs=1, seed=seed)

    model_cls = ALGOS[algo]
    model = model_cls("MlpPolicy", vec_env, verbose=1, seed=seed)

    logger.info("Training %s for %d timesteps (episode_length=%d, lookback=%d)",
                algo.upper(), total_timesteps, episode_length, lookback_window)
    model.learn(total_timesteps=total_timesteps)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{algo}_portfolio_agent.zip"
    model.save(save_path)
    logger.info("Model saved to %s", save_path)

    return model, save_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Phase 6 RL portfolio allocation agent")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic-tickers", type=int, default=8)
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS))
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--lookback-window", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--risk-aversion", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.synthetic:
        from eda.report import _make_synthetic_universe
        universe = _make_synthetic_universe(n_tickers=args.n_synthetic_tickers)
    else:
        tickers = NIFTY_50_TICKERS + [BENCHMARK_TICKER]
        universe = load_universe(tickers)
        if BENCHMARK_TICKER not in universe:
            raise RuntimeError(
                "No processed benchmark data found. Run `python -m data_layer.pipeline` "
                "first, or use --synthetic to demo."
            )

    stock_frames = {t: f for t, f in universe.items() if t != BENCHMARK_TICKER}
    returns_matrix = build_returns_matrix(stock_frames).dropna()

    model, path = train_agent(
        returns_matrix, algo=args.algo, total_timesteps=args.timesteps,
        lookback_window=args.lookback_window, episode_length=args.episode_length,
        transaction_cost_bps=args.transaction_cost_bps, risk_aversion=args.risk_aversion,
        seed=args.seed,
    )
    print(f"Trained {args.algo.upper()} model saved to: {path}")
