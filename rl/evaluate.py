"""
Phase 6 — Evaluate a trained RL agent against Phase 4/5 classical baselines.

Run via:
    python -m rl.evaluate --synthetic --model-path models/ppo_portfolio_agent.zip
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from config.logging_config import get_logger
from config.settings import BENCHMARK_TICKER, NIFTY_50_TICKERS, PROJECT_ROOT
from data_layer.storage import load_universe
from eda.correlation import build_returns_matrix
from optimization.metrics import performance_summary
from rl.baselines import run_baselines
from rl.env import PortfolioAllocationEnv

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ALGOS = {"ppo": PPO, "sac": SAC}


def evaluate_agent(
    model,
    returns_matrix: pd.DataFrame,
    lookback_window: int = 30,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 0.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Runs the trained policy DETERMINISTICALLY over the full available
    period (random_start=False, one long episode) and returns:
        daily_returns: portfolio return Series, correctly date-indexed
        weights_history: DataFrame of the agent's weights at every step
            (columns = tickers + 'cash')
    """
    episode_length = len(returns_matrix) - lookback_window - 1
    env = PortfolioAllocationEnv(
        returns_matrix, lookback_window=lookback_window, episode_length=episode_length,
        transaction_cost_bps=transaction_cost_bps, risk_aversion=risk_aversion,
        random_start=False,
    )

    obs, _ = env.reset()
    returns_list, weights_list = [], []
    truncated = False

    while not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        returns_list.append(info["portfolio_return"])
        weights_list.append(info["weights"])

    dates = returns_matrix.index[lookback_window : lookback_window + len(returns_list)]
    daily_returns = pd.Series(returns_list, index=dates, name="portfolio_return")
    weights_history = pd.DataFrame(weights_list, index=dates, columns=env.tickers + ["cash"])

    return daily_returns, weights_history


def _slice_and_rebase(daily_returns: pd.Series, start_date, end_date) -> pd.Series:
    """Restricts a return series to [start_date, end_date] for apples-to-apples comparison."""
    return daily_returns.loc[start_date:end_date]


def _df_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include="float").columns:
        formatted[col] = formatted[col].map(lambda x: float_format.format(x) if pd.notna(x) else "")
    try:
        return formatted.to_markdown()
    except ImportError:
        return formatted.to_string()


def plot_equity_curves(all_returns: dict[str, pd.Series], path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, rets in all_returns.items():
        cumulative = (1 + rets.fillna(0)).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=name, linewidth=1.6)
    ax.set_title("RL Agent vs. Classical Baselines — Equity Curves")
    ax.set_ylabel("Growth of ₹1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_agent_weight_evolution(weights_history: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(weights_history.index, weights_history.T.values, labels=weights_history.columns)
    ax.set_title("RL Agent's Portfolio Allocation Over Time")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=6, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def generate_report(
    model,
    price_frames: dict[str, pd.DataFrame],
    lookback_window: int = 30,
    transaction_cost_bps: float = 10.0,
    risk_aversion: float = 0.5,
    risk_free_rate: float = 0.065,
) -> tuple:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stock_frames = {t: f for t, f in price_frames.items() if t != BENCHMARK_TICKER}
    returns_matrix = build_returns_matrix(stock_frames).dropna()

    agent_returns, weights_history = evaluate_agent(
        model, returns_matrix, lookback_window, transaction_cost_bps, risk_aversion,
    )
    eval_start, eval_end = agent_returns.index[0], agent_returns.index[-1]

    baseline_results = run_baselines(returns_matrix, transaction_cost_bps=transaction_cost_bps)

    all_returns = {"rl_agent": agent_returns}
    for name, result in baseline_results.items():
        all_returns[name] = _slice_and_rebase(result.daily_returns, eval_start, eval_end)

    if BENCHMARK_TICKER in price_frames:
        bench_close = price_frames[BENCHMARK_TICKER]["Close"].reindex(returns_matrix.index).ffill()
        bench_rets = bench_close.pct_change().fillna(0)
        all_returns["nifty50_index"] = _slice_and_rebase(bench_rets, eval_start, eval_end)

    summary_rows = {
        name: performance_summary(rets.dropna(), risk_free_rate)
        for name, rets in all_returns.items()
    }
    summary_df = pd.DataFrame(summary_rows).T.sort_values("sharpe_ratio", ascending=False)

    # Portfolio stability metric requested by the roadmap: std-dev of period-over-period weight changes
    weight_churn = weights_history.diff().abs().sum(axis=1)
    stability_note = (
        f"Average daily portfolio turnover (stability proxy): {weight_churn.mean():.4f} "
        f"(lower = more stable allocation over time)"
    )

    plot_equity_curves(all_returns, FIGURES_DIR / "rl_vs_baselines_equity_curves.png")
    plot_agent_weight_evolution(weights_history, FIGURES_DIR / "rl_agent_weight_evolution.png")

    md = []
    md.append("# Phase 6 — Deep RL Portfolio Optimization Report\n")
    md.append(f"Evaluation period: {eval_start.date()} to {eval_end.date()} "
              f"({len(agent_returns)} trading days). Transaction cost: "
              f"{transaction_cost_bps} bps. Risk-free rate: {risk_free_rate:.1%}.\n")

    md.append("## Performance Comparison: RL Agent vs. Classical Baselines\n")
    md.append(_df_to_markdown(summary_df) + "\n")

    md.append("## Equity Curves\n")
    md.append("![Equity Curves](figures/rl_vs_baselines_equity_curves.png)\n")

    md.append("## RL Agent's Allocation Over Time\n")
    md.append("![Weight Evolution](figures/rl_agent_weight_evolution.png)\n")

    md.append("## Portfolio Stability\n")
    md.append(stability_note + "\n")

    md.append("## Honest Assessment\n")
    md.append(
        "This is a demonstration-scale run (see the training command's "
        "`--timesteps` argument). A properly tuned agent for real research "
        "would need substantially more training steps, hyperparameter "
        "search, and likely a train/validation/test split in time (train "
        "on one period, evaluate out-of-sample on a later, unseen period) "
        "rather than evaluating over the same data distribution it trained "
        "on. Treat any outperformance here as a proof of concept that the "
        "MDP/environment is wired correctly, not as evidence the RL "
        "approach beats classical methods in general.\n"
    )

    report_path = REPORTS_DIR / "rl_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("RL evaluation report written to %s", report_path)
    return report_path, summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 6 RL agent vs. classical baselines")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic-tickers", type=int, default=8)
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--lookback-window", type=int, default=30)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--risk-aversion", type=float, default=0.5)
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

    model_path = args.model_path or str(PROJECT_ROOT / "models" / f"{args.algo}_portfolio_agent.zip")
    model = ALGOS[args.algo].load(model_path)

    path, summary = generate_report(
        model, universe, lookback_window=args.lookback_window,
        transaction_cost_bps=args.transaction_cost_bps, risk_aversion=args.risk_aversion,
    )
    print(f"Report written to: {path}\n")
    print(summary)
