"""
Phase 5 — Backtest report: runs every Phase 4 strategy through the real
BacktestEngine (with transaction costs, proper rebalancing, no look-ahead)
and compares performance, turnover, and transaction cost drag.

Run via:
    python -m backtesting.report
    python -m backtesting.report --synthetic
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.logging_config import get_logger
from config.settings import BENCHMARK_TICKER, NIFTY_50_TICKERS, PROJECT_ROOT
from data_layer.storage import load_universe
from eda.correlation import build_returns_matrix
from backtesting.engine import BacktestEngine
from backtesting.strategy import (
    EqualWeightStrategy,
    FixedWeightStrategy,
    MaxSharpeStrategy,
    MinVarianceStrategy,
    RiskParityStrategy,
)
from optimization.metrics import performance_summary

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curves(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, result in results.items():
        ax.plot(result.cumulative_value.index, result.cumulative_value.values,
                label=name, linewidth=1.5)
    ax.set_title("Backtest Equity Curves (with transaction costs)")
    ax.set_ylabel("Growth of ₹1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_turnover(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, result in results.items():
        if len(result.turnover_history):
            ax.plot(result.turnover_history.index, result.turnover_history.values,
                    marker="o", markersize=3, label=name, linewidth=1)
    ax.set_title("Portfolio Turnover at Each Rebalance")
    ax.set_ylabel("Turnover (fraction of portfolio traded)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _df_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include="float").columns:
        formatted[col] = formatted[col].map(lambda x: float_format.format(x) if pd.notna(x) else "")
    try:
        return formatted.to_markdown()
    except ImportError:
        return formatted.to_string()


def generate_report(
    price_frames: dict[str, pd.DataFrame],
    rebalance_freq: str = "ME",
    transaction_cost_bps: float = 10.0,
    risk_free_rate: float = 0.065,
) -> tuple:
    _ensure_dirs()
    stock_frames = {t: f for t, f in price_frames.items() if t != BENCHMARK_TICKER}
    returns_matrix = build_returns_matrix(stock_frames).dropna()

    engine = BacktestEngine(
        returns_matrix, rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
    )

    strategies = [
        EqualWeightStrategy(),
        MinVarianceStrategy(),
        MaxSharpeStrategy(risk_free_rate=risk_free_rate),
        RiskParityStrategy(),
    ]
    results = engine.run_many(strategies)

    # Buy & Hold: fixed initial equal weights, held with NO rebalancing at all
    bh_engine = BacktestEngine(returns_matrix, rebalance_freq=None, transaction_cost_bps=transaction_cost_bps)
    initial_weights = pd.Series(1 / len(returns_matrix.columns), index=returns_matrix.columns)
    results["buy_and_hold"] = bh_engine.run(FixedWeightStrategy(initial_weights, name="buy_and_hold"))

    if BENCHMARK_TICKER in price_frames:
        bench_close = price_frames[BENCHMARK_TICKER]["Close"].reindex(returns_matrix.index).ffill()
        bench_returns = bench_close.pct_change().fillna(0)
        from backtesting.engine import BacktestResult
        results["nifty50_index"] = BacktestResult(
            strategy_name="nifty50_index", daily_returns=bench_returns,
            weights_history=pd.DataFrame(), turnover_history=pd.Series(dtype=float),
            transaction_costs=pd.Series(dtype=float),
        )

    summary_rows = {}
    for name, result in results.items():
        metrics = performance_summary(result.daily_returns.dropna(), risk_free_rate)
        metrics["total_txn_cost_drag"] = result.total_transaction_cost_drag
        metrics["n_rebalances"] = len(result.turnover_history)
        metrics["avg_turnover"] = result.turnover_history.mean() if len(result.turnover_history) else 0.0
        metrics["n_rebalance_failures"] = len(result.rebalance_failures)
        summary_rows[name] = metrics
    summary_df = pd.DataFrame(summary_rows).T.sort_values("sharpe_ratio", ascending=False)

    plot_equity_curves(results, FIGURES_DIR / "backtest_equity_curves.png")
    plot_turnover(results, FIGURES_DIR / "backtest_turnover.png")

    md = []
    md.append("# Phase 5 — Backtesting Engine Report\n")
    md.append(f"Universe: {len(stock_frames)} stocks. Rebalance frequency: {rebalance_freq}. "
              f"Transaction cost: {transaction_cost_bps} bps of turnover. "
              f"Risk-free rate: {risk_free_rate:.1%}.\n")

    md.append("## Performance Comparison (net of transaction costs)\n")
    md.append(_df_to_markdown(summary_df) + "\n")

    md.append("## Equity Curves\n")
    md.append("![Equity Curves](figures/backtest_equity_curves.png)\n")

    md.append("## Turnover per Rebalance\n")
    md.append("Higher turnover strategies pay more in transaction costs — compare "
              "`avg_turnover` and `total_txn_cost_drag` columns above against the "
              "raw (zero-cost) Phase 4 results to see the real-world impact.\n")
    md.append("![Turnover](figures/backtest_turnover.png)\n")

    any_failures = sum(len(r.rebalance_failures) for r in results.values())
    if any_failures:
        md.append(f"\n**Note:** {any_failures} rebalance(s) across all strategies fell back to "
                   "holding previous weights due to optimizer convergence issues (logged in "
                   "`logs/pipeline.log`).\n")

    report_path = REPORTS_DIR / "backtest_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Backtest report written to %s", report_path)
    return report_path, summary_df, results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5 backtest comparison report")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic-tickers", type=int, default=10)
    parser.add_argument("--rebalance-freq", type=str, default="ME")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.065)
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

    path, summary, _ = generate_report(
        universe, rebalance_freq=args.rebalance_freq,
        transaction_cost_bps=args.transaction_cost_bps, risk_free_rate=args.risk_free_rate,
    )
    print(f"Report written to: {path}\n")
    print(summary)
