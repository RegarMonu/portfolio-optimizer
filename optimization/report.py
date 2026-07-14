"""
Phase 4 — Report generator: runs every classical optimization method over
the same universe/period and produces a side-by-side performance comparison
against Equal Weight and a Buy & Hold benchmark, per the roadmap deliverable.

Run via:
    python -m optimization.report
    python -m optimization.report --synthetic
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
from eda.correlation import build_returns_matrix, covariance_matrix
from optimization.equal_weight import equal_weights
from optimization.markowitz import max_sharpe_weights, min_variance_weights
from optimization.metrics import performance_summary
from optimization.risk_parity import risk_parity_weights, verify_risk_parity
from optimization.simulate import simulate_portfolio

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_all_methods(
    price_frames: dict[str, pd.DataFrame],
    risk_free_rate: float = 0.065,
    rebalance_freq: str = "ME",
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """
    Returns:
        portfolio_returns: {method_name: daily return Series}
        weights_by_method: {method_name: weight Series}
    """
    stock_frames = {t: f for t, f in price_frames.items() if t != BENCHMARK_TICKER}
    returns_matrix = build_returns_matrix(stock_frames).dropna()
    mean_returns = returns_matrix.mean() * 252  # annualized
    cov = covariance_matrix(stock_frames, annualize=True)
    cov = cov.loc[returns_matrix.columns, returns_matrix.columns]

    tickers = returns_matrix.columns.tolist()

    weights_by_method = {
        "equal_weight": equal_weights(tickers),
        "min_variance": min_variance_weights(cov),
        "max_sharpe": max_sharpe_weights(mean_returns, cov, risk_free_rate),
        "risk_parity": risk_parity_weights(cov),
    }

    portfolio_returns = {}
    for method, weights in weights_by_method.items():
        portfolio_returns[method] = simulate_portfolio(returns_matrix, weights, rebalance_freq)
        logger.info("Simulated %s: %d rebalances at freq=%s", method, 1, rebalance_freq)

    # Buy & Hold benchmark = the NIFTY 50 index itself, if available
    if BENCHMARK_TICKER in price_frames:
        bench_close = price_frames[BENCHMARK_TICKER]["Close"].reindex(returns_matrix.index).ffill()
        portfolio_returns["buy_and_hold_benchmark"] = bench_close.pct_change().rename("portfolio_return")

    return portfolio_returns, weights_by_method


def plot_equity_curves(portfolio_returns: dict[str, pd.Series], path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for method, rets in portfolio_returns.items():
        cumulative = (1 + rets.fillna(0)).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=method, linewidth=1.5)
    ax.set_title("Equity Curves: Classical Portfolio Optimization Methods")
    ax.set_ylabel("Growth of ₹1")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_weight_allocations(weights_by_method: dict[str, pd.Series], path) -> None:
    df = pd.DataFrame(weights_by_method).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Portfolio Weight Allocation by Method")
    ax.set_ylabel("Weight")
    ax.legend(fontsize=8)
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


def generate_report(price_frames: dict[str, pd.DataFrame], risk_free_rate: float = 0.065) -> None:
    _ensure_dirs()
    portfolio_returns, weights_by_method = run_all_methods(price_frames, risk_free_rate)

    summary_rows = {}
    for method, rets in portfolio_returns.items():
        summary_rows[method] = performance_summary(rets.dropna(), risk_free_rate)
    summary_df = pd.DataFrame(summary_rows).T.sort_values("sharpe_ratio", ascending=False)

    plot_equity_curves(portfolio_returns, FIGURES_DIR / "optimization_equity_curves.png")
    plot_weight_allocations(weights_by_method, FIGURES_DIR / "optimization_weights.png")

    # Risk parity sanity check — confirm risk contributions are actually balanced
    cov = covariance_matrix(
        {t: f for t, f in price_frames.items() if t != BENCHMARK_TICKER}, annualize=True
    )
    rp_weights = weights_by_method["risk_parity"]
    risk_shares = verify_risk_parity(rp_weights, cov.loc[rp_weights.index, rp_weights.index])

    md = []
    md.append("# Phase 4 — Classical Portfolio Optimization Report\n")
    md.append(f"Universe: {len([t for t in price_frames if t != BENCHMARK_TICKER])} stocks. "
              f"Risk-free rate assumption: {risk_free_rate:.1%}. Rebalanced monthly.\n")

    md.append("## Performance Comparison\n")
    md.append(_df_to_markdown(summary_df) + "\n")

    md.append("## Equity Curves\n")
    md.append("![Equity Curves](figures/optimization_equity_curves.png)\n")

    md.append("## Weight Allocations by Method\n")
    md.append("![Weight Allocations](figures/optimization_weights.png)\n")

    md.append("## Risk Parity Sanity Check\n")
    md.append("Each asset's actual share of total portfolio risk (should all be "
              f"close to 1/{len(risk_shares)} = {1/len(risk_shares):.3f} in a well-converged solution):\n")
    md.append(_df_to_markdown(risk_shares.to_frame()) + "\n")

    report_path = REPORTS_DIR / "optimization_report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Optimization report written to %s", report_path)
    return report_path, summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 4 optimization comparison report")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-synthetic-tickers", type=int, default=10)
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

    path, summary = generate_report(universe, risk_free_rate=args.risk_free_rate)
    print(f"Report written to: {path}\n")
    print(summary)
