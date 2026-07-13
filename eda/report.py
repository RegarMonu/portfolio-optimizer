"""
Phase 2 — EDA: report generator.

Produces:
    reports/eda_report.md          — narrative summary + embedded tables
    reports/figures/*.png          — price trends, correlation heatmap,
                                      drawdown chart, return distributions,
                                      sector performance

Run via:
    python -m eda.report                       # uses processed data on disk
    python -m eda.report --synthetic            # generates synthetic demo data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from config.logging_config import get_logger
from config.settings import BENCHMARK_TICKER, NIFTY_50_TICKERS, PROJECT_ROOT
from data_layer.storage import load_universe
from eda.correlation import (
    average_pairwise_correlation,
    correlation_matrix,
    most_correlated_pairs,
)
from eda.distribution import distribution_summary_table
from eda.drawdown import drawdown_series, max_drawdown
from eda.regime import classify_regimes, regime_period_summary
from eda.returns import cumulative_return, returns_summary_table
from eda.sector import sector_performance_summary, unmapped_tickers
from eda.volatility import volatility_summary

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _make_synthetic_universe(n_tickers: int = 10, n_days: int = 750, seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Generates plausible synthetic OHLCV data for demoing/testing the EDA
    module when no real processed data is on disk (e.g. no network access).
    Uses correlated geometric Brownian motion so correlation/sector analysis
    has something non-trivial to show.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    tickers = NIFTY_50_TICKERS[:n_tickers]
    market_factor = rng.normal(0.0004, 0.011, n_days)  # shared market component

    frames = {}
    for ticker in tickers:
        idio = rng.normal(0.0002, 0.014, n_days)
        beta = rng.uniform(0.6, 1.4)
        daily_ret = beta * market_factor + idio
        price = 100 * (1 + daily_ret).cumprod()
        close = pd.Series(price, index=dates)
        frames[ticker] = pd.DataFrame({
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n_days),
            "is_filled": False,
            "is_price_jump_flagged": False,
        }, index=dates)

    benchmark_ret = market_factor
    benchmark_price = 20000 * (1 + benchmark_ret).cumprod()
    bench_close = pd.Series(benchmark_price, index=dates)
    frames[BENCHMARK_TICKER] = pd.DataFrame({
        "Open": bench_close.shift(1).fillna(bench_close.iloc[0]),
        "High": bench_close * 1.005, "Low": bench_close * 0.995,
        "Close": bench_close, "Volume": 0,
        "is_filled": False, "is_price_jump_flagged": False,
    }, index=dates)

    logger.info("Generated synthetic universe: %d tickers x %d days", n_tickers, n_days)
    return frames


def plot_cumulative_returns(price_frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker, frame in price_frames.items():
        cum = cumulative_return(frame["Close"].dropna())
        ax.plot(cum.index, cum.values, label=ticker, linewidth=1.2, alpha=0.85)
    ax.set_title("Growth of ₹1 Invested")
    ax.set_ylabel("Cumulative Return (×)")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Return Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_benchmark_drawdown_and_regimes(benchmark_close: pd.Series, path: Path) -> None:
    dd = drawdown_series(benchmark_close)
    regimes = classify_regimes(benchmark_close)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(benchmark_close.index, benchmark_close.values, color="black", linewidth=1)
    colors = {"bull": "#2ca02c", "bear": "#d62728", "neutral": "#7f7f7f"}
    for regime_name, color in colors.items():
        mask = regimes == regime_name
        ax1.fill_between(benchmark_close.index, benchmark_close.min(), benchmark_close.max(),
                          where=mask.values, color=color, alpha=0.12, step="mid")
    ax1.set_title("Benchmark Price with Bull/Bear Regime Shading")
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in colors.values()],
               labels=list(colors.keys()), loc="upper left", fontsize=8)

    ax2.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax2.set_title("Drawdown from Running Peak")
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_return_distribution(price_frames: dict[str, pd.DataFrame], path: Path, sample_tickers: int = 4) -> None:
    tickers = [t for t in price_frames if t != BENCHMARK_TICKER][:sample_tickers]
    fig, axes = plt.subplots(1, len(tickers), figsize=(4 * len(tickers), 4), sharey=True)
    if len(tickers) == 1:
        axes = [axes]
    for ax, ticker in zip(axes, tickers):
        rets = price_frames[ticker]["Close"].pct_change().dropna()
        ax.hist(rets, bins=60, color="#1f77b4", alpha=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(ticker, fontsize=9)
    fig.suptitle("Daily Return Distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sector_performance(sector_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(sector_summary.index, sector_summary["total_return"] * 100, color="#1f77b4")
    ax.set_xlabel("Total Return (%)")
    ax.set_title("Sector-wise Total Return")
    ax.grid(alpha=0.3, axis="x")
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


def generate_report(price_frames: dict[str, pd.DataFrame]) -> Path:
    _ensure_dirs()
    logger.info("Generating Phase 2 EDA report over %d tickers", len(price_frames))

    stock_frames = {t: f for t, f in price_frames.items() if t != BENCHMARK_TICKER}
    benchmark_close = price_frames[BENCHMARK_TICKER]["Close"].dropna()

    ret_summary = returns_summary_table(stock_frames)
    vol_summary = volatility_summary(stock_frames)
    corr = correlation_matrix(stock_frames)
    avg_corr = average_pairwise_correlation(corr)
    top_pairs = most_correlated_pairs(corr, top_n=10)
    dist_summary = distribution_summary_table(stock_frames)
    sector_summary = sector_performance_summary(stock_frames)
    unmapped = unmapped_tickers(stock_frames)

    regimes = classify_regimes(benchmark_close)
    regime_periods = regime_period_summary(regimes)
    bench_mdd = max_drawdown(benchmark_close)

    plot_cumulative_returns(stock_frames, FIGURES_DIR / "cumulative_returns.png")
    plot_correlation_heatmap(corr, FIGURES_DIR / "correlation_heatmap.png")
    plot_benchmark_drawdown_and_regimes(benchmark_close, FIGURES_DIR / "drawdown_regimes.png")
    plot_return_distribution(stock_frames, FIGURES_DIR / "return_distributions.png")
    plot_sector_performance(sector_summary, FIGURES_DIR / "sector_performance.png")

    md = []
    md.append("# Phase 2 — Exploratory Data Analysis Report\n")
    md.append(f"Universe: {len(stock_frames)} stocks + benchmark ({BENCHMARK_TICKER}). "
              f"Date range: {benchmark_close.index.min().date()} to {benchmark_close.index.max().date()}.\n")

    md.append("## 1. Price Trends\n")
    md.append("![Cumulative Returns](figures/cumulative_returns.png)\n")

    md.append("## 2. Returns & Volatility Summary\n")
    md.append(_df_to_markdown(ret_summary) + "\n")
    md.append(_df_to_markdown(vol_summary) + "\n")

    md.append("## 3. Correlation Structure\n")
    md.append(f"Average pairwise correlation across the universe: **{avg_corr:.3f}** "
              f"(closer to 1 = less diversification benefit available).\n")
    md.append("![Correlation Heatmap](figures/correlation_heatmap.png)\n")
    md.append("### Most correlated pairs\n")
    md.append(_df_to_markdown(top_pairs) + "\n")

    md.append("## 4. Sector-wise Performance\n")
    if unmapped:
        md.append(f"_Note: {len(unmapped)} ticker(s) had no sector mapping and were excluded: "
                  f"{unmapped}_\n")
    md.append(_df_to_markdown(sector_summary) + "\n")
    md.append("![Sector Performance](figures/sector_performance.png)\n")

    md.append("## 5. Drawdowns & Bull/Bear Regimes\n")
    md.append(f"Benchmark maximum drawdown over the sample period: **{bench_mdd:.1%}**\n")
    md.append("![Drawdown and Regimes](figures/drawdown_regimes.png)\n")
    md.append("### Regime episodes\n")
    md.append(_df_to_markdown(regime_periods) + "\n")

    md.append("## 6. Distribution of Returns\n")
    md.append("Fat tails / skew matter for Phase 4: mean-variance optimization assumes "
              "near-normal returns, and Indian equities frequently violate that "
              "assumption (see kurtosis_excess column below; >0 means fatter "
              "tails than normal, i.e. extreme moves are more common than a "
              "normal distribution would predict).\n")
    md.append(_df_to_markdown(dist_summary) + "\n")
    md.append("![Return Distributions](figures/return_distributions.png)\n")

    report_text = "\n".join(md)
    report_path = REPORTS_DIR / "eda_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("EDA report written to %s", report_path)
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 EDA report generator")
    parser.add_argument("--synthetic", action="store_true",
                         help="Use synthetic demo data instead of reading data_store/processed")
    parser.add_argument("--n-synthetic-tickers", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.synthetic:
        universe = _make_synthetic_universe(n_tickers=args.n_synthetic_tickers)
    else:
        tickers = NIFTY_50_TICKERS + [BENCHMARK_TICKER]
        universe = load_universe(tickers)
        if BENCHMARK_TICKER not in universe:
            raise RuntimeError(
                f"No processed data found for benchmark {BENCHMARK_TICKER}. "
                "Run `python -m data_layer.pipeline` first, or use --synthetic to demo."
            )

    path = generate_report(universe)
    print(f"Report written to: {path}")
