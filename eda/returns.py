"""
Phase 2 — EDA: return computations.

All functions take a `prices` Series/DataFrame of Close prices indexed by
date and return returns at the requested frequency. Kept separate from
Phase 3's `feature_engineering` package (which will build ON TOP of these
for model features) — this module is purely descriptive/analytical.
"""

from __future__ import annotations

import pandas as pd


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily percentage returns."""
    return close.pct_change().rename("daily_return")


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns — additive across time, used for compounding/CAGR math."""
    import numpy as np
    return np.log(close / close.shift(1)).rename("log_return")


def periodic_returns(close: pd.Series, freq: str = "W") -> pd.Series:
    """
    Resample to period-end price then compute simple returns.
    freq: 'W' (weekly), 'ME' (month-end), 'YE' (year-end) — pandas offset aliases.
    """
    period_end_price = close.resample(freq).last()
    return period_end_price.pct_change().rename(f"{freq}_return")


def cumulative_return(close: pd.Series) -> pd.Series:
    """Growth of ₹1 invested at the start of the series."""
    return (close / close.iloc[0]).rename("cumulative_return")


def cagr(close: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate over the full span of the series."""
    n_periods = len(close.dropna())
    if n_periods < 2:
        return float("nan")
    total_return = close.iloc[-1] / close.iloc[0]
    years = n_periods / periods_per_year
    return total_return ** (1 / years) - 1


def returns_summary_table(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One-row-per-ticker summary: annualized return, annualized vol, CAGR.
    price_frames: {ticker: DataFrame with a 'Close' column}
    """
    rows = []
    for ticker, frame in price_frames.items():
        close = frame["Close"].dropna()
        if len(close) < 2:
            continue
        daily = daily_returns(close).dropna()
        rows.append({
            "ticker": ticker,
            "cagr": cagr(close),
            "annualized_return_mean": daily.mean() * 252,
            "annualized_volatility": daily.std() * (252 ** 0.5),
            "n_observations": len(close),
            "start_date": close.index.min(),
            "end_date": close.index.max(),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("cagr", ascending=False)
