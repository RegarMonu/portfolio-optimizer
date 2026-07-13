"""
Phase 3 — Feature Engineering: market-wide features.

These are the same value for every stock on a given date (they describe
the overall market environment, not the individual stock), so they get
broadcast/joined onto each ticker's feature frame in `pipeline.py`.
"""

from __future__ import annotations

import pandas as pd


def benchmark_return_feature(benchmark_close: pd.Series, window: int = 1) -> pd.Series:
    """NIFTY 50 return over `window` days — the core 'market factor' feature."""
    if window == 1:
        return benchmark_close.pct_change().rename("nifty_return_1d")
    return (benchmark_close / benchmark_close.shift(window) - 1).rename(f"nifty_return_{window}d")


def india_vix_level(vix_close: pd.Series) -> pd.Series:
    """Raw India VIX level — a direct market-implied-volatility gauge."""
    return vix_close.rename("india_vix_level")


def india_vix_change(vix_close: pd.Series) -> pd.Series:
    return vix_close.pct_change().rename("india_vix_change")


def sector_relative_strength(
    stock_close: pd.Series, sector_return_series: pd.Series
) -> pd.Series:
    """
    Stock's daily return minus its sector's equal-weighted daily return —
    is this stock outperforming or underperforming its own peer group?
    """
    stock_ret = stock_close.pct_change()
    aligned_sector_ret = sector_return_series.reindex(stock_ret.index)
    return (stock_ret - aligned_sector_ret).rename("sector_relative_strength")
