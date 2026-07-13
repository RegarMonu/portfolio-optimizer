"""
Phase 2 — EDA: volatility measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series:
    """
    Rolling standard deviation of returns. Default window=21 trading days (~1 month).
    Annualized by sqrt(252) unless annualize=False.
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.rename(f"rolling_vol_{window}d")


def realized_volatility_regimes(
    returns: pd.Series, window: int = 21, low_q: float = 0.33, high_q: float = 0.67
) -> pd.Series:
    """
    Buckets each date into 'low' / 'medium' / 'high' volatility regime based
    on where its rolling vol falls relative to the full-sample quantiles.
    Useful later for regime-conditional analysis (e.g. does momentum work
    differently in high-vol periods?).
    """
    vol = rolling_volatility(returns, window=window)
    low_thresh, high_thresh = vol.quantile(low_q), vol.quantile(high_q)

    def _bucket(v):
        if pd.isna(v):
            return np.nan
        if v <= low_thresh:
            return "low"
        if v >= high_thresh:
            return "high"
        return "medium"

    return vol.apply(_bucket).rename("vol_regime")


def volatility_summary(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-row-per-ticker: current vol, historical avg vol, vol-of-vol."""
    rows = []
    for ticker, frame in price_frames.items():
        close = frame["Close"].dropna()
        if len(close) < 30:
            continue
        rets = close.pct_change().dropna()
        vol_series = rolling_volatility(rets, window=21).dropna()
        rows.append({
            "ticker": ticker,
            "current_21d_vol": vol_series.iloc[-1] if len(vol_series) else np.nan,
            "avg_21d_vol": vol_series.mean(),
            "max_21d_vol": vol_series.max(),
            "vol_of_vol": vol_series.std(),
        })
    return pd.DataFrame(rows).set_index("ticker").sort_values("avg_21d_vol", ascending=False)
