"""
Phase 2 — EDA: sector-wise performance.
Uses config.settings.SECTOR_MAP to group tickers, then aggregates
equal-weighted sector returns for comparison.
"""

from __future__ import annotations

import pandas as pd

from config.settings import SECTOR_MAP
from eda.correlation import build_returns_matrix


def sector_returns(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Equal-weighted daily return per sector = mean of member-stock daily returns.
    Tickers without a sector mapping are excluded (logged as a gap, not silently
    dropped from the caller's perspective — see `unmapped_tickers`).
    """
    returns_matrix = build_returns_matrix(price_frames)
    sector_frames = {}

    for sector in sorted(set(SECTOR_MAP.values())):
        members = [t for t in returns_matrix.columns if SECTOR_MAP.get(t) == sector]
        if members:
            sector_frames[sector] = returns_matrix[members].mean(axis=1)

    return pd.DataFrame(sector_frames)


def unmapped_tickers(price_frames: dict[str, pd.DataFrame]) -> list[str]:
    return [t for t in price_frames if t not in SECTOR_MAP]


def sector_performance_summary(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sec_returns = sector_returns(price_frames)
    rows = []
    for sector in sec_returns.columns:
        rets = sec_returns[sector].dropna()
        cumulative = (1 + rets).prod() - 1
        rows.append({
            "sector": sector,
            "total_return": cumulative,
            "annualized_return": rets.mean() * 252,
            "annualized_volatility": rets.std() * (252 ** 0.5),
            "n_members": sum(1 for t, s in SECTOR_MAP.items() if s == sector and t in price_frames),
        })
    return pd.DataFrame(rows).set_index("sector").sort_values("total_return", ascending=False)
