"""
Phase 2 — EDA: correlation & covariance analysis across the universe.
"""

from __future__ import annotations

import pandas as pd


def build_returns_matrix(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aligns all tickers' daily returns into one wide DataFrame
    (columns = tickers, index = dates), inner-joined on common dates so the
    correlation matrix isn't distorted by mismatched date ranges.
    """
    series_dict = {}
    for ticker, frame in price_frames.items():
        rets = frame["Close"].pct_change().dropna()
        series_dict[ticker] = rets
    matrix = pd.DataFrame(series_dict).dropna(how="all")
    return matrix


def correlation_matrix(price_frames: dict[str, pd.DataFrame], method: str = "pearson") -> pd.DataFrame:
    returns_matrix = build_returns_matrix(price_frames)
    return returns_matrix.corr(method=method)


def covariance_matrix(price_frames: dict[str, pd.DataFrame], annualize: bool = True) -> pd.DataFrame:
    returns_matrix = build_returns_matrix(price_frames)
    cov = returns_matrix.cov()
    return cov * 252 if annualize else cov


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    Mean of all off-diagonal correlation values — a single-number proxy for
    'how diversified is this universe right now'. Closer to 1 = less
    diversification benefit; closer to 0 = more.
    """
    import numpy as np
    values = corr_matrix.to_numpy(dtype=float)
    n = values.shape[0]
    if n < 2:
        return float("nan")
    off_diagonal_mask = ~np.eye(n, dtype=bool)
    return float(values[off_diagonal_mask].mean())


def most_correlated_pairs(corr_matrix: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top N ticker pairs by absolute correlation (excluding self-pairs and duplicates)."""
    pairs = []
    cols = corr_matrix.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append({"ticker_a": a, "ticker_b": b, "correlation": corr_matrix.loc[a, b]})
    df = pd.DataFrame(pairs)
    return df.reindex(df["correlation"].abs().sort_values(ascending=False).index).head(top_n)
