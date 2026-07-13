"""
Phase 2 — EDA: drawdown analysis.
"""

from __future__ import annotations

import pandas as pd


def drawdown_series(close: pd.Series) -> pd.Series:
    """
    Drawdown at each point = (current price / running peak) - 1.
    Always <= 0. A value of -0.20 means "20% below the highest price seen so far".
    """
    running_peak = close.cummax()
    return (close / running_peak - 1).rename("drawdown")


def max_drawdown(close: pd.Series) -> float:
    return drawdown_series(close).min()


def drawdown_periods(close: pd.Series, threshold: float = -0.10) -> pd.DataFrame:
    """
    Identifies distinct drawdown episodes deeper than `threshold` (default -10%).
    Returns one row per episode: start, trough, recovery date (or None if
    still ongoing at series end), depth, and duration in trading days.
    """
    dd = drawdown_series(close)
    in_drawdown = dd <= threshold

    episodes = []
    start_idx = None

    for i, (date, flagged) in enumerate(in_drawdown.items()):
        if flagged and start_idx is None:
            start_idx = i
        elif not flagged and start_idx is not None:
            episodes.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        episodes.append((start_idx, len(dd) - 1))

    rows = []
    for s, e in episodes:
        segment = dd.iloc[s : e + 1]
        trough_date = segment.idxmin()
        # Recovery = first date after trough where price regains the pre-drawdown peak
        peak_price = close.iloc[:s+1].cummax().iloc[-1] if s > 0 else close.iloc[0]
        post_trough = close.loc[trough_date:]
        recovery_candidates = post_trough[post_trough >= peak_price]
        recovery_date = recovery_candidates.index[0] if len(recovery_candidates) else None

        rows.append({
            "start_date": dd.index[s],
            "trough_date": trough_date,
            "recovery_date": recovery_date,
            "depth": segment.min(),
            "duration_trading_days": e - s + 1,
            "still_ongoing": recovery_date is None,
        })

    return pd.DataFrame(rows).sort_values("depth")


def calmar_ratio(close: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |Max Drawdown| — reward per unit of worst-case pain."""
    from eda.returns import cagr as _cagr
    mdd = max_drawdown(close)
    if mdd == 0:
        return float("nan")
    return _cagr(close, periods_per_year) / abs(mdd)
