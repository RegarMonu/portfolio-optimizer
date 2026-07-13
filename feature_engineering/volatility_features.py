"""
Phase 3 — Feature Engineering: volatility features.

Note: `rolling_volatility` already exists in `eda/volatility.py` (built in
Phase 2 for descriptive analysis). Rather than duplicate it, this module
re-exports it for use as a model feature and adds ATR, which needs
High/Low/Close (not just Close) so it didn't fit naturally in the EDA module.
"""

from __future__ import annotations

import pandas as pd

from eda.volatility import rolling_volatility  # re-exported as a Phase 3 feature too

__all__ = ["rolling_volatility", "average_true_range"]


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    range1 = high - low
    range2 = (high - prev_close).abs()
    range3 = (low - prev_close).abs()
    return pd.concat([range1, range2, range3], axis=1).max(axis=1)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Average True Range (Wilder's smoothing) — captures volatility including
    gaps, unlike a simple High-Low range. Widely used for position sizing
    and stop-loss placement.
    """
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr.rename(f"atr_{window}")
