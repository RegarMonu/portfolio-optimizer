"""
Phase 3 — Feature Engineering: trend features.
"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window).mean().rename(f"sma_{window}")


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (more weight on recent prices than SMA)."""
    return close.ewm(span=span, adjust=False).mean().rename(f"ema_{span}")


def price_to_sma_ratio(close: pd.Series, window: int) -> pd.Series:
    """
    Close / SMA - 1. Positive means price is trading above its trend line.
    A common, simple trend-strength signal for downstream models.
    """
    moving_avg = sma(close, window)
    return (close / moving_avg - 1).rename(f"price_to_sma_{window}_ratio")


def sma_crossover_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """
    +1 when fast SMA > slow SMA ("golden cross" regime), -1 otherwise
    ("death cross" regime). Classic trend-following signal.
    """
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)
    signal = (fast_ma > slow_ma).astype(int) * 2 - 1
    return signal.rename(f"sma_crossover_{fast}_{slow}")
