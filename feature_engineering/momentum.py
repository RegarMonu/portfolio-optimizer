"""
Phase 3 — Feature Engineering: momentum features.
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing method).
    Bounded 0-100. Traditionally >70 = overbought, <30 = oversold, though
    those thresholds are a starting heuristic, not a hard rule.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = an EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (pure uptrend), RS is inf -> RSI should be 100
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    return rsi_values.rename(f"rsi_{window}")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.
    Returns a DataFrame with columns: macd_line, signal_line, macd_histogram.
    Histogram = macd_line - signal_line; crossovers of histogram through zero
    are the classic MACD trading signal.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "macd_histogram": histogram,
    })


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    """Rate of Change: % change in price over `window` periods."""
    return (close / close.shift(window) - 1).rename(f"roc_{window}")
