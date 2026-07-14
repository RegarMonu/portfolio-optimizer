"""
Phase 4 — Equal Weight Portfolio (the baseline every other method must beat).
"""

from __future__ import annotations

import pandas as pd


def equal_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    return pd.Series(1 / n, index=tickers, name="weight")
