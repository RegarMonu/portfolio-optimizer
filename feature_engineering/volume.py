"""
Phase 3 — Feature Engineering: volume features.
"""

from __future__ import annotations

import pandas as pd


def volume_change(volume: pd.Series) -> pd.Series:
    """Day-over-day % change in traded volume."""
    return volume.pct_change().rename("volume_change")


def volume_moving_average(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window).mean().rename(f"volume_sma_{window}")


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Today's volume relative to its own recent average — a spike above 1.0
    (or a chosen threshold like 2.0) often flags unusual interest (news,
    earnings, institutional activity).
    """
    avg_vol = volume_moving_average(volume, window)
    return (volume / avg_vol).rename(f"relative_volume_{window}")
