"""
Unit tests for data_layer.cleaner — pure pandas logic, no network calls.
Run with: python -m pytest tests/test_cleaner.py -v
"""

import numpy as np
import pandas as pd
import pytest

from data_layer.cleaner import build_trading_calendar, clean_single_ticker, clean_universe


def make_frame(dates, closes):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=pd.DatetimeIndex(dates),
    )


def test_build_trading_calendar_is_union_of_all_dates():
    dates_a = pd.date_range("2024-01-01", periods=5)
    dates_b = pd.date_range("2024-01-03", periods=5)
    frames = {
        "A": make_frame(dates_a, [100] * 5),
        "B": make_frame(dates_b, [200] * 5),
    }
    calendar = build_trading_calendar(frames)
    assert len(calendar) == 7  # union of Jan 1-5 and Jan 3-7


def test_short_gap_is_forward_filled():
    dates = pd.date_range("2024-01-01", periods=10)
    closes = [100, 101, 102, np.nan, np.nan, 105, 106, 107, 108, 109]
    frame = make_frame(dates, closes)
    # Simulate a gap by dropping rows 4,5 entirely (not NaN, actually missing)
    frame = frame.drop(frame.index[[3, 4]])

    calendar = pd.DatetimeIndex(dates)
    cleaned = clean_single_ticker("TEST", frame, calendar)

    assert cleaned is not None
    assert cleaned["Close"].isna().sum() == 0
    assert cleaned.loc[dates[3], "is_filled"] == True  # noqa: E712
    assert cleaned.loc[dates[3], "Close"] == 102  # forward-filled from prior close


def test_long_gap_causes_ticker_drop():
    dates = pd.date_range("2024-01-01", periods=30)
    frame = make_frame(dates, list(range(100, 130)))
    # Drop 20 consecutive days -> way more than max_consecutive_missing_days
    frame = frame.drop(frame.index[5:25])

    calendar = pd.DatetimeIndex(dates)
    cleaned = clean_single_ticker("TEST", frame, calendar)

    assert cleaned is None  # should be dropped, not filled


def test_extreme_price_jump_is_flagged_not_removed():
    dates = pd.date_range("2024-01-01", periods=5)
    closes = [100, 101, 102, 250, 251]  # >35% jump on day 4
    frame = make_frame(dates, closes)

    calendar = pd.DatetimeIndex(dates)
    cleaned = clean_single_ticker("TEST", frame, calendar)

    assert cleaned is not None
    assert len(cleaned) == 5  # nothing removed
    assert cleaned["is_price_jump_flagged"].sum() == 1


def test_clean_universe_drops_bad_tickers_keeps_good_ones():
    dates = pd.date_range("2024-01-01", periods=30)
    good = make_frame(dates, list(range(100, 130)))
    bad = make_frame(dates, list(range(100, 130))).drop(
        make_frame(dates, list(range(100, 130))).index[5:25]
    )

    result = clean_universe({"GOOD": good, "BAD": bad})
    assert "GOOD" in result
    assert "BAD" not in result


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
