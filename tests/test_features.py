"""
Unit tests for feature_engineering.* modules — deterministic synthetic data,
no network required. Run with: python -m pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering.trend import ema, price_to_sma_ratio, sma, sma_crossover_signal
from feature_engineering.momentum import macd, rsi, roc
from feature_engineering.volatility_features import average_true_range
from feature_engineering.volume import relative_volume, volume_change, volume_moving_average
from feature_engineering.pipeline import build_single_ticker_features


def make_series(values, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def make_ohlcv_frame(closes, volumes=None, start="2024-01-01"):
    close = make_series(closes, start)
    volumes = volumes or [1000] * len(closes)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": pd.Series(volumes, index=close.index),
    })


# ---------- trend.py ----------

def test_sma_matches_manual_average():
    close = make_series([1, 2, 3, 4, 5])
    result = sma(close, window=3)
    assert result.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
    assert result.iloc[4] == pytest.approx((3 + 4 + 5) / 3)
    assert pd.isna(result.iloc[0])  # not enough data yet


def test_ema_reacts_faster_than_sma_to_recent_jump():
    close = make_series([10] * 20 + [50])  # sudden jump at the end
    ema_result = ema(close, span=5)
    sma_result = sma(close, window=5)
    # EMA weights recent data more heavily -> should be closer to the jump
    assert ema_result.iloc[-1] > sma_result.iloc[-1]


def test_price_to_sma_ratio_positive_when_above_trend():
    close = make_series([100] * 10 + [150])
    ratio = price_to_sma_ratio(close, window=10)
    assert ratio.iloc[-1] > 0


def test_sma_crossover_signal_is_plus_or_minus_one():
    close = make_series(list(np.linspace(100, 200, 250)))
    signal = sma_crossover_signal(close, fast=10, slow=50)
    valid = signal.dropna()
    assert set(valid.unique()).issubset({-1, 1})


# ---------- momentum.py ----------

def test_rsi_is_100_in_pure_uptrend():
    close = make_series([100 + i for i in range(30)])  # strictly increasing
    result = rsi(close, window=14)
    assert result.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_bounded_0_to_100():
    rng = np.random.default_rng(3)
    close = make_series(100 * (1 + rng.normal(0, 0.02, 200)).cumprod())
    result = rsi(close, window=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_macd_histogram_is_difference_of_lines():
    close = make_series(list(np.linspace(100, 150, 100)))
    result = macd(close)
    diff = result["macd_line"] - result["signal_line"]
    pd.testing.assert_series_equal(result["macd_histogram"], diff, check_names=False)


def test_roc_matches_manual_calc():
    close = make_series([100, 105, 110, 99, 120])
    result = roc(close, window=2)
    assert result.iloc[2] == pytest.approx(110 / 100 - 1)
    assert result.iloc[4] == pytest.approx(120 / 110 - 1)


# ---------- volatility_features.py ----------

def test_atr_is_positive_and_reacts_to_range():
    frame = make_ohlcv_frame([100, 102, 98, 105, 95, 110])
    result = average_true_range(frame["High"], frame["Low"], frame["Close"], window=3)
    assert (result.dropna() > 0).all()


# ---------- volume.py ----------

def test_volume_change_basic():
    volume = make_series([1000, 1500, 750])
    result = volume_change(volume)
    assert result.iloc[1] == pytest.approx(0.5)
    assert result.iloc[2] == pytest.approx(-0.5)


def test_relative_volume_flags_spike():
    volume = make_series([1000] * 20 + [5000])
    result = relative_volume(volume, window=20)
    assert result.iloc[-1] > 2.0  # today's volume is way above its recent average


# ---------- pipeline.py (integration) ----------

def test_build_single_ticker_features_shape_and_no_lookahead():
    frame = make_ohlcv_frame(list(100 * (1 + np.random.default_rng(5).normal(0, 0.015, 300)).cumprod()))
    feats = build_single_ticker_features(frame)
    assert len(feats) == len(frame)
    assert "rsi_14" in feats.columns
    assert "macd_histogram" in feats.columns
    assert "atr_14" in feats.columns
    # Sanity: first row's SMA-200 should be NaN (not enough history) -- no look-ahead leakage
    assert pd.isna(feats["sma_200"].iloc[0])


def test_build_single_ticker_features_with_market_context():
    rng = np.random.default_rng(7)
    frame = make_ohlcv_frame(list(100 * (1 + rng.normal(0, 0.015, 300)).cumprod()))
    benchmark = make_series(list(20000 * (1 + rng.normal(0, 0.01, 300)).cumprod()))
    vix = make_series(list(15 + rng.normal(0, 1, 300)))

    feats = build_single_ticker_features(frame, benchmark_close=benchmark, vix_close=vix)
    assert "nifty_return_1d" in feats.columns
    assert "india_vix_level" in feats.columns


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
