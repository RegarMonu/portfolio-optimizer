"""
Unit tests for eda.* modules — pure pandas/numpy logic, no network calls.
Run with: python -m pytest tests/test_eda.py -v
"""

import numpy as np
import pandas as pd
import pytest

from eda.returns import cagr, cumulative_return, daily_returns, returns_summary_table
from eda.volatility import rolling_volatility
from eda.drawdown import drawdown_series, max_drawdown, drawdown_periods, calmar_ratio
from eda.correlation import build_returns_matrix, correlation_matrix, average_pairwise_correlation
from eda.regime import classify_regimes, regime_period_summary
from eda.distribution import distribution_stats
from eda.sector import sector_returns, unmapped_tickers


def make_price_series(values, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=dates, name="Close")


def make_frame(values, start="2024-01-01"):
    close = make_price_series(values, start)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1000,
    })


# ---------- returns.py ----------

def test_daily_returns_basic():
    close = make_price_series([100, 110, 99])
    rets = daily_returns(close)
    assert rets.iloc[1] == pytest.approx(0.10)
    assert rets.iloc[2] == pytest.approx(-0.10, abs=1e-6)


def test_cumulative_return_starts_at_one():
    close = make_price_series([100, 150, 200])
    cum = cumulative_return(close)
    assert cum.iloc[0] == 1.0
    assert cum.iloc[-1] == 2.0


def test_cagr_doubling_in_one_year():
    close = make_price_series([100] * 1 + [200], start="2024-01-01")
    # Force exactly 252 trading days between the two points
    dates = pd.bdate_range("2024-01-01", periods=252)
    close = pd.Series([100] + [np.nan] * 250 + [200], index=dates).ffill()
    close.iloc[0] = 100
    close.iloc[-1] = 200
    result = cagr(close, periods_per_year=252)
    assert result == pytest.approx(1.0, abs=0.01)  # ~100% annual growth


# ---------- volatility.py ----------

def test_rolling_volatility_annualization():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, 300))
    vol = rolling_volatility(rets, window=21, annualize=True)
    # Annualized vol should be roughly std * sqrt(252) ~ 0.01 * 15.87 ≈ 0.159
    assert vol.dropna().mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.3)


# ---------- drawdown.py ----------

def test_drawdown_series_zero_at_new_highs():
    close = make_price_series([100, 110, 120])
    dd = drawdown_series(close)
    assert (dd == 0).all()


def test_max_drawdown_detects_known_drop():
    close = make_price_series([100, 120, 60, 90])
    mdd = max_drawdown(close)
    assert mdd == pytest.approx(60 / 120 - 1)  # -0.5


def test_drawdown_periods_detects_episode():
    close = make_price_series([100, 100, 70, 70, 105, 105])
    episodes = drawdown_periods(close, threshold=-0.10)
    assert len(episodes) == 1
    assert episodes.iloc[0]["depth"] == pytest.approx(0.70 - 1, abs=0.01)
    assert episodes.iloc[0]["recovery_date"] is not None


def test_calmar_ratio_runs():
    close = make_price_series([100, 90, 80, 110, 130])
    result = calmar_ratio(close)
    assert isinstance(result, float)


# ---------- correlation.py ----------

def test_build_returns_matrix_shape():
    frames = {"A": make_frame([100, 101, 102]), "B": make_frame([200, 198, 202])}
    matrix = build_returns_matrix(frames)
    assert list(matrix.columns) == ["A", "B"]
    assert len(matrix) == 2  # 3 prices -> 2 returns


def test_perfectly_correlated_series():
    a = [100, 110, 121, 133.1]
    frames = {"A": make_frame(a), "B": make_frame([2 * x for x in a])}
    corr = correlation_matrix(frames)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)


def test_average_pairwise_correlation_excludes_diagonal():
    # Non-constant growth rates so each return series has nonzero variance
    # (constant growth -> zero-variance returns -> correlation undefined, 0/0).
    prices = [100, 110, 108, 125, 119]
    frames = {"A": make_frame(prices), "B": make_frame(prices)}  # identical series
    corr = correlation_matrix(frames)
    avg = average_pairwise_correlation(corr)
    assert avg == pytest.approx(1.0, abs=1e-6)  # only off-diagonal A-B pair, which is 1.0


# ---------- regime.py ----------

def test_classify_regimes_detects_bear_then_bull():
    # Peak at 100, drop to 70 (-30%, breaches -20% bear threshold),
    # then rally back to 90 (+28.6% off trough, breaches +20% bull threshold)
    values = [100, 95, 80, 70, 75, 85, 90]
    close = make_price_series(values)
    regimes = classify_regimes(close, bear_threshold=-0.20, bull_recovery_threshold=0.20)
    assert regimes.iloc[3] == "bear"   # at the 70 trough
    assert regimes.iloc[-1] == "bull"  # after the rally


def test_regime_period_summary_covers_full_series():
    close = make_price_series([100, 95, 80, 70, 75, 85, 90])
    regimes = classify_regimes(close)
    summary = regime_period_summary(regimes)
    assert summary["start_date"].iloc[0] == regimes.index[0]
    assert summary["end_date"].iloc[-1] == regimes.index[-1]


# ---------- distribution.py ----------

def test_distribution_stats_normal_data():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.01, 2000))
    stats_dict = distribution_stats(rets)
    assert abs(stats_dict["skewness"]) < 0.2
    assert abs(stats_dict["kurtosis_excess"]) < 0.5
    assert stats_dict["var_95"] < 0  # 95% VaR should be a negative return


def test_distribution_stats_insufficient_data():
    result = distribution_stats(pd.Series([0.01, 0.02]))
    assert "error" in result


# ---------- sector.py ----------

def test_unmapped_tickers_detected():
    frames = {"FAKE_TICKER_XYZ.NS": make_frame([100, 101])}
    unmapped = unmapped_tickers(frames)
    assert "FAKE_TICKER_XYZ.NS" in unmapped


def test_sector_returns_groups_known_tickers():
    frames = {
        "TCS.NS": make_frame([100, 105, 110]),      # IT
        "INFY.NS": make_frame([200, 210, 220]),     # IT
        "HDFCBANK.NS": make_frame([300, 295, 305]), # Financials
    }
    sec_rets = sector_returns(frames)
    assert "IT" in sec_rets.columns
    assert "Financials" in sec_rets.columns


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
