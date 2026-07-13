"""
Phase 2 — EDA: bull/bear market regime detection.

Definition used (standard, widely-cited convention):
    BEAR market   -> benchmark is >= 20% below its most recent peak
    BULL market   -> benchmark is >= 20% above its most recent trough
    NEUTRAL       -> everything in between

This is computed on the BENCHMARK (NIFTY 50 index) series and can then be
joined onto individual stock series for regime-conditional analysis
(e.g. "how did HDFCBANK perform specifically during bear markets?").
"""

from __future__ import annotations

import pandas as pd

from eda.drawdown import drawdown_series


def classify_regimes(
    benchmark_close: pd.Series,
    bear_threshold: float = -0.20,
    bull_recovery_threshold: float = 0.20,
) -> pd.Series:
    """
    Returns a Series of {'bull', 'bear', 'neutral'} aligned to benchmark_close.index.

    Algorithm:
      - Track running peak and running trough-since-last-peak.
      - Once drawdown from peak <= bear_threshold -> regime = 'bear' from that
        point until price rallies >= bull_recovery_threshold off the trough
        reached during the bear phase, at which point regime flips to 'bull'.
      - Outside of an active bear/bull transition, regime = 'neutral'.
    """
    dd = drawdown_series(benchmark_close)
    regimes = []
    state = "neutral"
    trough_since_peak = benchmark_close.iloc[0]

    for date, price in benchmark_close.items():
        current_dd = dd.loc[date]

        if state == "neutral":
            if current_dd <= bear_threshold:
                state = "bear"
                trough_since_peak = price
            regimes.append(state if state == "bear" else "neutral")
            continue

        if state == "bear":
            trough_since_peak = min(trough_since_peak, price)
            rally_from_trough = price / trough_since_peak - 1
            if rally_from_trough >= bull_recovery_threshold:
                state = "bull"
            regimes.append(state)
            continue

        if state == "bull":
            # Bull regime persists until a fresh bear threshold breach from the new peak
            if current_dd <= bear_threshold:
                state = "bear"
                trough_since_peak = price
            regimes.append(state)

    return pd.Series(regimes, index=benchmark_close.index, name="regime")


def regime_period_summary(regime_series: pd.Series) -> pd.DataFrame:
    """Collapses the daily regime Series into contiguous labeled episodes with start/end dates."""
    rows = []
    current_regime = regime_series.iloc[0]
    start_date = regime_series.index[0]

    for date, regime in regime_series.items():
        if regime != current_regime:
            rows.append({
                "regime": current_regime,
                "start_date": start_date,
                "end_date": date,
                "duration_trading_days": len(regime_series.loc[start_date:date]) - 1,
            })
            current_regime = regime
            start_date = date

    rows.append({
        "regime": current_regime,
        "start_date": start_date,
        "end_date": regime_series.index[-1],
        "duration_trading_days": len(regime_series.loc[start_date:]) - 1,
    })
    return pd.DataFrame(rows)


def stock_return_by_regime(
    stock_close: pd.Series, regime_series: pd.Series
) -> pd.Series:
    """Mean daily return of a single stock, grouped by benchmark regime label."""
    aligned_regime = regime_series.reindex(stock_close.index, method="ffill")
    daily_ret = stock_close.pct_change()
    return daily_ret.groupby(aligned_regime).mean().rename("mean_daily_return")
