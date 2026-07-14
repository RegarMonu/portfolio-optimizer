"""
Phase 4 — Simple static-weight portfolio simulation.

This is DELIBERATELY simple: fixed target weights, rebalanced at a chosen
frequency, zero transaction costs. A configurable, general-purpose
backtesting engine (variable strategies, transaction cost models, slippage)
is explicitly Phase 5's job — this module exists only so Phase 4 can
produce an honest performance comparison between the classical allocation
methods, not to be the platform's final backtester.
"""

from __future__ import annotations

import pandas as pd


def simulate_portfolio(
    returns_matrix: pd.DataFrame,
    weights: pd.Series,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """
    returns_matrix: wide DataFrame (index=dates, columns=tickers) of daily returns.
    weights: target weights, reindexed to returns_matrix.columns.
    rebalance_freq: pandas offset alias ('ME' = month-end, 'W' = weekly,
                    None/'never' = buy-and-hold, no rebalancing at all).

    Returns the portfolio's daily return series.
    """
    weights = weights.reindex(returns_matrix.columns).fillna(0)
    weights = weights / weights.sum()

    if rebalance_freq is None or rebalance_freq == "never":
        # Buy-and-hold: weights drift with relative performance, never reset
        cumulative_asset_growth = (1 + returns_matrix).cumprod()
        portfolio_value = (cumulative_asset_growth * weights).sum(axis=1)
        portfolio_returns = portfolio_value.pct_change().fillna(portfolio_value.iloc[0] - 1)
        return portfolio_returns.rename("portfolio_return")

    # Periodic rebalancing: reset to target weights at each period boundary
    rebalance_dates = returns_matrix.resample(rebalance_freq).first().index
    daily_portfolio_returns = []
    current_weights = weights.copy()

    for date, day_returns in returns_matrix.iterrows():
        day_returns = day_returns.fillna(0)
        port_return = (current_weights * day_returns).sum()
        daily_portfolio_returns.append(port_return)

        # Drift weights forward with today's returns...
        grown = current_weights * (1 + day_returns)
        current_weights = grown / grown.sum()

        # ...then snap back to target on rebalance dates
        if date in rebalance_dates:
            current_weights = weights.copy()

    return pd.Series(daily_portfolio_returns, index=returns_matrix.index, name="portfolio_return")
