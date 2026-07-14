"""
Phase 5 — Configurable Backtesting Engine.

This generalizes Phase 4's `optimization/simulate.py` (which was
deliberately minimal — fixed weights, no costs) into a real, reusable
engine:
    * Configurable backtest period (via the returns_matrix passed in)
    * Configurable rebalancing frequency
    * Real transaction cost simulation (backtesting.transaction_costs)
    * Any Strategy (Phase 4 method or future RL policy) plugs in identically
    * Full portfolio tracking: daily returns, weights history, turnover,
      costs paid — nothing is thrown away, everything is auditable after
      the fact.

No-look-ahead guarantee: at each date, a strategy only ever receives
`returns_matrix.iloc[:i]` — data strictly BEFORE that date. This is
enforced by the engine itself, not left to each strategy to get right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.logging_config import get_logger
from backtesting.strategy import Strategy
from backtesting.transaction_costs import transaction_cost, turnover as compute_turnover

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    strategy_name: str
    daily_returns: pd.Series
    weights_history: pd.DataFrame       # index=rebalance dates, columns=tickers
    turnover_history: pd.Series         # index=rebalance dates
    transaction_costs: pd.Series        # index=rebalance dates, cost as fraction of portfolio value
    rebalance_failures: list = field(default_factory=list)  # (date, error message) pairs

    @property
    def cumulative_value(self) -> pd.Series:
        return (1 + self.daily_returns).cumprod().rename("cumulative_value")

    @property
    def total_transaction_cost_drag(self) -> float:
        """Total return given up to transaction costs over the whole backtest."""
        return float(self.transaction_costs.sum())


class BacktestEngine:
    def __init__(
        self,
        returns_matrix: pd.DataFrame,
        rebalance_freq: str = "ME",
        transaction_cost_bps: float = 10.0,
        min_lookback_days: int = 60,
    ):
        """
        returns_matrix: wide DataFrame, index=dates, columns=tickers, of daily returns.
        rebalance_freq: pandas offset alias ('ME'=month-end, 'W'=weekly,
            'QE'=quarter-end), or None for buy-and-hold (rebalance once at start only).
        transaction_cost_bps: cost per unit of turnover, in basis points.
        min_lookback_days: minimum history required before the FIRST
            rebalance can happen; before that, the portfolio holds cash
            (zero return) rather than trading on insufficient data.
        """
        self.returns_matrix = returns_matrix
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps
        self.min_lookback_days = min_lookback_days

    def run(self, strategy: Strategy) -> BacktestResult:
        dates = self.returns_matrix.index
        rebalance_dates = (
            set(self.returns_matrix.resample(self.rebalance_freq).first().index)
            if self.rebalance_freq else set()
        )

        current_weights = pd.Series(0.0, index=self.returns_matrix.columns)
        daily_returns, weights_records, turnover_records, cost_records = [], {}, {}, {}
        failures = []
        initialized = False

        for i, date in enumerate(dates):
            history = self.returns_matrix.iloc[:i]  # strictly before `date` -- no look-ahead
            should_rebalance = (not initialized) or (date in rebalance_dates)
            cost_today = 0.0

            if should_rebalance and len(history) >= self.min_lookback_days:
                try:
                    target = strategy.compute_weights(history)
                    target = target.reindex(self.returns_matrix.columns).fillna(0)
                    target_sum = target.sum()
                    if target_sum <= 0:
                        raise ValueError(f"strategy returned non-positive total weight ({target_sum})")
                    target = target / target_sum

                    t = compute_turnover(current_weights, target)
                    cost_today = transaction_cost(t, self.transaction_cost_bps)

                    current_weights = target
                    initialized = True
                    weights_records[date] = current_weights.copy()
                    turnover_records[date] = t
                    cost_records[date] = cost_today
                except Exception as exc:  # noqa: BLE001 -- strategy errors are heterogeneous
                    logger.warning(
                        "%s: rebalance failed on %s (%s) -- holding previous weights",
                        strategy.name, date.date(), exc,
                    )
                    failures.append((date, str(exc)))

            day_ret = self.returns_matrix.loc[date].fillna(0)
            port_ret = (current_weights * day_ret).sum() - cost_today
            daily_returns.append(port_ret)

            if initialized:
                grown = current_weights * (1 + day_ret)
                total = grown.sum()
                if total > 0:
                    current_weights = grown / total  # weights drift with performance until next rebalance

        if failures:
            logger.warning("%s: %d/%d rebalances failed and fell back to holding weights",
                            strategy.name, len(failures), len(rebalance_dates) or 1)

        return BacktestResult(
            strategy_name=strategy.name,
            daily_returns=pd.Series(daily_returns, index=dates, name="portfolio_return"),
            weights_history=pd.DataFrame(weights_records).T,
            turnover_history=pd.Series(turnover_records, name="turnover"),
            transaction_costs=pd.Series(cost_records, name="transaction_cost"),
            rebalance_failures=failures,
        )

    def run_many(self, strategies: list[Strategy]) -> dict[str, BacktestResult]:
        return {s.name: self.run(s) for s in strategies}
