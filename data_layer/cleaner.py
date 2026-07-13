"""
Phase 1 — Data cleaning.

Handles:
    1. Missing trading-day rows (reindex to a shared trading calendar).
    2. Short gaps -> forward-filled (bounded, so we never silently paper
       over a genuinely delisted/halted stock).
    3. Long gaps / excessive missingness -> ticker is DROPPED, not filled,
       and the drop is logged so it's auditable, not silent.
    4. Extreme single-day price jumps -> FLAGGED (extra column), not removed.
       These are often legitimate (bonus issues not caught by auto_adjust,
       circuit-breaker days) so we never delete price rows automatically —
       that decision needs a human/downstream policy.

Output schema per ticker (columns): Open, High, Low, Close, Volume,
    is_filled (bool), is_price_jump_flagged (bool)
"""

from __future__ import annotations

import pandas as pd

from config.logging_config import get_logger
from config.settings import CLEAN_CFG

logger = get_logger(__name__)


def build_trading_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all observed trading dates across the universe — our reference calendar."""
    all_dates = pd.DatetimeIndex([])
    for frame in frames.values():
        if frame is not None and not frame.empty:
            all_dates = all_dates.union(frame.index)
    return all_dates.sort_values()


def clean_single_ticker(
    ticker: str,
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None

    frame = frame.copy()
    frame = frame[~frame.index.duplicated(keep="last")]

    original_len = len(frame)
    reindexed = frame.reindex(calendar)
    reindexed["is_filled"] = reindexed["Close"].isna()

    # Bounded forward-fill: never fill a gap longer than max_consecutive_missing_days
    missing_mask = reindexed["Close"].isna()
    fill_group = (~missing_mask).cumsum()
    gap_lengths = missing_mask.groupby(fill_group).cumsum()

    fillable = missing_mask & (gap_lengths <= CLEAN_CFG.max_consecutive_missing_days)
    reindexed.loc[fillable, ["Open", "High", "Low", "Close"]] = (
        reindexed[["Open", "High", "Low", "Close"]].ffill().loc[fillable]
    )
    reindexed.loc[fillable, "Volume"] = 0  # no real volume traded on a filled day

    still_missing_pct = reindexed["Close"].isna().mean()
    if still_missing_pct > CLEAN_CFG.drop_tickers_missing_pct_above:
        logger.warning(
            "%s: dropping ticker — %.1f%% of rows unfillable (long gaps / delisting suspected)",
            ticker, still_missing_pct * 100,
        )
        return None

    reindexed = reindexed.dropna(subset=["Close"])

    # Flag (don't drop) extreme single-day moves
    daily_return = reindexed["Close"].pct_change()
    reindexed["is_price_jump_flagged"] = daily_return.abs() > CLEAN_CFG.price_jump_flag_threshold
    n_flagged = int(reindexed["is_price_jump_flagged"].sum())
    if n_flagged:
        logger.info("%s: flagged %d extreme single-day price moves for review", ticker, n_flagged)

    logger.info(
        "%s: cleaned %d -> %d rows (%d filled, %d dropped for missingness)",
        ticker, original_len, len(reindexed), int(reindexed["is_filled"].sum()),
        len(calendar) - len(reindexed) - (len(calendar) - original_len),
    )
    return reindexed


def clean_universe(raw_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    calendar = build_trading_calendar(raw_frames)
    cleaned: dict[str, pd.DataFrame] = {}

    for ticker, frame in raw_frames.items():
        result = clean_single_ticker(ticker, frame, calendar)
        if result is not None:
            cleaned[ticker] = result

    logger.info("Cleaning complete: %d/%d tickers retained", len(cleaned), len(raw_frames))
    return cleaned
