"""
Phase 1 — Data Layer: historical OHLCV ingestion for NSE-listed equities.

Design goals:
    * Idempotent: re-running for a date range that's already cached is a no-op
      unless `force_refresh=True`.
    * Resilient: retries with exponential backoff on transient network/API errors.
    * Auditable: every fetch run is logged with row counts and any tickers
      that came back suspiciously short or empty.
    * Adjusted prices: uses yfinance's `auto_adjust=True` so splits/dividends
      are baked into OHLC directly (Phase 0 "corporate actions" requirement).

This module has ONE external dependency surface: `yfinance`. If the data
vendor changes in the future, only this file (and `universe.py`) should need
to change — everything downstream consumes the standardized parquet schema
defined in `storage.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import time
from config.logging_config import get_logger
from config.settings import FETCH_CFG

logger = get_logger(__name__)

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yfinance is required for the data layer. Install with: "
        "pip install yfinance --break-system-packages"
    ) from exc

@dataclass
class FetchResult:
    ticker: str
    frame: pd.DataFrame | None
    ok: bool
    reason: str = ""


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _validate_frame(ticker: str, frame: pd.DataFrame) -> FetchResult:
    if frame is None or frame.empty:
        return FetchResult(ticker, None, False, "empty response")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_cols:
        return FetchResult(ticker, None, False, f"missing columns: {missing_cols}")

    if len(frame) < FETCH_CFG.min_acceptable_rows:
        return FetchResult(
            ticker, frame, True,
            f"WARNING: only {len(frame)} rows (< {FETCH_CFG.min_acceptable_rows})",
        )

    return FetchResult(ticker, frame, True)


def fetch_single_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> FetchResult:
    """Fetch one ticker with retry + exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(1, FETCH_CFG.max_retries + 1):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,       # bakes in split/dividend adjustments
                progress=False,
                threads=False,
                timeout=FETCH_CFG.request_timeout_seconds,
            )
            # yfinance sometimes returns MultiIndex columns even for a single ticker
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)

            result = _validate_frame(ticker, frame)
            if result.reason.startswith("WARNING"):
                logger.warning("%s: %s", ticker, result.reason)
            return result

        except Exception as exc:  # noqa: BLE001 — vendor errors are heterogeneous
            last_error = exc
            wait = FETCH_CFG.retry_backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Fetch failed for %s (attempt %d/%d): %s. Retrying in %.1fs",
                ticker, attempt, FETCH_CFG.max_retries, exc, wait,
            )
            time.sleep(wait)

    return FetchResult(ticker, None, False, f"exhausted retries: {last_error}")


def fetch_universe(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> dict[str, FetchResult]:
    """
    Fetch a list of tickers in small batches (politeness sleep between
    batches to avoid vendor rate-limiting), returning a ticker -> FetchResult map.
    """
    results: dict[str, FetchResult] = {}
    batch_size = FETCH_CFG.batch_size

    start_time = time.time()
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info("Fetching batch %d-%d of %d: %s", i, i + len(batch), len(tickers), batch)

        for ticker in batch:
            results[ticker] = fetch_single_ticker(ticker, start, end, interval)

        if i + batch_size < len(tickers):
            time.sleep(FETCH_CFG.inter_batch_sleep_seconds)

    elapsed = time.time() - start_time
    logger.info("Fetch completed in %.1fs", elapsed)
    ok_count = sum(1 for r in results.values() if r.ok)
    logger.info("Fetch complete: %d/%d tickers OK", ok_count, len(tickers))
    return results
