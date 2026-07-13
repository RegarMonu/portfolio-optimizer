"""
Phase 1 — Storage layer.

Uses Parquet (columnar, compressed, fast) instead of CSV:
    * ~5-10x smaller on disk than CSV for OHLCV data.
    * Preserves dtypes (no re-parsing dates/floats on every load).
    * Supports predicate pushdown if this ever moves to a real data lake.

Layout:
    data_store/raw/<TICKER>.parquet          — raw vendor data, untouched
    data_store/processed/<TICKER>.parquet    — cleaned data (see cleaner.py)
    data_store/processed/_manifest.json      — run metadata (audit trail)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config.logging_config import get_logger
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = get_logger(__name__)

MANIFEST_PATH = PROCESSED_DATA_DIR / "_manifest.json"


def _safe_filename(ticker: str) -> str:
    return ticker.replace("^", "INDEX_").replace("&", "AND")


def save_raw(ticker: str, frame: pd.DataFrame) -> None:
    path = RAW_DATA_DIR / f"{_safe_filename(ticker)}.parquet"
    frame.to_parquet(path, compression="snappy")


def save_processed(ticker: str, frame: pd.DataFrame) -> None:
    path = PROCESSED_DATA_DIR / f"{_safe_filename(ticker)}.parquet"
    frame.to_parquet(path, compression="snappy")


def load_processed(ticker: str) -> pd.DataFrame | None:
    path = PROCESSED_DATA_DIR / f"{_safe_filename(ticker)}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for ticker in tickers:
        frame = load_processed(ticker)
        if frame is not None:
            out[ticker] = frame
        else:
            logger.warning("No processed data found on disk for %s", ticker)
    return out


def write_manifest(
    tickers_requested: list[str],
    tickers_succeeded: list[str],
    tickers_dropped: list[str],
    start: str,
    end: str,
) -> None:
    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start, "end": end},
        "tickers_requested": tickers_requested,
        "tickers_succeeded": tickers_succeeded,
        "tickers_dropped": tickers_dropped,
        "success_rate": (
            len(tickers_succeeded) / len(tickers_requested) if tickers_requested else 0
        ),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", MANIFEST_PATH)


def load_manifest() -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH) as f:
        return json.load(f)
