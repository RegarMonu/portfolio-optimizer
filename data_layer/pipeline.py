"""
Phase 1 — End-to-end orchestrator: fetch -> clean -> store -> manifest.

Usage:
    python -m data_layer.pipeline
    python -m data_layer.pipeline --start 2018-01-01 --end 2026-07-01
    python -m data_layer.pipeline --tickers RELIANCE.NS,TCS.NS --force-refresh
"""

from __future__ import annotations

import argparse

from config.logging_config import get_logger
from config.settings import (
    BENCHMARK_TICKER,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    INDIA_VIX_TICKER,
    NIFTY_50_TICKERS,
)
from data_layer.cleaner import clean_universe
from data_layer.fetcher import fetch_universe
from data_layer.storage import save_processed, save_raw, write_manifest

logger = get_logger(__name__)


def run_pipeline(
    tickers: list[str],
    start: str,
    end: str,
    include_market_indicators: bool = True,
) -> None:
    universe = list(tickers)
    if include_market_indicators:
        universe = universe + [BENCHMARK_TICKER, INDIA_VIX_TICKER]

    logger.info("=== Phase 1 pipeline starting: %d symbols, %s -> %s ===", len(universe), start, end)

    # 1. Fetch
    fetch_results = fetch_universe(universe, start, end)
    raw_frames = {}
    failed = []

    for ticker, result in fetch_results.items():
        if result.ok and result.frame is not None:
            save_raw(ticker, result.frame)
            raw_frames[ticker] = result.frame
        elif not result.ok:
            failed.append(ticker)

    if failed:
        logger.warning("Failed to fetch %d tickers: %s", len(failed), failed)

    # 2. Clean
    cleaned_frames = clean_universe(raw_frames)

    # 3. Store
    for ticker, frame in cleaned_frames.items():
        save_processed(ticker, frame)

    # 4. Manifest (audit trail)
    dropped = sorted(set(raw_frames.keys()) - set(cleaned_frames.keys())) + failed
    write_manifest(
        tickers_requested=universe,
        tickers_succeeded=sorted(cleaned_frames.keys()),
        tickers_dropped=sorted(set(dropped)),
        start=start,
        end=end,
    )

    logger.info(
        "=== Phase 1 pipeline complete: %d/%d symbols usable ===",
        len(cleaned_frames), len(universe),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 data ingestion pipeline")
    parser.add_argument("--tickers", type=str, default=None,
                         help="Comma-separated tickers. Defaults to full NIFTY 50.")
    parser.add_argument("--start", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--end", type=str, default=DEFAULT_END_DATE)
    parser.add_argument("--no-market-indicators", action="store_true",
                         help="Skip fetching ^NSEI / ^INDIAVIX")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tickers = args.tickers.split(",") if args.tickers else NIFTY_50_TICKERS
    run_pipeline(
        tickers=tickers,
        start=args.start,
        end=args.end,
        include_market_indicators=not args.no_market_indicators,
    )
