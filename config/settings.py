"""
Central configuration for the Indian Portfolio Optimization Platform.

Keep ALL tunable constants here so downstream modules never hardcode
paths, tickers, or dates. This is the single source of truth for Phase 1+.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data_store"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Universe: NIFTY 50 constituents (Yahoo Finance NSE tickers, ".NS" suffix)
# NOTE: Index constituents change over time (rebalanced semi-annually by
# NSE Indices Ltd). This list should be refreshed periodically — see
# `data_layer/universe.py::refresh_nifty50_list` (Phase 1 stretch goal).
# Last manually verified: Jan 2026.
# --------------------------------------------------------------------------
NIFTY_50_TICKERS: list[str] = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BPCL.NS",
    "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "LTIM.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS",
    "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS", "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS",
]

BENCHMARK_TICKER = "^NSEI"  # NIFTY 50 index itself
INDIA_VIX_TICKER = "^INDIAVIX"

# --------------------------------------------------------------------------
# Sector mapping (NSE broad sector classification) — used in Phase 2 EDA
# for sector-wise performance breakdowns. Manually curated, Jan 2026.
# Known limitation: a few conglomerates (e.g. Adani, Reliance) are
# multi-sector in reality; this maps each to its PRIMARY reported segment.
# --------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    "ADANIENT.NS": "Diversified", "ADANIPORTS.NS": "Infrastructure",
    "APOLLOHOSP.NS": "Healthcare", "ASIANPAINT.NS": "Consumer Goods",
    "AXISBANK.NS": "Financials", "BAJAJ-AUTO.NS": "Automobile",
    "BAJFINANCE.NS": "Financials", "BAJAJFINSV.NS": "Financials",
    "BEL.NS": "Industrials", "BPCL.NS": "Energy",
    "BHARTIARTL.NS": "Telecom", "BRITANNIA.NS": "Consumer Goods",
    "CIPLA.NS": "Healthcare", "COALINDIA.NS": "Energy",
    "DIVISLAB.NS": "Healthcare", "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Automobile", "GRASIM.NS": "Materials",
    "HCLTECH.NS": "IT", "HDFCBANK.NS": "Financials",
    "HDFCLIFE.NS": "Financials", "HEROMOTOCO.NS": "Automobile",
    "HINDALCO.NS": "Materials", "HINDUNILVR.NS": "Consumer Goods",
    "ICICIBANK.NS": "Financials", "ITC.NS": "Consumer Goods",
    "INDUSINDBK.NS": "Financials", "INFY.NS": "IT",
    "JSWSTEEL.NS": "Materials", "KOTAKBANK.NS": "Financials",
    "LT.NS": "Infrastructure", "LTIM.NS": "IT",
    "M&M.NS": "Automobile", "MARUTI.NS": "Automobile",
    "NTPC.NS": "Energy", "NESTLEIND.NS": "Consumer Goods",
    "ONGC.NS": "Energy", "POWERGRID.NS": "Energy",
    "RELIANCE.NS": "Energy", "SBILIFE.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials", "SBIN.NS": "Financials",
    "SUNPHARMA.NS": "Healthcare", "TCS.NS": "IT",
    "TATACONSUM.NS": "Consumer Goods", "TATAMOTORS.NS": "Automobile",
    "TATASTEEL.NS": "Materials", "TECHM.NS": "IT",
    "TITAN.NS": "Consumer Goods", "ULTRACEMCO.NS": "Materials",
    "WIPRO.NS": "IT",
}

# --------------------------------------------------------------------------
# Date ranges
# --------------------------------------------------------------------------
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = str(date.today())

# --------------------------------------------------------------------------
# Data fetch behavior
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FetchConfig:
    max_retries: int = 4
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: int = 30
    batch_size: int = 10          # tickers per batch download call
    inter_batch_sleep_seconds: float = 1.5
    min_acceptable_rows: int = 100  # flag a ticker as suspect below this


@dataclass(frozen=True)
class CleaningConfig:
    max_consecutive_missing_days: int = 5   # forward-fill up to this many days
    price_jump_flag_threshold: float = 0.35  # |daily return| above this -> flagged, not dropped
    drop_tickers_missing_pct_above: float = 0.30  # drop ticker if >30% rows missing post-fill


FETCH_CFG = FetchConfig()
CLEAN_CFG = CleaningConfig()

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("PORTFOLIO_LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "pipeline.log"
