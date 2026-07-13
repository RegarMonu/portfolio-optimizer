"""
Phase 3 — Feature Engineering: pipeline orchestrator.

Combines return features (Phase 2's eda.returns), trend, momentum,
volatility, volume, and market-wide features into ONE feature DataFrame
per ticker, then optionally stitches all tickers into a single long-format
panel (ticker, date, feature columns) ready for either classical models
(Phase 4) or RL state construction (Phase 6).

Run via:
    python -m feature_engineering.pipeline                 # real processed data
    python -m feature_engineering.pipeline --synthetic       # demo data, no network
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.logging_config import get_logger
from config.settings import (
    BENCHMARK_TICKER,
    INDIA_VIX_TICKER,
    NIFTY_50_TICKERS,
    PROCESSED_DATA_DIR,
)
from data_layer.storage import load_universe
from eda.returns import daily_returns, periodic_returns
from eda.sector import sector_returns
from feature_engineering.market import (
    benchmark_return_feature,
    india_vix_change,
    india_vix_level,
    sector_relative_strength,
)
from feature_engineering.momentum import macd, rsi, roc
from feature_engineering.trend import ema, price_to_sma_ratio, sma, sma_crossover_signal
from feature_engineering.volatility_features import average_true_range, rolling_volatility
from feature_engineering.volume import relative_volume, volume_change, volume_moving_average

logger = get_logger(__name__)

FEATURES_DIR = PROCESSED_DATA_DIR.parent / "features"


def build_single_ticker_features(
    frame: pd.DataFrame,
    benchmark_close: pd.Series | None = None,
    vix_close: pd.Series | None = None,
    sector_return_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    frame: this ticker's OHLCV DataFrame (Open, High, Low, Close, Volume).
    Returns a DataFrame of engineered features aligned to `frame`'s index.
    """
    close, high, low, volume = frame["Close"], frame["High"], frame["Low"], frame["Volume"]
    feats = {}

    # --- Return features ---
    feats["daily_return"] = daily_returns(close)
    feats["weekly_return"] = periodic_returns(close, freq="W").reindex(close.index, method="ffill")
    feats["monthly_return"] = periodic_returns(close, freq="ME").reindex(close.index, method="ffill")

    # --- Trend features ---
    for w in (20, 50, 200):
        feats[f"sma_{w}"] = sma(close, w)
    feats["ema_12"] = ema(close, 12)
    feats["ema_26"] = ema(close, 26)
    feats["price_to_sma_50_ratio"] = price_to_sma_ratio(close, 50)
    feats["sma_crossover_50_200"] = sma_crossover_signal(close, 50, 200)

    # --- Momentum features ---
    feats["rsi_14"] = rsi(close, 14)
    macd_df = macd(close)
    feats["macd_line"] = macd_df["macd_line"]
    feats["macd_signal"] = macd_df["signal_line"]
    feats["macd_histogram"] = macd_df["macd_histogram"]
    feats["roc_10"] = roc(close, 10)

    # --- Volatility features ---
    daily_ret = feats["daily_return"]
    feats["rolling_vol_21d"] = rolling_volatility(daily_ret, window=21)
    feats["atr_14"] = average_true_range(high, low, close, window=14)

    # --- Volume features ---
    feats["volume_change"] = volume_change(volume)
    feats["volume_sma_20"] = volume_moving_average(volume, 20)
    feats["relative_volume_20"] = relative_volume(volume, 20)

    # --- Market features (broadcast onto this ticker's dates) ---
    if benchmark_close is not None:
        feats["nifty_return_1d"] = benchmark_return_feature(benchmark_close).reindex(close.index)
    if vix_close is not None:
        feats["india_vix_level"] = india_vix_level(vix_close).reindex(close.index)
        feats["india_vix_change"] = india_vix_change(vix_close).reindex(close.index)
    if sector_return_series is not None:
        feats["sector_relative_strength"] = sector_relative_strength(close, sector_return_series)

    return pd.DataFrame(feats, index=close.index)


def build_feature_panel(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds features for every ticker (excluding benchmark/VIX, which are
    inputs rather than modeled assets) and stacks them into one long-format
    panel: MultiIndex (ticker, date) x feature columns.
    """
    benchmark_close = price_frames.get(BENCHMARK_TICKER, {}).get("Close") \
        if BENCHMARK_TICKER in price_frames else None
    if BENCHMARK_TICKER in price_frames:
        benchmark_close = price_frames[BENCHMARK_TICKER]["Close"]
    vix_close = price_frames[INDIA_VIX_TICKER]["Close"] if INDIA_VIX_TICKER in price_frames else None

    stock_frames = {
        t: f for t, f in price_frames.items() if t not in (BENCHMARK_TICKER, INDIA_VIX_TICKER)
    }
    sec_returns = sector_returns(stock_frames)

    panels = []
    for ticker, frame in stock_frames.items():
        from config.settings import SECTOR_MAP
        sector = SECTOR_MAP.get(ticker)
        sector_series = sec_returns[sector] if sector in sec_returns.columns else None

        feats = build_single_ticker_features(
            frame, benchmark_close=benchmark_close, vix_close=vix_close,
            sector_return_series=sector_series,
        )
        feats["ticker"] = ticker
        panels.append(feats)

        n_valid = feats.dropna().shape[0]
        logger.info(
            "%s: built %d features, %d/%d rows fully populated (post warm-up)",
            ticker, feats.shape[1] - 1, n_valid, len(feats),
        )

    full_panel = pd.concat(panels)
    full_panel = full_panel.set_index("ticker", append=True).reorder_levels([1, 0]).sort_index()
    full_panel.index.set_names(["ticker", "date"], inplace=True)
    return full_panel


def save_feature_panel(panel: pd.DataFrame) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / "feature_panel.parquet"
    panel.to_parquet(path, compression="snappy")
    logger.info("Feature panel saved to %s (%d rows, %d columns)", path, *panel.shape)


def _make_synthetic_universe_for_features():
    """Reuse Phase 2's synthetic generator so this module is independently testable."""
    from eda.report import _make_synthetic_universe
    return _make_synthetic_universe(n_tickers=8, n_days=400)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 feature engineering pipeline")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.synthetic:
        universe = _make_synthetic_universe_for_features()
    else:
        tickers = NIFTY_50_TICKERS + [BENCHMARK_TICKER, INDIA_VIX_TICKER]
        universe = load_universe(tickers)
        if BENCHMARK_TICKER not in universe:
            raise RuntimeError(
                "No processed benchmark data found. Run `python -m data_layer.pipeline` "
                "first, or use --synthetic to demo."
            )

    panel = build_feature_panel(universe)
    save_feature_panel(panel)
    print(f"Feature panel shape: {panel.shape}")
    print(panel.tail())
