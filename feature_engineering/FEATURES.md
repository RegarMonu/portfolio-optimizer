# Feature Documentation — Phase 3

Every feature produced by `feature_engineering.pipeline.build_feature_panel`,
grouped as in the project roadmap. All are computed **causally** (only using
data up to and including the current date) — verified in
`tests/test_features.py::test_build_single_ticker_features_shape_and_no_lookahead`,
so none of these leak future information when used for modeling.

## Return Features
| Feature | Definition | Source |
|---|---|---|
| `daily_return` | 1-day simple % return | `eda.returns.daily_returns` |
| `weekly_return` | Return over the trading week, forward-filled to daily | `eda.returns.periodic_returns` |
| `monthly_return` | Return over the calendar month, forward-filled to daily | `eda.returns.periodic_returns` |

## Trend Features
| Feature | Definition |
|---|---|
| `sma_20`, `sma_50`, `sma_200` | Simple moving average over N days |
| `ema_12`, `ema_26` | Exponential moving average (same spans MACD uses internally) |
| `price_to_sma_50_ratio` | `Close / SMA(50) - 1` — how far price has stretched from its 50-day trend |
| `sma_crossover_50_200` | `+1` if SMA(50) > SMA(200) ("golden cross" regime), else `-1` |

## Momentum Features
| Feature | Definition |
|---|---|
| `rsi_14` | Relative Strength Index, Wilder's smoothing, 14-day window. 0-100 bounded. |
| `macd_line` | EMA(12) − EMA(26) |
| `macd_signal` | 9-day EMA of `macd_line` |
| `macd_histogram` | `macd_line − macd_signal`; sign changes are the classic MACD signal |
| `roc_10` | Rate of Change over 10 days: `Close_t / Close_{t-10} − 1` |

## Volatility Features
| Feature | Definition |
|---|---|
| `rolling_vol_21d` | Annualized rolling std-dev of daily returns, 21-day window (re-exported from Phase 2's `eda.volatility`) |
| `atr_14` | Average True Range, Wilder's smoothing, 14-day window — volatility measure that accounts for gaps, not just High−Low |

## Volume Features
| Feature | Definition |
|---|---|
| `volume_change` | Day-over-day % change in traded volume |
| `volume_sma_20` | 20-day simple moving average of volume |
| `relative_volume_20` | `Volume_t / volume_sma_20` — >1 flags unusually high activity |

## Market Features
| Feature | Definition |
|---|---|
| `nifty_return_1d` | Benchmark (^NSEI) 1-day return, broadcast onto every stock's rows for that date |
| `india_vix_level` | Raw India VIX level for that date |
| `india_vix_change` | Day-over-day % change in India VIX |
| `sector_relative_strength` | Stock's daily return minus its sector's equal-weighted daily return (Phase 2's `eda.sector.sector_returns`) — is this stock beating its own peer group? |

## Known limitations / not yet built
- No fundamental features (P/E, P/B, earnings growth) — this platform is
  currently price/volume-only. Would need a separate fundamentals data
  source (not in Phase 1's scope).
- `sector_relative_strength` is `NaN` for any ticker missing from
  `config.settings.SECTOR_MAP` (see Phase 1/2 notes) — check
  `eda.sector.unmapped_tickers()` before relying on it for a given universe.
- Warm-up period: the longest lookback is `sma_200`, so the first ~200 rows
  of any ticker's feature frame will have NaNs in slower-moving columns.
  Downstream consumers (Phase 4 optimizer, Phase 6 RL environment) should
  either drop or explicitly handle the warm-up window rather than silently
  training on partially-NaN rows.
