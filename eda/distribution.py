"""
Phase 2 — EDA: distribution of returns (skewness, kurtosis, tail behavior).

Why this matters for Phase 4+ (Markowitz etc.): mean-variance optimization
implicitly assumes returns are roughly normal. Indian equities — especially
mid/small caps — are known to show negative skew and fat tails (kurtosis > 3),
i.e. crashes are more frequent/severe than a normal distribution would predict.
This module quantifies that so later phases can decide whether to
risk-adjust for it (e.g. CVaR-based optimization instead of pure variance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def distribution_stats(returns: pd.Series) -> dict:
    clean = returns.dropna()
    if len(clean) < 10:
        return {"error": "insufficient data"}

    jb_stat, jb_pvalue = stats.jarque_bera(clean)

    return {
        "mean": clean.mean(),
        "std": clean.std(),
        "skewness": stats.skew(clean),
        "kurtosis_excess": stats.kurtosis(clean),  # 0 = normal; >0 = fat tails
        "min": clean.min(),
        "max": clean.max(),
        "var_95": clean.quantile(0.05),   # historical 1-day 95% VaR
        "var_99": clean.quantile(0.01),
        "cvar_95": clean[clean <= clean.quantile(0.05)].mean(),  # expected shortfall
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "is_normal_at_5pct": jb_pvalue > 0.05,
        "n_observations": len(clean),
    }


def distribution_summary_table(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, frame in price_frames.items():
        rets = frame["Close"].pct_change().dropna()
        if len(rets) < 10:
            continue
        stats_dict = distribution_stats(rets)
        stats_dict["ticker"] = ticker
        rows.append(stats_dict)
    return pd.DataFrame(rows).set_index("ticker").sort_values("skewness")
