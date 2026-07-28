from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf, pacf


def stationarity_test(values: pd.Series) -> dict[str, Any]:
    statistic, p_value, lags, observations, critical_values, _ = adfuller(values.dropna(), autolag="AIC")
    return {"test": "Augmented Dickey-Fuller", "statistic": float(statistic), "p_value": float(p_value), "used_lags": int(lags), "observations": int(observations), "critical_values": {key: float(value) for key, value in critical_values.items()}, "reject_unit_root_at_5pct": bool(p_value < 0.05)}


def autocorrelation_diagnostics(values: pd.Series, max_lags: int = 20) -> dict[str, Any]:
    clean = values.dropna().to_numpy(float)
    safe_lags = max(1, min(max_lags, (len(clean) - 1) // 2))
    autocorrelations = acf(clean, nlags=safe_lags, fft=True)
    partial = pacf(clean, nlags=safe_lags, method="ywm")
    squared_box = acorr_ljungbox(pd.Series(clean**2), lags=[min(10, safe_lags)], return_df=True).iloc[0]
    return {"max_lags": safe_lags, "acf": [float(value) for value in autocorrelations], "pacf": [float(value) for value in partial], "ljung_box_squared_returns": {"statistic": float(squared_box["lb_stat"]), "p_value": float(squared_box["lb_pvalue"])}}


def distribution_diagnostics(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().to_numpy(float)
    statistic, p_value = normaltest(clean) if len(clean) >= 20 else (np.nan, np.nan)
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean, ddof=1)), "skewness": float(pd.Series(clean).skew()), "kurtosis": float(pd.Series(clean).kurt()), "normality_test": {"test": "D'Agostino K-squared", "statistic": float(statistic) if np.isfinite(statistic) else None, "p_value": float(p_value) if np.isfinite(p_value) else None}}

