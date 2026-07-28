from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    return float(-returns.quantile(1.0 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.99) -> float:
    mean = float(returns.mean())
    standard_deviation = float(returns.std(ddof=1))
    return float(-(mean + norm.ppf(1.0 - confidence) * standard_deviation))


def monte_carlo_var(returns: pd.Series, confidence: float = 0.99, simulations: int = 10000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    simulations_values = rng.normal(float(returns.mean()), float(returns.std(ddof=1)), simulations)
    return float(-np.quantile(simulations_values, 1.0 - confidence))


def forecast_historical_var(history: pd.Series, confidence: float = 0.99, window: int = 252) -> float:
    return historical_var(history.tail(window), confidence)


def forecast_parametric_var(history: pd.Series, confidence: float = 0.99, window: int = 252) -> float:
    return parametric_var(history.tail(window), confidence)


def garch_var(mean: float, volatility: float, confidence: float = 0.99) -> float:
    return float(-(mean + norm.ppf(1.0 - confidence) * volatility))


def backtest_var(realized_returns: pd.Series, var_forecast: pd.Series, confidence: float = 0.99) -> dict[str, Any]:
    aligned = pd.concat([realized_returns.rename("returns"), var_forecast.rename("var")], axis=1).dropna()
    exceptions = aligned["returns"] < -aligned["var"]
    count = int(exceptions.sum())
    observations = len(aligned)
    expected_rate = 1.0 - confidence
    observed_rate = count / observations if observations else np.nan
    if observations and 0 < count < observations:
        likelihood_unrestricted = (1 - observed_rate) ** (observations - count) * observed_rate**count
        likelihood_restricted = (1 - expected_rate) ** (observations - count) * expected_rate**count
        kupiec_statistic = float(-2.0 * np.log(likelihood_restricted / likelihood_unrestricted))
        kupiec_p_value = float(1.0 - chi2.cdf(kupiec_statistic, 1))
    else:
        kupiec_statistic, kupiec_p_value = None, None
    return {"observations": observations, "exceptions": count, "expected_exception_rate": expected_rate, "observed_exception_rate": observed_rate, "kupiec_lr": kupiec_statistic, "kupiec_p_value": kupiec_p_value}

