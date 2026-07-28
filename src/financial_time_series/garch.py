from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GARCHResult:
    backend: str
    omega: float
    alpha: float
    beta: float
    mean: float
    conditional_volatility: np.ndarray
    forecast_variance: np.ndarray
    fitted_model: object | None = None


def _fallback_fit(returns_percent: np.ndarray, horizon: int) -> GARCHResult:
    values = np.asarray(returns_percent, dtype=float)
    variance = float(np.var(values)) or 1e-6

    def recursion(params: np.ndarray) -> np.ndarray:
        omega, alpha, beta = params
        sigma2 = np.empty(len(values))
        sigma2[0] = variance
        residuals = values - np.mean(values)
        for index in range(1, len(values)):
            sigma2[index] = omega + alpha * residuals[index - 1] ** 2 + beta * sigma2[index - 1]
        return np.maximum(sigma2, 1e-10)

    def objective(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        sigma2 = recursion(params)
        residuals = values - np.mean(values)
        return float(0.5 * np.sum(np.log(sigma2) + residuals**2 / sigma2))

    optimized = minimize(objective, x0=np.array([0.05 * variance, 0.05, 0.9]), method="Nelder-Mead", options={"maxiter": 4000})
    omega, alpha, beta = optimized.x if optimized.success else (0.05 * variance, 0.05, 0.9)
    sigma2 = recursion(np.array([omega, alpha, beta]))
    forecast = np.empty(horizon)
    last_residual = values[-1] - np.mean(values)
    last_variance = sigma2[-1]
    for index in range(horizon):
        forecast[index] = omega + alpha * last_residual**2 + beta * last_variance
        last_residual, last_variance = 0.0, forecast[index]
    return GARCHResult("scipy-fallback", float(omega), float(alpha), float(beta), float(np.mean(values) / 100.0), np.sqrt(sigma2) / 100.0, forecast / 10000.0)


def fit_garch(returns: pd.Series, horizon: int = 1) -> GARCHResult:
    """Fit GARCH(1,1), preferring arch and falling back to SciPy MLE."""
    values = returns.dropna().to_numpy(float)
    if len(values) < 30:
        raise ValueError("At least 30 returns are required for GARCH estimation.")
    returns_percent = values * 100.0
    try:
        from arch import arch_model

        model = arch_model(returns_percent, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
        fitted = model.fit(update_freq=0, disp="off")
        params = fitted.params
        forecast_variance = np.asarray(fitted.forecast(horizon=horizon, reindex=False).variance.iloc[-1], dtype=float) / 10000.0
        conditional = np.asarray(fitted.conditional_volatility, dtype=float) / 100.0
        return GARCHResult("arch", float(params.get("omega", 0.0)), float(params.get("alpha[1]", 0.0)), float(params.get("beta[1]", 0.0)), float(params.get("mu", 0.0) / 100.0), conditional, forecast_variance, fitted)
    except ImportError:
        return _fallback_fit(returns_percent, horizon)


def forecast_volatility(result: GARCHResult) -> np.ndarray:
    return np.sqrt(np.maximum(result.forecast_variance, 1e-12))

