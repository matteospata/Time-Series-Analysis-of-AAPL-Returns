from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ARIMAResult:
    order: tuple[int, int, int]
    fitted_model: object
    forecast: np.ndarray
    aic: float


def select_order(values: pd.Series, max_p: int = 2, max_q: int = 2) -> tuple[int, int, int]:
    """Select an ARIMA(p,0,q) order by AIC on the training series only."""
    best_order, best_aic = (1, 0, 1), np.inf
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                fit = ARIMA(np.asarray(values, dtype=float), order=(p, 0, q), trend="c").fit()
                if np.isfinite(fit.aic) and fit.aic < best_aic:
                    best_order, best_aic = (p, 0, q), float(fit.aic)
            except (ValueError, np.linalg.LinAlgError):
                continue
    return best_order


def fit_and_forecast(values: pd.Series, horizon: int, order: tuple[int, int, int] | None = None) -> ARIMAResult:
    chosen_order = order or select_order(values)
    model = ARIMA(np.asarray(values, dtype=float), order=chosen_order, trend="c")
    fitted = model.fit()
    forecast = np.asarray(fitted.forecast(steps=horizon), dtype=float)
    return ARIMAResult(chosen_order, fitted, forecast, float(fitted.aic))
