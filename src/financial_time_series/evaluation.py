from __future__ import annotations

import numpy as np


def forecast_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    error = actual - forecast
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))), "directional_accuracy": float(np.mean(np.sign(actual) == np.sign(forecast)))}

