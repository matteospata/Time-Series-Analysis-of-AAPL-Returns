from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(prices: pd.DataFrame, volatility_window: int = 21) -> pd.DataFrame:
    if "price" not in prices.columns:
        raise ValueError("Expected a 'price' column.")
    frame = prices.copy()
    frame["log_return"] = np.log(frame["price"]).diff()
    frame["simple_return"] = frame["price"].pct_change()
    frame["rolling_volatility"] = frame["log_return"].rolling(volatility_window).std() * np.sqrt(252)
    frame["drawdown"] = frame["price"] / frame["price"].cummax() - 1.0
    return frame


def clean_returns(frame: pd.DataFrame) -> pd.Series:
    if "log_return" not in frame.columns:
        raise ValueError("Run add_return_features before extracting returns.")
    result = frame["log_return"].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(result) < 30:
        raise ValueError("At least 30 valid returns are required for time-series analysis.")
    return result

