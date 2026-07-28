from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_overview(frame: pd.DataFrame, path: str | Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    frame["price"].plot(ax=axes[0], color="#2563eb", title="Price")
    frame["log_return"].plot(ax=axes[1], color="#64748b", title="Log returns")
    frame["rolling_volatility"].plot(ax=axes[2], color="#9333ea", title="Annualized rolling volatility")
    _save(figure, path)


def plot_acf_pacf(diagnostics: dict, path: str | Path) -> None:
    acf_values, pacf_values = diagnostics["acf"], diagnostics["pacf"]
    lags = np.arange(len(acf_values))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].stem(lags, acf_values, basefmt=" ")
    axes[0].set_title("ACF of log returns")
    axes[0].set_xlabel("Lag")
    axes[1].stem(lags, pacf_values, basefmt=" ")
    axes[1].set_title("PACF of log returns")
    axes[1].set_xlabel("Lag")
    _save(figure, path)


def plot_var_backtest(returns: pd.Series, forecasts: dict[str, pd.Series], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    returns.plot(ax=axis, color="#64748b", linewidth=0.8, label="Realized log return")
    for name, values in forecasts.items():
        (-values).plot(ax=axis, linewidth=1.2, label=f"-{name} VaR")
    axis.set_title("VaR backtest: realized returns versus loss thresholds")
    axis.legend()
    _save(figure, path)


def plot_forecasts(actual: pd.Series, forecasts: dict[str, np.ndarray], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_values = np.arange(len(actual))
    axis.plot(x_values, actual.to_numpy(), color="#111827", linewidth=1.4, label="Actual return")
    for name, values in forecasts.items():
        axis.plot(x_values, values, linewidth=1.1, label=name)
    tick_positions = np.linspace(0, len(actual) - 1, min(6, len(actual)), dtype=int)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([actual.index[index].strftime("%Y-%m-%d") for index in tick_positions], rotation=30, ha="right")
    axis.set_title("Out-of-sample return forecasts")
    axis.legend()
    _save(figure, path)
