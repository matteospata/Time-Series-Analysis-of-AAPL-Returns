from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from financial_time_series.arima import fit_and_forecast
from financial_time_series.data import download_prices, load_price_csv, save_price_csv
from financial_time_series.diagnostics import autocorrelation_diagnostics, distribution_diagnostics, stationarity_test
from financial_time_series.evaluation import forecast_metrics
from financial_time_series.features import add_return_features, clean_returns
from financial_time_series.garch import fit_garch, forecast_volatility
from financial_time_series.lstm import fit_lstm
from financial_time_series.plotting import plot_acf_pacf, plot_forecasts, plot_overview, plot_var_backtest
from financial_time_series.var import backtest_var, forecast_historical_var, forecast_parametric_var, garch_var


@dataclass
class PipelineConfig:
    ticker: str = "AAPL"
    start: str | None = None
    end: str | None = None
    years: int = 10
    data_path: str | None = None
    output_dir: str = "artifacts"
    test_size: float = 0.2
    confidence: float = 0.99
    var_window: int = 252
    lstm_window: int = 20
    lstm_epochs: int = 30
    seed: int = 42


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def resolve_start_date(start: str | None, end: str | None, years: int) -> str:
    if start:
        return start
    if years < 1:
        raise ValueError("years must be at least 1.")
    reference_date = pd.Timestamp(end).normalize() if end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return str((reference_date - pd.DateOffset(years=years)).date())


def run_pipeline(config: PipelineConfig) -> dict:
    _seed_everything(config.seed)
    resolved_start = resolve_start_date(config.start, config.end, config.years)
    prices = load_price_csv(config.data_path) if config.data_path else download_prices(config.ticker, resolved_start, config.end)
    features = add_return_features(prices)
    returns = clean_returns(features)
    if not 0.05 < config.test_size < 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    split_index = int(len(returns) * (1.0 - config.test_size))
    train_returns, test_returns = returns.iloc[:split_index], returns.iloc[split_index:]
    if len(train_returns) < max(30, config.lstm_window + 3) or len(test_returns) < 3:
        raise ValueError("The series is too short for the requested test split and LSTM window.")

    arima = fit_and_forecast(train_returns, len(test_returns))
    garch = fit_garch(train_returns, len(test_returns))
    garch_volatility = forecast_volatility(garch)
    lstm = fit_lstm(train_returns, window=config.lstm_window, epochs=config.lstm_epochs, seed=config.seed)
    lstm_forecast = lstm.forecast(train_returns, len(test_returns))

    history = train_returns.copy()
    historical_values: list[float] = []
    parametric_values: list[float] = []
    for value in test_returns:
        historical_values.append(forecast_historical_var(history, config.confidence, config.var_window))
        parametric_values.append(forecast_parametric_var(history, config.confidence, config.var_window))
        history = pd.concat([history, pd.Series([value])])
    garch_values = np.asarray([garch_var(garch.mean, volatility, config.confidence) for volatility in garch_volatility])
    var_forecasts = {"historical": pd.Series(historical_values, index=test_returns.index), "parametric": pd.Series(parametric_values, index=test_returns.index), "garch": pd.Series(garch_values, index=test_returns.index)}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(config.output_dir) / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_price_csv(prices, artifact_dir / "prices.csv")
    features.to_csv(artifact_dir / "features.csv", index_label="Date")
    plot_overview(features, artifact_dir / "overview.png")
    diagnostics = {"stationarity": stationarity_test(returns), "autocorrelation": autocorrelation_diagnostics(returns), "distribution": distribution_diagnostics(returns)}
    plot_acf_pacf(diagnostics["autocorrelation"], artifact_dir / "acf_pacf.png")
    plot_forecasts(test_returns, {f"ARIMA{arima.order}": arima.forecast, "LSTM": lstm_forecast}, artifact_dir / "forecast_comparison.png")
    plot_var_backtest(test_returns, var_forecasts, artifact_dir / "var_backtest.png")

    metrics = {"ARIMA": forecast_metrics(test_returns.to_numpy(), arima.forecast), "LSTM": forecast_metrics(test_returns.to_numpy(), lstm_forecast)}
    var_backtests = {name: backtest_var(test_returns, values, config.confidence) for name, values in var_forecasts.items()}
    observed_years = (prices.index.max() - prices.index.min()).days / 365.25
    report = {"project": "Financial Time Series Lab", "ticker": config.ticker if config.data_path is None else "CSV_INPUT", "data_window": {"requested_years": config.years if config.data_path is None else None, "observed_years": round(float(observed_years), 3), "resolved_start": resolved_start if config.data_path is None else str(prices.index.min().date()), "end": config.end or ("latest_available" if config.data_path is None else str(prices.index.max().date()))}, "observations": {"prices": len(prices), "returns": len(returns), "train_returns": len(train_returns), "test_returns": len(test_returns)}, "configuration": asdict(config), "models": {"arima": {"order": arima.order, "aic": arima.aic}, "garch": {"backend": garch.backend, "omega": garch.omega, "alpha": garch.alpha, "beta": garch.beta, "mean": garch.mean}, "lstm": {"window": lstm.window, "epochs": config.lstm_epochs, "final_training_loss": lstm.history[-1]}}, "diagnostics": diagnostics, "forecast_metrics": metrics, "var_backtests": var_backtests, "artifacts": ["prices.csv", "features.csv", "overview.png", "acf_pacf.png", "forecast_comparison.png", "var_backtest.png"]}
    _write_json(report, artifact_dir / "report.json")
    pd.DataFrame([{"model": model, **values} for model, values in metrics.items()]).to_csv(artifact_dir / "forecast_metrics.csv", index=False)
    pd.DataFrame([{ "method": method, **values } for method, values in var_backtests.items()]).to_csv(artifact_dir / "var_backtest_metrics.csv", index=False)
    return {"artifact_dir": str(artifact_dir), "report": report}
