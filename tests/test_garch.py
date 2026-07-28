import numpy as np
import pandas as pd

from financial_time_series.garch import fit_garch, forecast_volatility


def test_garch_returns_positive_volatility():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, 60))
    result = fit_garch(returns, horizon=4)
    assert result.backend in {"arch", "scipy-fallback"}
    assert len(result.forecast_variance) == 4
    assert np.all(forecast_volatility(result) > 0)

