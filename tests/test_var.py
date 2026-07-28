import pandas as pd

from financial_time_series.var import backtest_var, historical_var, parametric_var


def test_var_is_positive_for_loss_tail():
    returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.02, 0.03])
    assert historical_var(returns, confidence=0.8) > 0
    assert parametric_var(returns, confidence=0.8) > 0
    report = backtest_var(returns, pd.Series([0.01] * len(returns)), confidence=0.8)
    assert report["observations"] == len(returns)

