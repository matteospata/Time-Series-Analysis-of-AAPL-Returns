import pandas as pd

from financial_time_series.features import add_return_features, clean_returns


def test_return_features_have_expected_columns_and_length():
    prices = pd.DataFrame({"price": [100.0 + index * 0.25 + (index % 3) for index in range(31)]}, index=pd.date_range("2024-01-01", periods=31))
    features = add_return_features(prices, volatility_window=2)
    assert {"log_return", "simple_return", "rolling_volatility", "drawdown"}.issubset(features.columns)
    assert len(clean_returns(features)) == 30
