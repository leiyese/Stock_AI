import pandas as pd
from app.data.data_fetcher import get_data_alpha_vantage


def test_stock_data_validation():
    df = get_data_alpha_vantage("AAPL")

    # Testing if the df is in a DataFrame after downloading from API
    assert isinstance(df, pd.DataFrame)

    # Testing for required columns
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    assert all(col in df.columns for col in required_columns)

    # Testing for missing values
    assert not df.isnull().values.any()
