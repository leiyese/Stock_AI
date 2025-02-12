import pandas as pd
from app.data.data_split import split_stock_data


def test_split_stock_data():

    # Creating a sample dataframe
    data = {
        "date": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [101, 102, 103, 104, 105],
        "volume": [1000, 2000, 3000, 4000, 5000],
    }

    df = pd.DataFrame(data)
    df["volume"] = df["volume"].astype("int64")

    split_date = "2000-01-03"
    split_ratio = 1 / 10  # 1:10 ratio split

    adjusted_df = split_stock_data(df, split_date, split_ratio)

    expected_data = {
        "date": ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
        "open": [100 / 10, 101 / 10, 102, 103, 104],
        "high": [105 / 10, 106 / 10, 107, 108, 109],
        "low": [95 / 10, 96 / 10, 97, 98, 99],
        "close": [101 / 10, 102 / 10, 103, 104, 105],
        "volume": [1000 * 10, 2000 * 10, 3000, 4000, 5000],
    }

    expected_df = pd.DataFrame(expected_data)
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    expected_df["volume"] = expected_df["volume"].astype("int64")

    # Assertion for dataframe
    pd.testing.assert_frame_equal(adjusted_df, expected_df)

    # Assertion if volume is int
    assert adjusted_df["volume"].dtype == "int64"
