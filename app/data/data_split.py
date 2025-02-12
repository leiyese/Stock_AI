import pandas as pd


def split_stock_data(df, split_date, split_ratio) -> pd.DataFrame:
    """
    Adjust stock price data for a stock split (excludes volume from adjustment).

    Parameters:
        df (pd.DataFrame): Stock price DataFrame (must contain 'date', 'open', 'high', 'low', 'close', 'volume').
        split_date (str): Date of the stock split (format: 'YYYY-MM-DD').
        split_ratio (float): Split ratio (e.g., 2.0 for 1:2 split, 0.5 for 2:1 split).

    Returns:
        pd.DataFrame: Adjusted stock data.
    """
    # Convert date column and split_date to datetime
    df["date"] = pd.to_datetime(df["date"])
    split_date = pd.to_datetime(split_date, errors="coerce")

    if split_date not in df["date"].values:
        raise ValueError(f"Split date {split_date} not found in the dataset.")

    # Get index of df that corresponds to split_date
    idx = df.index[df["date"] == split_date][0]

    # Separate pre-split and post-split data
    df_adjusted = df.iloc[:idx].copy()  # Copy to avoid modifying original df
    df_no_adjusted = df.iloc[idx:].copy()

    # Define columns to adjust (excluding 'volume')
    price_columns = ["open", "high", "low", "close"]

    # Apply split ratio only to price columns
    df_adjusted[price_columns] *= split_ratio

    # Adjust volume (inverse of price ratio)
    df_adjusted["volume"] /= split_ratio
    # change back the dtype of the column to int64 before returning the whole df
    df_adjusted["volume"] = df_adjusted["volume"].astype("int64")

    # Concatenate adjusted and unadjusted data
    adjusted_df = pd.concat([df_adjusted, df_no_adjusted], ignore_index=True)

    return adjusted_df
