import pandas as pd


def split_stock_data(df, split_date, split_ratio) -> pd.DataFrame:
    """
    Manual split for data when stock does a split
    """
