from app.models import StockData_daily, Ticker
from app.extensions import db
import pandas as pd
from datetime import datetime


def save_stock_data_to_db(ticker, new_data):
    """
    Save stock data to the database using a normalized DB of ticker and stock_data

    Args:
        ticker (str):    Stock ticker symbol (e.g. NVDA)
        new_data (pd.DataFrame):    Stock data with columns: Date, Open, High, Low, Close, Volume
    """
    # checking if ticker already exist in database
    ticker_symbol = Ticker.query.filter_by(symbol=ticker).first()
    if not ticker_symbol:  # If ticker does not exist, create a new ticker in db
        ticker_symbol = Ticker(symbol=ticker)
        db.session.add(ticker_symbol)
        db.session.commit()

    # checking for duplicates
    existing_data = get_stock_data_from_db(ticker)

    unique_data = check_duplicates_date(existing_data, new_data)

    if unique_data.empty:
        print(
            f"No new data to add for {ticker}. All dates already exist in the database."
        )
        return  # Exit function if there's nothing new to add

    print(f"Adding {len(unique_data)} new rows for {ticker} to the database.")

    # Iterating through the data, saving to stock_data table in db
    for index, row in unique_data.iterrows():
        stock_data = StockData_daily(
            ticker_id=ticker_symbol.id,
            date=pd.to_datetime(row["date"], errors="coerce"),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        db.session.add(stock_data)

    db.session.commit()
    print(
        f"Successfully saved {len(unique_data)} new rows for {ticker} to the database."
    )


def get_stock_data_from_db(ticker):
    """
    Retreiving data from db for a specific ticker.

    Args:
        ticker(str):    The stock ticker symbol (e.g. NVDA).

    Returns:
        StockData (pd.DataFrame)
    """

    ticker = Ticker.query.filter_by(symbol=ticker).first()

    if not ticker:
        raise ValueError(f"Ticker {ticker} not found in database.")

    stock_data = StockData_daily.query.filter_by(ticker_id=ticker.id).all()

    # Convert stock_data to pd.DataFrame
    stock_data = [
        {
            "date": data.date,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.close,
            "volume": data.volume,
        }
        for data in stock_data
    ]
    stock_data = pd.DataFrame(stock_data)
    # Convert date to datetime object for date
    if not stock_data.empty:
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        print(
            f"Successfully loaded {len(stock_data)} new rows for {ticker} from the database."
        )
    else:
        print("No data was loaded from database")
    return stock_data


def append_data_to_csv(ticker, new_df):
    """
    Adding data to a CSV file based on ticker name

    Args:
        ticker(str)
        StockData(pd.DataFrame) that needs to be added
    """
    filename_ending = "_data.csv"
    filepath = "stock_data/" + ticker + filename_ending

    df = read_csv_data(filepath, ticker)

    unique_df_new = check_duplicates_date(df, new_df)

    # Append only unique dates
    if not unique_df_new.empty:
        unique_df_new.to_csv(filepath, mode="a", header=not df.shape[0], index=False)
        print(f"Added {len(unique_df_new)} new rows.")
    else:
        print("No new unique dates to add.")


def read_csv_data(filepath, ticker):

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # If the file does not exist

        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    # Automatically convert 'date'
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def check_duplicates_date(existing_df, new_df):
    """
    Checking to see if there are duplicates in the dates of the dataframes.

    Returns:
        pd.DataFrame: A new dataframe with only unique dates.
    """

    # Check if existing_df is empty
    if existing_df.empty:
        print("Warning: existing_df is empty, returning new_df.")
        return new_df  # Return DataFrame immediately

    # Ensure 'date' column exists in both DataFrames before processing
    if "date" not in existing_df.columns:
        raise KeyError("'date' column not found in existing_df.")
    if "date" not in new_df.columns:
        raise KeyError("'date' column not found in new_df.")

    # Convert date columns to datetime format for accurate comparison
    existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
    new_df["date"] = pd.to_datetime(new_df["date"], errors="coerce")

    # Filter out dates that already exist in the existing_df
    unique_df_new = new_df[~new_df["date"].isin(existing_df["date"])]

    return unique_df_new
