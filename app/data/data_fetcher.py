import pandas as pd
import os
from dotenv import load_dotenv
import requests
from app.data.data_handler import append_data_to_csv


load_dotenv()
api_key = os.getenv("ALPHA_VANTAGE_API")


def get_data_alpha_vantage(ticker):
    # API endpoint for daily stock prices
    start = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
    end = "&outputsize=full&apikey={api_key}"
    ticker = ticker
    url = start + ticker + end

    # Make the API request
    response = requests.get(url)
    data = response.json()

    # Extract the 'Time Series (Daily)' data
    daily_data = data["Time Series (Daily)"]

    # Convert to a pandas DataFrame
    df_new = pd.DataFrame.from_dict(daily_data, orient="index")

    # Rename columns for clarity
    df_new.columns = ["open", "high", "low", "close", "volume"]

    # Convert data types
    df_new = df_new.astype(
        {
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "int",
        }
    )

    # Convert index to datetime and add as a separate "date" column
    df_new.index = pd.to_datetime(df_new.index)
    df_new = df_new.sort_index()
    df_new.reset_index(inplace=True)  # Reset the index to make "date" a regular column
    df_new.rename(
        columns={"index": "date"}, inplace=True
    )  # Rename the index column to "date"

    # Reorder columns to make "date" the first column
    df_new = df_new[["date", "open", "high", "low", "close", "volume"]]

    # append_data_to_csv(ticker, df_new)
    return df_new

    # # Save the DataFrame to a CSV file and copy to google drive
    # filename="/content/" + ticker +"_historical_data.csv"
    # df.to_csv(filename, index=False)
    # !cp -r "{filename}" "/content/drive/MyDrive/Colab Notebooks/stock_predictor"
    # print("File saved to stock_predictor map")


# filename_ending = "_data.csv"
# filepath = "stock_data/" + "NVDA" + filename_ending
# df = pd.read_csv(filepath)
# print(df.dtypes)
# print(df)

# get_data_alpha_vantage("NVDA")
