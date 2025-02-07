# %%
import os
from app import create_app
from app.models import StockData_daily, Ticker, db
from app.data.data_handler import check_duplicates_date, read_csv_data
import pandas as pd

# %%
ticker = "MSFT"
path = (
    "/Users/leiye/Downloads/Jensens/4 Python forts/inlamning/stock_data/MSFT_data.csv"
)
df = read_csv_data(path, ticker)

print(df)

# %%

new_df = pd.DataFrame()
print(new_df.head())

# %%

unique_df = check_duplicates_date(df, new_df)
unique_df.head()

# %%
