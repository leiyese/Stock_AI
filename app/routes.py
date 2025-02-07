from flask import render_template, request, redirect, url_for

from app.extensions import db, bp

from app.data.data_handler import save_stock_data_to_db, append_data_to_csv
from app.data.data_handler import read_csv_data, get_stock_data_from_db

# from app.data.data_split
# from app.data.data_fetcher import get_data_alpha_vantage

from app.models import Ticker, StockData_daily

# from flask import current_app as app


@bp.route("/")
def index():

    ## MANUELL TESTNING

    ticker = "AAPL2"
    # filename_ending = "_data.csv"
    # filepath = "stock_data/" + ticker + filename_ending
    # df = read_csv_data(filepath, ticker)
    # save_stock_data_to_db(ticker, df)

    df = get_stock_data_from_db(ticker)

    print(df)

    return render_template("index.html")
