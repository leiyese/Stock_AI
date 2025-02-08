from flask import render_template, request, redirect, url_for

from app.extensions import db, bp

from app.data.data_fetcher import get_data_alpha_vantage
from app.data.data_handler import save_stock_data_to_db, append_data_to_csv
from app.data.data_handler import read_csv_data, get_stock_data_from_db
from app.analysis_models.lstm_model import (
    df_to_windowed_df,
    windowed_df_to_date_X_y,
    split_data_train_val_test,
    lstm_model,
)

# from app.data.data_split
# from app.data.data_fetcher import get_data_alpha_vantage

from app.models import Ticker, StockData_daily

# from flask import current_app as app


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/download", methods=["POST", "GET"])
def download():
    if request.method == "POST":
        print("Received form data:", request.form)  # Debugging

        if "ticker" not in request.form:
            return "Ticker is required!", 400  # Return error if missing

        ticker = request.form["ticker"]
        data = get_data_alpha_vantage(ticker)
        save_stock_data_to_db(ticker, data)

        return redirect(url_for("bp.index"))

    return render_template("download.html")


@bp.route("/analyze", methods=["GET", "POST"])
def analyze():
    tickers = Ticker.query.all()  # Fetch all tickers from the database
    ticker = [ticker.symbol for ticker in tickers]

    if request.method == "POST":
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        df = get_stock_data_from_db(ticker)
        windowed_df = df_to_windowed_df(df, start_date, end_date, n=3)
        dates, X, y = windowed_df_to_date_X_y(windowed_df)
        (
            dates_train,
            X_train,
            y_train,
            dates_val,
            X_val,
            y_val,
            dates_test,
            X_test,
            y_test,
        ) = split_data_train_val_test(dates, X, y)

        model = lstm_model(X_train, y_train, X_val, y_val)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

        return render_template("analyze.html", tickers=tickers)

    return render_template("analyze.html", tickers=tickers)


# @bp.route("/split", methods=["POST", "GET"])
# def split():
#     ticker = request.form["ticker"]
#     split_date = request.form["split_date"]

#     return render_template("split.html")
