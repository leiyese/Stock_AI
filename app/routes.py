from flask import render_template, request, redirect, url_for
from dotenv import load_dotenv
import os


from app.extensions import db, bp
from app.data.data_fetcher import get_data_alpha_vantage
from app.data.data_handler import (
    save_stock_data_to_db,
    get_stock_data_from_db,
    append_data_to_csv,
)
from app.data.data_split import split_stock_data
from app.analysis_models.lstm_model import (
    df_to_windowed_df,
    windowed_df_to_date_X_y,
    split_data_train_val_test,
    lstm_model,
    plot_lstm,
    plot_lstm_whole,
)
from app.analysis_models.model_handler import (
    save_lstm_model,
    load_model,
    get_available_models,
)

# from app.data.data_split
# from app.data.data_fetcher import get_data_alpha_vantage

from app.models import Ticker, StockData_daily

# from flask import current_app as app

# Use the loaded MODEL_DIR
MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")


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


@bp.route("/train", methods=["GET", "POST"])
def train():

    tickers = Ticker.query.all()  # Fetch all tickers from the database

    if request.method == "POST":
        selected_ticker = request.form["ticker"]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        df = get_stock_data_from_db(selected_ticker)
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

        model = lstm_model()
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
        model.summary()

        # TODO in the future: SHOW MSE and evaluation score!

        # Save the trained model
        os.makedirs(
            "trained_models", exist_ok=True
        )  # Create a folder if it doesn't exist
        save_lstm_model(model, version=1)

        return render_template(
            "index.html", tickers=tickers, selected_ticker=selected_ticker
        )

    return render_template("train.html", tickers=tickers)


# @bp.route("/analyse", methods=["GET", "POST"])
# def analyse():
#     pass


@bp.route("/analyse", methods=["POST", "GET"])
def analyse():
    tickers = Ticker.query.all()  # Fetch all tickers from the database
    print(tickers)
    available_models = get_available_models()
    if request.method == "POST":
        # Get user selections
        selected_model = request.form.get("model")
        selected_ticker = request.form.get("ticker")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")

        # Load model
        loaded_model = load_model(selected_model)

        # Load data
        df = get_stock_data_from_db(selected_ticker)
        selected_start_date = start_date
        selected_end_date = end_date
        windowed_df = df_to_windowed_df(df, selected_start_date, selected_end_date, n=3)
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

        img_base64 = plot_lstm(
            selected_ticker, loaded_model, dates_test, X_test, y_test, "test"
        )

        return render_template(
            "analyse.html",
            img_data=img_base64,
            available_models=available_models,
            tickers=tickers,
            selected_model=selected_model,
            selected_ticker=selected_ticker,
            start_date=start_date,
            end_date=end_date,
        )
    return render_template(
        "analyse.html", available_models=available_models, tickers=tickers
    )


@bp.route("/full_plot")
def test():
    """
    Flask route to test LSTM model predictions for train, validation, and test data.
    """
    # Load model
    loaded_model = load_model("lstm", version=1)

    # Get stock data
    df = get_stock_data_from_db("MSFT")
    start_date = "2020-01-01"
    end_date = "2023-09-01"

    # Prepare data
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

    # Generate Base64 encoded plot
    img_base64 = plot_lstm_whole(
        loaded_model,
        dates_train,
        X_train,
        y_train,
        dates_val,
        X_val,
        y_val,
        dates_test,
        X_test,
        y_test,
    )

    return render_template("full_plot.html", img_data=img_base64)


# @bp.route("/split", methods=["POST", "GET"])
@bp.route("/split")
def split():
    # ticker = request.form["ticker"]
    # split_date = request.form["split_date"]

    ticker = "MSFT"
    split_dates = [
        "1987-09-21",
        "1990-04-16",
        "1991-06-27",
        "1992-06-15",
        "1994-05-23",
        "1996-12-09",
        "1998-02-23",
        "1999-03-29",
        "2003-02-18",
    ]
    split_ratio = [0.5, 0.5, 3 / 2, 3 / 2, 0.5, 0.5, 0.5, 0.5, 0.5]
    df = get_stock_data_from_db(ticker)
    df_new = split_stock_data(df, split_dates[-1], split_ratio[-1])
    print(df.dtypes)
    print(df_new.dtypes)
    append_data_to_csv("MSFT_test", df_new)

    return render_template("index.html")
