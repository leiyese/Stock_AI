from flask import render_template, request, redirect, url_for
from dotenv import load_dotenv
import os
from app.extensions import db, bp
from app.data.data_fetcher import get_data_alpha_vantage
from app.data.data_handler import save_stock_data_to_db
from app.data.data_handler import get_stock_data_from_db
from app.analysis_models.lstm_model import (
    df_to_windowed_df,
    windowed_df_to_date_X_y,
    split_data_train_val_test,
    lstm_model,
)
from app.analysis_models.model_handler import get_model_path, save_lstm_model

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

        model = lstm_model(X_train, y_train, X_val, y_val)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
        model.summary()

        # TODO in the future: SHOW MSE and evaluation score!

        # Save the trained model
        os.makedirs(
            "trained_models", exist_ok=True
        )  # Create a folder if it doesn't exist
        save_lstm_model(model, version=1)

        return render_template(
            "train.html", tickers=tickers, selected_ticker=selected_ticker
        )

    return render_template("train.html", tickers=tickers)


# @bp.route("/analyse", methods=["GET", "POST"])
# def analyse():
#     pass


@bp.route("/analyse")
def analyse():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64

    # Generate a figure
    fig, ax = plt.subplots(figsize=(8, 5))  # Control size

    # Example modern-looking plot
    x = [1, 2, 3, 4, 5]
    y = [10, 15, 7, 12, 18]
    sns.lineplot(x=x, y=y, marker="o", ax=ax)

    ax.set_title("Modern Pyplot in Flask", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Axis", fontsize=12)
    ax.set_ylabel("Y Axis", fontsize=12)

    # Save the figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")  # Ensures no cropping
    img.seek(0)  # Move cursor to start

    # Encode image to base64 string
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Prevents memory leaks

    return render_template("analyse.html", img_data=img_base64)


# @bp.route("/evaluate", methods=["GET", "POST"])
# def evaluate():
#     tickers = Ticker.query.all()

#     if request.method == "POST":
#         selected_ticker = request.form["ticker"]
#         start_date = request.form["start_date"]
#         end_date = request.form["end_date"]

#         df = get_stock_data_from_db(selected_ticker)
#         windowed_df = df_to_windowed_df(df, start_date, end_date, n=3)
#         dates, X, y = windowed_df_to_date_X_y(windowed_df)

#         _, _, _, _, _, _, dates_test, X_test, y_test = split_data_train_val_test(
#             dates, X, y
#         )

#         # ✅ Load the saved model
#         MODEL_PATH = os.getenv("MODEL_PATH", "trained_models/lstm_model")
#         if not os.path.exists(MODEL_PATH):
#             return "Error: No trained model found. Please train first."

#         loaded_model = keras.saving.load_model(MODEL_PATH)

#         # Get model predictions
#         predictions = model.predict(X_test)

#         # Evaluate model accuracy (example: mean squared error)
#         mse = np.mean((predictions - y_test) ** 2)

#         return render_template(
#             "evaluate.html",
#             mse=mse,
#             selected_ticker=selected_ticker,
#             predictions=predictions,
#             actual=y_test,
#         )

#     return render_template("evaluate.html", tickers=tickers)


# @bp.route("/split", methods=["POST", "GET"])
# def split():
#     ticker = request.form["ticker"]
#     split_date = request.form["split_date"]

#     return render_template("split.html")
