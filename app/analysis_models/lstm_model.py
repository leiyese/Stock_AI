import pandas as pd
import datetime
import numpy as np

# Plot imports
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


def str_to_datetime(s):
    """
    restructuring the data "date" into a datetime type
    """
    # If s is already a datetime, return it
    if isinstance(s, pd.Timestamp):
        return s
    if isinstance(s, str):
        split = s.split("-")  # Splits depending on the - symbol
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    raise ValueError(f"Unexpected date format: {s}")


def df_to_windowed_df(df, first_date_str, last_date_str, n=3):
    df["date"] = df["date"].apply(
        str_to_datetime
    )  # apply the function to the whole date column

    df.index = df.pop("date")  # remove index

    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates, X, Y = [], [], []
    last_time = False

    while True:
        df_subset = df.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f"Error: Window of size {n} is too large for date {target_date}")
            return

        values = df_subset["close"].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = df.loc[target_date : target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split("T")[0]
        year_month_day = next_date_str.split("-")
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df["Target Date"] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f"Target-{n-i}"] = X[:, i]

    ret_df["Target"] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_df):
    """
    change windowed pd.dataframe to 3 numpy variables: Date / X / y in format Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    # Change whole dataframe to numpy dataframe
    df_np = windowed_df.to_numpy()

    # Get dates from the first column
    dates = df_np[:, 0]

    # Get X matrix from 2nd to 4th column
    VARIABLE = 1  # Univariate forecasting using only closing price, eg only 1 variable
    middle_matrix = df_np[:, 1:-1]
    # Need to reshape the tensor for LSTM
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], VARIABLE))

    # Get output column from last column
    y = df_np[:, -1]

    return (
        dates,
        X.astype(np.float32),
        y.astype(np.float32),
    )  # Need float type for the arrays


def split_data_train_val_test(dates, X, y):
    """
    Args:
        dates(np.array)
        X(np.array)
        y(np.array)
    return:
        train, test, validation split of dates, X and y in np.array type
    """

    q_80 = int(len(dates) * 0.8)  # 0-80 % of data
    q_90 = int(len(dates) * 0.9)  # 80-90 % of data

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    return (
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


def lstm_model():

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import layers

    model = Sequential(
        [
            layers.Input((3, 1)),  # 3 dates back and 1 variable thus (3,1)
            layers.LSTM(
                64
            ),  # The bigger the number the more complicated the model and more prone to overfitting
            layers.Dense(32, activation="relu"),  # ReLu activation function for layers
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )  # No need to specify activation since the last output is linear (default)

    model.compile(
        loss="mse",  # MeanSquaredLoss
        optimizer=Adam(
            learning_rate=0.001
        ),  # Change learning_rate for different results in the ML model
        metrics=["mean_absolute_error"],
    )

    return model


# Display the model's architecture
# model.summary()


# TODO  for improvements
# Make it work on GPU
# Save models with specific names and load models


def plot_lstm_whole(
    model,
    dates_train,
    X_train,
    y_train,
    dates_val,
    X_val,
    y_val,
    dates_test,
    X_test,
    y_test,
):
    """
    Generates a combined LSTM prediction plot for train, validation, and test data.
    Returns a Base64 string for rendering in HTML.
    """
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot predictions and actual values
    sns.lineplot(
        x=dates_train,
        y=train_predictions,
        label="Train Predictions",
        linestyle="dashed",
        ax=ax,
    )
    sns.lineplot(x=dates_train, y=y_train, label="Train Observations", ax=ax)
    sns.lineplot(
        x=dates_val,
        y=val_predictions,
        label="Validation Predictions",
        linestyle="dashed",
        ax=ax,
    )
    sns.lineplot(x=dates_val, y=y_val, label="Validation Observations", ax=ax)
    sns.lineplot(
        x=dates_test,
        y=test_predictions,
        label="Test Predictions",
        linestyle="dashed",
        ax=ax,
    )
    sns.lineplot(x=dates_test, y=y_test, label="Test Observations", ax=ax)

    # Title and labels
    ax.set_title(
        "LSTM Model Predictions for Train, Validation & Test",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()

    # Save figure to memory
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    # Convert to Base64
    img_base64 = base64.b64encode(img.getvalue()).decode()

    # Cleanup
    plt.close(fig)
    plt.clf()

    return img_base64


def plot_lstm(ticker, model, dates, X, y, data_type="test"):
    """
    Plots LSTM predictions vs actual values.

    Parameters:
    - model: Trained LSTM model
    - dates: Dates corresponding to data points
    - X: Feature set (input data)
    - y: Ground truth (actual values)
    - data_type: String ("train", "val", or "test") to customize the legend

    Returns:
    - Base64-encoded PNG image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Make predictions
    predictions = model.predict(X).flatten()

    # Define legend names dynamically
    legend_pred = f"{data_type.capitalize()} Predictions"
    legend_obs = f"{data_type.capitalize()} Observations"

    sns.lineplot(x=dates, y=predictions, label=legend_pred, linestyle="dashed", ax=ax)
    sns.lineplot(x=dates, y=y, label=legend_obs, marker="o", ax=ax)

    # Title and labels
    ax.set_title(
        f"{ticker} LSTM Predictions vs Actual ({data_type.capitalize()} Data)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()

    # Save figure to memory
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)

    # Convert to Base64
    img_base64 = base64.b64encode(img.getvalue()).decode()

    # Cleanup
    plt.close(fig)
    plt.clf()

    return img_base64
