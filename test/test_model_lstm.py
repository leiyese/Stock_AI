from app.analysis_models.lstm_model import lstm_model


def test_create_lstm_model():
    model = lstm_model()

    assert model.input_shape == (None, 3, 1)
    # Explanation of Each Dimension:
    # 1.	None:
    # •	This represents the batch size.
    # •	It is set to None because the batch size can vary (e.g., 32, 64, or any number of samples).
    # •	The model can handle any number of input sequences in a batch.
    # 2.	3:
    # •	This is the time step or sequence length.
    # •	It means the model looks at 3 time steps at a time.
    # •	For example, if you’re predicting stock prices, the model would look at the prices from the past 3 days to predict the next one.
    # 3.	1:
    # •	This is the number of features.
    # •	It indicates that one feature is being fed into the LSTM at each time step.
    # •	Common in univariate time series (e.g., only the closing price of a stock).

    assert model.output_shape == (None, 1)
    # The output is the next days price
