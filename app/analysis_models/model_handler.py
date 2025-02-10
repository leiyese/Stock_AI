import os
import tensorflow as tf


def get_model_path(model_name, version):
    """Generate model path dynamically based on name and version."""
    return os.path.join("trained_models", f"{model_name}_v{version}.keras")


def save_lstm_model(model, version=1):
    path = get_model_path("lstm", version)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Check model directory exists
    model.save(path)
    print(f"LSTM model saved at {path}")


def load_model(model_name, version):
    path = get_model_path(model_name, version)

    if model_name == "lstm":
        print(f"{model_name} was loaded")
        return tf.keras.models.load_model(path)
    # ADD OTHER MODEL NAMES HERE IF AVAILABLE such as random forest
    else:
        raise ValueError("Unknown model type!")
