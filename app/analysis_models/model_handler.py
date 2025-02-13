import os
import tensorflow as tf

MODEL_DIR = os.getenv("MODEL_DIR", "trained_models")


def get_model_path(model_name, version):
    """Generate model path dynamically based on name and version."""
    return os.path.join(MODEL_DIR, f"{model_name}_v{version}.keras")


def save_lstm_model(model, model_name, version=1):
    path = get_model_path(model_name, version)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Check model directory exists
    model.save(path)
    print(f"LSTM model saved at {path}")


def load_model(selected_model):
    """Load a model based on the selected model name."""
    model_path = os.path.join(MODEL_DIR, selected_model)
    print(model_path)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Model not found: {selected_model}")


def get_available_models():
    """List available models in the trained_models directory."""

    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    return models
