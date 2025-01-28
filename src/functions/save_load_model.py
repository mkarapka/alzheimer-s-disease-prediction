import joblib
import os


def save_model(model, file_name):
    model_filepath = os.path.join("models", "trained", file_name) + ".joblib"
    joblib.dump(model, model_filepath)
    print(f"Model has been saved as {model_filepath}")


def load_model(file_name):
    model_filepath = os.path.join("models", "trained", file_name) + ".joblib"
    loaded_model = joblib.load(model_filepath)
    return loaded_model
