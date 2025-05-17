# train.py
import typer
from typing_extensions import Annotated
import joblib
import os

from data import load_data, preprocess_data
from models import get_model
from evaluate import evaluate_model
from config import DATA_PATH
from utils import set_seeds

app = typer.Typer()

@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="Experiment name")],
    data_path: Annotated[str, typer.Option(help="Path to dataset CSV file")] = DATA_PATH
):
    set_seeds()
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = get_model()
    model.fit(X_train, y_train)

    acc = evaluate_model(model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{experiment_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    app()
