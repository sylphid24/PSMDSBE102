# serve.py
import typer
import joblib

app = typer.Typer()

@app.command()
def run(run_id: str, threshold: float = 0.5):
    model = joblib.load(f"models/{run_id}.pkl")
    print(f"Model loaded with threshold = {threshold}")

if __name__ == "__main__":
    app()
