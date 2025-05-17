# config.py
from pathlib import Path
import mlflow

# Set up MLflow tracking
MODEL_REGISTRY = Path("/tmp/mlflow")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Shared constants
DEFAULT_SEED = 42
DATA_PATH = "C:/Users/7119001/OneDrive - Western Digital/Documents/data_requested/tipqc_psmds/special_topics_data_science/UCI_Heart_Disease_Dataset_Combined.csv"
