# data.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
