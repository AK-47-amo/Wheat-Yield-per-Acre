"""
Model training and evaluation module with MLflow tracking for Wheat Yield prediction (MP130).
"""

import hashlib
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.wheat_yield.data import clean_dataframe, compute_sha256


NUMERIC_FEATURES = [
    "rainfall_mm", "fertilizer_kg", "sowing_delay_days",
    "irrigation_count", "soil_ph", "farm_acres"
]
CATEGORICAL_FEATURES = ["seed_variety", "tehsil"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "yield_qtl_per_acre"


def build_pipeline(regressor) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES)
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])


def train_models():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    tracking_uri = params["mlflow"]["tracking_uri"]
    exp_name = params["mlflow"]["experiment_name"]
    model_name = params["mlflow"]["registered_model_name"]
    data_sha = compute_sha256("data/raw/train.csv")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    # Load and clean training data
    train_raw = pd.read_csv("data/raw/train.csv")
    train_clean = clean_dataframe(train_raw, is_train=True)

    later_raw = pd.read_csv("data/raw/later.csv")
    later_clean = clean_dataframe(later_raw, is_train=False)

    X = train_clean[FEATURE_COLS]
    y = train_clean[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"]
    )

    X_later = later_clean[FEATURE_COLS]
    y_later = later_clean[TARGET_COL]

    configurations = [
        {
            "name": "RandomForest_100",
            "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 10}
        },
        {
            "name": "GradientBoosting_depth5",
            "model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
            "params": {"model_type": "GradientBoosting", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05}
        },
        {
            "name": "GradientBoosting_depth6_200",
            "model": GradientBoostingRegressor(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=42),
            "params": {"model_type": "GradientBoosting", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.03}
        }
    ]

    best_val_mae = float("inf")
    best_run_id = None
    best_pipe = None

    for config in configurations:
        with mlflow.start_run(run_name=config["name"]) as run:
            pipe = build_pipeline(config["model"])
            pipe.fit(X_train, y_train)

            val_preds = pipe.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, val_preds))

            later_preds = pipe.predict(X_later)
            later_mae = float(mean_absolute_error(y_later, later_preds))

            # Logging requirements from handbook
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_params(config["params"])
            mlflow.log_metric("mae_val", val_mae)
            mlflow.log_metric("mae_later", later_mae)
            mlflow.log_metric("mae", val_mae)

            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(f"Run {config['name']}: val_mae={val_mae:.4f}, later_mae={later_mae:.4f}, run_id={run.info.run_id}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_run_id = run.info.run_id
                best_pipe = pipe

    # Register best v1 model
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_v1 = mlflow.register_model(model_uri, model_name)
    print(f"Registered v1 model: {model_name} version {reg_v1.version}")

    # Set alias champion initially to v1
    client.set_registered_model_alias(model_name, "champion", reg_v1.version)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, "models/model.joblib")
    print(f"Saved best v1 model to models/model.joblib (run_id: {best_run_id})")

    return best_run_id, reg_v1.version, best_pipe


if __name__ == "__main__":
    train_models()
