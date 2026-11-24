"""
Drift monitoring and model retraining module for Wheat Yield prediction (MP130).
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.wheat_yield.data import clean_dataframe, compute_sha256
from src.wheat_yield.train import build_pipeline, FEATURE_COLS, TARGET_COL, NUMERIC_FEATURES
from sklearn.ensemble import GradientBoostingRegressor


def calculate_psi(reference: pd.Series, target: pd.Series, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Compute Population Stability Index (PSI) between reference and target series using 10 quantile bins.
    """
    ref = reference.dropna().values
    tgt = target.dropna().values

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    tgt_pct = tgt_counts / len(tgt)

    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    tgt_pct = np.where(tgt_pct == 0, epsilon, tgt_pct)

    ref_pct = ref_pct / np.sum(ref_pct)
    tgt_pct = tgt_pct / np.sum(tgt_pct)

    psi_val = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    return float(psi_val)


def evaluate_mae_by_group(model, df: pd.DataFrame, group_col: str = "tehsil", target_col: str = "yield_qtl_per_acre") -> dict:
    """
    Compute MAE for each subgroup of group_col.
    """
    results = {}
    for val in sorted(df[group_col].unique()):
        grp = df[df[group_col] == val]
        X = grp[FEATURE_COLS]
        y = grp[target_col]
        preds = model.predict(X)
        mae = mean_absolute_error(y, preds)
        results[val] = float(round(mae, 4))
    return results


def run_monitoring_and_retrain():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    tracking_uri = params["mlflow"]["tracking_uri"]
    exp_name = params["mlflow"]["experiment_name"]
    model_name = params["mlflow"]["registered_model_name"]
    data_sha = compute_sha256("data/raw/train.csv")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    client = MlflowClient()

    # Load and clean datasets
    train_raw = pd.read_csv("data/raw/train.csv")
    train_clean = clean_dataframe(train_raw, is_train=True)

    later_raw = pd.read_csv("data/raw/later.csv")
    later_clean = clean_dataframe(later_raw, is_train=False)

    # 1. Compute PSI for every numeric column
    psi_dict = {}
    for col in NUMERIC_FEATURES:
        psi = calculate_psi(train_clean[col], later_clean[col])
        psi_dict[col] = float(round(psi, 4))

    most_drifted = max(psi_dict, key=psi_dict.get)
    print("PSI results:", psi_dict)
    print("Most drifted feature:", most_drifted)

    # Load v1 champion model
    v1_model = joblib.load("models/model.joblib")

    # 2. Compute v1 MAE by tehsil on later.csv
    mae_by_group = evaluate_mae_by_group(v1_model, later_clean, group_col="tehsil")
    worst_served = max(mae_by_group, key=mae_by_group.get)
    print("MAE by tehsil:", mae_by_group)
    print("Worst served group:", worst_served)

    # 3. Sort later.csv by date. Retrain v2 on train.csv + first 70% of later.csv
    later_sorted = later_clean.sort_values("date").reset_index(drop=True)
    split_idx = int(len(later_sorted) * 0.7)
    later_70 = later_sorted.iloc[:split_idx]
    later_30 = later_sorted.iloc[split_idx:]

    train_v2 = pd.concat([train_clean, later_70], ignore_index=True)

    X_train_v2 = train_v2[FEATURE_COLS]
    y_train_v2 = train_v2[TARGET_COL]

    X_check = later_30[FEATURE_COLS]
    y_check = later_30[TARGET_COL]

    # Evaluate v1 on check set (last 30%)
    preds_v1_check = v1_model.predict(X_check)
    mae_v1_on_check = float(round(mean_absolute_error(y_check, preds_v1_check), 4))

    # Retrain v2
    v2_regressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.04, max_depth=6, random_state=42)
    v2_pipe = build_pipeline(v2_regressor)
    v2_pipe.fit(X_train_v2, y_train_v2)

    preds_v2_check = v2_pipe.predict(X_check)
    mae_v2_on_check = float(round(mean_absolute_error(y_check, preds_v2_check), 4))

    print(f"v1 MAE on check (last 30%): {mae_v1_on_check:.4f}")
    print(f"v2 MAE on check (last 30%): {mae_v2_on_check:.4f}")

    # 4. Promotion rule: promote v2 only if its MAE on last 30% is at least 10% lower than v1's
    # i.e. mae_v2_on_check <= 0.90 * mae_v1_on_check
    improvement = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    print(f"Improvement: {improvement * 100:.2f}%")

    with mlflow.start_run(run_name="Retrain_v2") as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("model_type", "v2_retrain_GradientBoosting")
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)
        mlflow.log_metric("mae", mae_v2_on_check)

        mlflow.sklearn.log_model(
            v2_pipe,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        v2_run_id = run.info.run_id

    reg_v2 = mlflow.register_model(f"runs:/{v2_run_id}/model", model_name)
    print(f"Registered v2 model: {model_name} version {reg_v2.version}")

    if improvement >= 0.10:
        decision = "promote"
        champion_version = reg_v2.version
        client.set_registered_model_alias(model_name, "champion", reg_v2.version)
        joblib.dump(v2_pipe, "models/model.joblib")
        print(f"Decision: PROMOTE v2 as @champion (improvement >= 10%)")
    else:
        decision = "keep"
        champion_version = 1
        client.set_registered_model_alias(model_name, "champion", 1)
        print(f"Decision: KEEP v1 as @champion (improvement < 10%)")

    monitoring_summary = {
        "psi": psi_dict,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served,
        "mae_v1_on_check": mae_v1_on_check,
        "mae_v2_on_check": mae_v2_on_check,
        "decision": decision,
        "champion_version": champion_version
    }
    return monitoring_summary


if __name__ == "__main__":
    run_monitoring_and_retrain()
