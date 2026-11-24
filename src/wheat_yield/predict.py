"""
Inference module to generate holdout predictions for Wheat Yield (MP130).
"""

import os
import joblib
import pandas as pd
from src.wheat_yield.data import clean_dataframe
from src.wheat_yield.train import FEATURE_COLS


def generate_holdout_predictions(
    model_path: str = "models/model.joblib",
    holdout_path: str = "data/raw/holdout_features.csv",
    output_path: str = "predictions/holdout_predictions.csv"
):
    """
    Generate predictions for holdout_features.csv and save to output_path.
    Output format: exactly two columns: record_id,prediction
    """
    model = joblib.load(model_path)
    holdout_df = pd.read_csv(holdout_path)

    # Clean holdout features
    holdout_clean = clean_dataframe(holdout_df, is_train=False)
    X_holdout = holdout_clean[FEATURE_COLS]

    preds = model.predict(X_holdout)

    out_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": preds
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Generated {len(out_df)} predictions to {output_path}")
    return out_df


if __name__ == "__main__":
    generate_holdout_predictions()
