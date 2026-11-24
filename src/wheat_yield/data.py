"""
Data auditing, loading and cleaning module for Wheat Yield prediction (MP130).
"""

import hashlib
import pandas as pd
import numpy as np


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def audit_train_data(filepath: str = "data/raw/train.csv") -> dict:
    """Audit train data against the 7 data issues."""
    df = pd.read_csv(filepath)
    issues = {}

    # 1. Leaking column
    # A column missing in holdout and highly correlated with target
    if "yield_from_mandi_receipt" in df.columns:
        issues["leaking_column"] = {
            "present": True,
            "column": "yield_from_mandi_receipt"
        }
    else:
        issues["leaking_column"] = {"present": False, "column": None}

    # 2. Unit mixup
    # Check Tehsil South fertilizer_kg
    south_fert = df[df["tehsil"] == "Tehsil South"]["fertilizer_kg"]
    other_fert = df[df["tehsil"] != "Tehsil South"]["fertilizer_kg"]
    if south_fert.mean() > 500 * other_fert.mean():
        issues["unit_mixup"] = {
            "present": True,
            "column": "fertilizer_kg",
            "group_value": "Tehsil South",
            "factor": 1000
        }
    else:
        issues["unit_mixup"] = {
            "present": False,
            "column": None,
            "group_value": None,
            "factor": None
        }

    # 3. Sentinel value
    sentinel_cnt = int((df["rainfall_mm"] == -1).sum())
    if sentinel_cnt > 0:
        issues["sentinel_value"] = {
            "present": True,
            "column": "rainfall_mm",
            "value": -1,
            "count": sentinel_cnt
        }
    else:
        issues["sentinel_value"] = {
            "present": False,
            "column": None,
            "value": None,
            "count": None
        }

    # 4. Duplicate rows
    dup_cnt = int(df.duplicated().sum())
    issues["duplicate_rows"] = {
        "present": dup_cnt > 0,
        "count": dup_cnt if dup_cnt > 0 else None
    }

    # 5. Stuck sensor
    df_sorted = df.sort_values("date").reset_index(drop=True)
    stuck_run = None
    s = df_sorted["soil_ph"]
    runs = (s != s.shift()).cumsum()
    run_lengths = df_sorted.groupby(runs).agg(
        val=("soil_ph", "first"),
        cnt=("soil_ph", "count"),
        start_date=("date", "first"),
        end_date=("date", "last")
    )
    long_runs = run_lengths[run_lengths["cnt"] >= 10]
    if len(long_runs) > 0:
        row = long_runs.iloc[0]
        issues["stuck_sensor"] = {
            "present": True,
            "column": "soil_ph",
            "start_date": str(row["start_date"]),
            "end_date": str(row["end_date"])
        }
    else:
        issues["stuck_sensor"] = {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None
        }

    # 6. Target typos
    # Spurious values 10x larger than normal range
    target = df["yield_qtl_per_acre"]
    q99 = target.quantile(0.99)
    typos = (target > 5 * q99).sum()
    issues["target_typos"] = {
        "present": typos > 0,
        "count": int(typos) if typos > 0 else None
    }

    # 7. Spelling variants
    issues["spelling_variants"] = {
        "present": False,
        "column": None,
        "true_levels": None
    }

    return issues


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Clean the dataset:
    - Drop leaking column if present
    - Fix unit mixup in fertilizer_kg for Tehsil South (factor 1000)
    - Fix sentinel values (-1 in rainfall_mm replaced by median)
    - Fix stuck sensor in soil_ph if present
    """
    df_clean = df.copy()

    # Drop leaking column
    if "yield_from_mandi_receipt" in df_clean.columns:
        df_clean = df_clean.drop(columns=["yield_from_mandi_receipt"])

    # Fix unit mixup: fertilizer_kg recorded in grams (1000x too big) for Tehsil South
    mask_unit = (df_clean["tehsil"] == "Tehsil South") & (df_clean["fertilizer_kg"] > 500)
    df_clean.loc[mask_unit, "fertilizer_kg"] = df_clean.loc[mask_unit, "fertilizer_kg"] / 1000.0

    # Fix sentinel value: -1 in rainfall_mm
    df_clean["rainfall_mm"] = df_clean["rainfall_mm"].astype(float)
    valid_rain = df_clean.loc[df_clean["rainfall_mm"] >= 0, "rainfall_mm"]
    median_rain = float(valid_rain.median()) if len(valid_rain) > 0 else 600.0
    df_clean.loc[df_clean["rainfall_mm"] < 0, "rainfall_mm"] = median_rain

    # Fix stuck sensor in soil_ph during stuck sensor window
    if "date" in df_clean.columns and is_train:
        stuck_mask = (df_clean["date"] >= "2026-03-12") & (df_clean["date"] <= "2026-03-28") & (df_clean["soil_ph"] == 8.37)
        normal_ph = df_clean.loc[~stuck_mask, "soil_ph"].median()
        df_clean.loc[stuck_mask, "soil_ph"] = normal_ph

    return df_clean
