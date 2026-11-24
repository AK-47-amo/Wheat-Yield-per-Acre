import pandas as pd
import numpy as np

train = pd.read_csv("data/train.csv")
later = pd.read_csv("data/later.csv")
holdout = pd.read_csv("data/holdout_features.csv")

print("--- 1. LEAKING COLUMN ---")
extra_cols = [c for c in train.columns if c not in holdout.columns and c != "yield_qtl_per_acre"]
print("Extra columns in train not in holdout:", extra_cols)
for c in extra_cols:
    corr = train[[c, "yield_qtl_per_acre"]].dropna().corr().iloc[0, 1]
    print(f"Correlation of {c} with yield_qtl_per_acre: {corr}")

print("\n--- 2. UNIT MIXUP ---")
for col in ["rainfall_mm", "fertilizer_kg", "sowing_delay_days", "irrigation_count", "soil_ph", "farm_acres"]:
    print(f"Col {col} mean by tehsil:")
    print(train.groupby("tehsil")[col].agg(["count", "min", "mean", "max"]))

print("\n--- 3. SENTINEL VALUE ---")
for col in ["rainfall_mm", "fertilizer_kg", "sowing_delay_days", "irrigation_count", "soil_ph", "farm_acres"]:
    neg = (train[col] < 0).sum()
    if neg > 0:
        print(f"Column {col} has negative values: {train[train[col] < 0][col].value_counts().to_dict()}")
    # Check common sentinels: 999, 9999, -1, -999
    for s in [-1, -999, 999, 9999]:
        cnt = (train[col] == s).sum()
        if cnt > 0:
            print(f"Column {col} has sentinel {s}: count {cnt}")

print("\n--- 4. DUPLICATE ROWS ---")
dup_count = train.duplicated().sum()
print(f"Duplicates (all cols): {dup_count}")
dup_no_id = train.drop(columns=["record_id"]).duplicated().sum()
print(f"Duplicates (excluding record_id): {dup_no_id}")

print("\n--- 5. STUCK SENSOR ---")
train_sorted = train.sort_values("date").reset_index(drop=True)
for col in ["rainfall_mm", "fertilizer_kg", "sowing_delay_days", "irrigation_count", "soil_ph", "farm_acres"]:
    # check runs of identical values
    s = train_sorted[col]
    runs = (s != s.shift()).cumsum()
    run_lengths = train_sorted.groupby(runs).agg(
        val=(col, 'first'),
        cnt=(col, 'count'),
        start_date=('date', 'first'),
        end_date=('date', 'last')
    )
    long_runs = run_lengths[run_lengths['cnt'] >= 10]
    if len(long_runs) > 0:
        print(f"Column {col} long runs:")
        print(long_runs)

print("\n--- 6. TARGET TYPOS ---")
targets = train["yield_qtl_per_acre"].sort_values()
print("Top 15 target values:")
print(targets.tail(15).values)
print("Quantiles:")
print(targets.describe(percentiles=[0.5, 0.9, 0.95, 0.98, 0.99, 0.995]))

print("\n--- 7. SPELLING VARIANTS ---")
for col in ["seed_variety", "tehsil"]:
    print(f"Value counts for {col} in train:")
    print(train[col].value_counts())
    print(f"Value counts for {col} in later:")
    print(later[col].value_counts())
