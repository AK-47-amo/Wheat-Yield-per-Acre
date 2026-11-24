import pandas as pd
import numpy as np
from src.wheat_yield.data import clean_dataframe


def test_clean_leaking_column():
    df = pd.DataFrame({
        "rainfall_mm": [500, 600],
        "fertilizer_kg": [50.0, 60.0],
        "sowing_delay_days": [5, 10],
        "irrigation_count": [3, 4],
        "soil_ph": [6.5, 7.0],
        "farm_acres": [5.0, 6.0],
        "seed_variety": ["PBW-343", "DBW-187"],
        "tehsil": ["Tehsil North", "Tehsil West"],
        "yield_from_mandi_receipt": [15.2, 18.4],
        "yield_qtl_per_acre": [15.1, 18.5]
    })
    clean = clean_dataframe(df)
    assert "yield_from_mandi_receipt" not in clean.columns
    assert "yield_qtl_per_acre" in clean.columns


def test_clean_unit_mixup():
    df = pd.DataFrame({
        "rainfall_mm": [500, 600],
        "fertilizer_kg": [85000.0, 85.0],
        "sowing_delay_days": [5, 10],
        "irrigation_count": [3, 4],
        "soil_ph": [6.5, 7.0],
        "farm_acres": [5.0, 6.0],
        "seed_variety": ["PBW-343", "DBW-187"],
        "tehsil": ["Tehsil South", "Tehsil North"]
    })
    clean = clean_dataframe(df)
    # 85000 in Tehsil South should be converted to 85.0 (divided by 1000)
    assert clean.loc[0, "fertilizer_kg"] == 85.0
    assert clean.loc[1, "fertilizer_kg"] == 85.0


def test_clean_sentinel_value():
    df = pd.DataFrame({
        "rainfall_mm": [-1, 700.0],
        "fertilizer_kg": [50.0, 60.0],
        "sowing_delay_days": [5, 10],
        "irrigation_count": [3, 4],
        "soil_ph": [6.5, 7.0],
        "farm_acres": [5.0, 6.0],
        "seed_variety": ["PBW-343", "DBW-187"],
        "tehsil": ["Tehsil North", "Tehsil West"]
    })
    clean = clean_dataframe(df)
    assert clean.loc[0, "rainfall_mm"] == 700.0
