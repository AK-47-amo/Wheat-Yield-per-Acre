import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.wheat_yield.train import build_pipeline, FEATURE_COLS
from src.wheat_yield.drift import calculate_psi


def test_pipeline_fit_predict():
    X = pd.DataFrame({
        "rainfall_mm": [500.0, 600.0, 700.0, 800.0],
        "fertilizer_kg": [50.0, 60.0, 70.0, 80.0],
        "sowing_delay_days": [5, 10, 15, 20],
        "irrigation_count": [2, 3, 4, 5],
        "soil_ph": [6.5, 7.0, 7.5, 6.8],
        "farm_acres": [2.0, 4.0, 6.0, 8.0],
        "seed_variety": ["PBW-343", "DBW-187", "HD-2967", "PBW-343"],
        "tehsil": ["Tehsil North", "Tehsil South", "Tehsil West", "Tehsil North"]
    })
    y = pd.Series([15.0, 18.0, 21.0, 24.0])

    pipeline = build_pipeline(RandomForestRegressor(n_estimators=10, random_state=42))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == 4
    assert isinstance(preds[0], (float, np.floating))


def test_calculate_psi_identical_distributions():
    s1 = pd.Series(np.random.normal(50, 10, 1000))
    s2 = s1.copy()
    psi = calculate_psi(s1, s2)
    assert psi < 0.05
