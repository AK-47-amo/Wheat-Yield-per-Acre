from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid():
    payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 75.0,
        "sowing_delay_days": 12,
        "irrigation_count": 4,
        "soil_ph": 7.2,
        "farm_acres": 8.5,
        "seed_variety": "PBW-343",
        "tehsil": "Tehsil North"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


def test_predict_rule1_farm_acres_above_20_rejected():
    # Rule 1: farm_acres above 20 is impossible. Reject with 422.
    payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 75.0,
        "sowing_delay_days": 12,
        "irrigation_count": 4,
        "soil_ph": 7.2,
        "farm_acres": 25.0,  # Invalid: > 20
        "seed_variety": "PBW-343",
        "tehsil": "Tehsil North"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rule2_fertilizer_below_10_rejected():
    # Rule 2: fertilizer_kg below 10 is a mistake. Reject with 422.
    payload = {
        "rainfall_mm": 650.0,
        "fertilizer_kg": 5.0,  # Invalid: < 10
        "sowing_delay_days": 12,
        "irrigation_count": 4,
        "soil_ph": 7.2,
        "farm_acres": 8.5,
        "seed_variety": "PBW-343",
        "tehsil": "Tehsil North"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
