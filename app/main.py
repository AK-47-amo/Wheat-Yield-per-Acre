import logging
import sys
import os
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pythonjsonlogger import jsonlogger

from app.schemas import PredictionRequest, PredictionResponse

# Configure structured JSON logger to stdout
logger = logging.getLogger("mp130_api")
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s"
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully", extra={"model_path": MODEL_PATH})
    else:
        logger.warning("Model file not found during startup", extra={"model_path": MODEL_PATH})
    yield


app = FastAPI(title="Wheat Yield Prediction API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")

    features_dict = payload.model_dump()
    df = pd.DataFrame([features_dict])

    pred = float(model.predict(df)[0])

    # Structured JSON log per prediction without personal details
    logger.info(
        "Prediction requested",
        extra={
            "inputs": {
                "rainfall_mm": payload.rainfall_mm,
                "fertilizer_kg": payload.fertilizer_kg,
                "sowing_delay_days": payload.sowing_delay_days,
                "irrigation_count": payload.irrigation_count,
                "soil_ph": payload.soil_ph,
                "farm_acres": payload.farm_acres,
                "seed_variety": payload.seed_variety,
                "tehsil": payload.tehsil
            },
            "prediction": pred
        }
    )

    return PredictionResponse(prediction=pred)
