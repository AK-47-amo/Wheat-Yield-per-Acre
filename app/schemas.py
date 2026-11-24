from pydantic import BaseModel, Field
from typing import Literal


class PredictionRequest(BaseModel):
    rainfall_mm: float
    fertilizer_kg: float = Field(..., ge=10, description="Fertilizer in kg, must be >= 10")
    sowing_delay_days: int
    irrigation_count: int
    soil_ph: float
    farm_acres: float = Field(..., le=20, description="Farm acres, must be <= 20")
    seed_variety: str
    tehsil: str


class PredictionResponse(BaseModel):
    prediction: float
