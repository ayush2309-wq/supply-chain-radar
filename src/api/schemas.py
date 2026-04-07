from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class RiskPredictionResponse(BaseModel):
    date: str
    risk_label: str
    risk_score: float
    prob_low: float
    prob_medium: float
    prob_high: float


class ForecastResponse(BaseModel):
    date: str
    risk_score: float
    risk_label: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


class PipelineResponse(BaseModel):
    status: str
    records_ingested: dict
    matrix_shape: list
    timestamp: datetime