import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class ValidatedPriceFeature(BaseModel):
    date: datetime
    label: str
    close: float
    pct_change: float
    rolling_7d_avg: Optional[float] = None
    rolling_7d_std: Optional[float] = None
    volatility_flag: bool = False  # True if std > 2% — abnormal movement

    @field_validator("close")
    @classmethod
    def close_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Close price must be positive, got {v}")
        return v


class ValidatedNewsFeature(BaseModel):
    published_at: datetime
    source: str
    title: str
    raw_sentiment: Optional[float] = None   # filled in Phase 3
    risk_keywords_found: list[str] = []
    urgency_score: float = 0.0              # 0–1 based on keyword weight


class ValidatedGeoFeature(BaseModel):
    country_code: str
    composite_score: float
    risk_tier: str  # LOW / MEDIUM / HIGH / CRITICAL

    @field_validator("risk_tier", mode="before")
    @classmethod
    def assign_tier(cls, v, info):
        score = info.data.get("composite_score", 0)
        if score >= 80:   return "CRITICAL"
        if score >= 60:   return "HIGH"
        if score >= 40:   return "MEDIUM"
        return "LOW"