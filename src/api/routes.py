import logging
import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import (
    RiskPredictionResponse,
    ForecastResponse,
    HealthResponse,
    PipelineResponse,
)
from src.models.risk_classifier import RiskClassifier
from src.models.forecaster import RiskForecaster
from src.processing.pipeline import run_processing

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Load models at startup ────────────────────────────────────────────────────
classifier = RiskClassifier()
try:
    classifier.load()
    logger.info("Risk classifier loaded.")
except Exception as e:
    logger.warning(f"Classifier not loaded: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@router.get("/risk/current", response_model=list[RiskPredictionResponse])
def get_current_risk():
    """Return risk scores for all available dates in feature matrix."""
    try:
        df = pd.read_csv("data/features/feature_matrix.csv")
        predictions = classifier.predict(df)
        return [
            RiskPredictionResponse(**row)
            for row in predictions.to_dict(orient="records")
        ]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature matrix not found. Run pipeline first.")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/latest", response_model=RiskPredictionResponse)
def get_latest_risk():
    """Return risk score for the most recent date only."""
    try:
        df = pd.read_csv("data/features/feature_matrix.csv")
        predictions = classifier.predict(df)
        latest = predictions.iloc[-1]
        return RiskPredictionResponse(**latest.to_dict())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Feature matrix not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=list[ForecastResponse])
def get_forecast():
    """Return 28-day supply chain risk forecast."""
    try:
        df = pd.read_csv("data/features/feature_matrix.csv")
        df["date"] = pd.to_datetime(df["date"])

        forecaster = RiskForecaster(forecast_days=28)
        forecaster.train(df)
        forecaster.forecast()
        risk_forecast = forecaster.build_risk_forecast()

        risk_forecast["date"] = risk_forecast["date"].dt.strftime("%Y-%m-%d")
        risk_forecast["risk_label"] = risk_forecast["risk_label"].astype(str)

        return [
            ForecastResponse(**row)
            for row in risk_forecast.to_dict(orient="records")
        ]
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline/run", response_model=PipelineResponse)
def run_pipeline(background_tasks: BackgroundTasks):
    """Trigger full data ingestion + feature engineering pipeline."""
    try:
        from src.ingestion.pipeline import run_ingestion
        raw = run_ingestion()
        matrix = run_processing()

        return PipelineResponse(
            status="success",
            records_ingested={k: len(v) for k, v in raw.items()},
            matrix_shape=list(matrix.shape),
            timestamp=datetime.utcnow(),
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geo/risk")
def get_geo_risk():
    """Return current geopolitical risk scores per country."""
    from src.ingestion.geo_fetcher import GeoFetcher
    from src.processing.feature_store import FeatureStore
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    regions = config["ingestion"]["geo"]["regions"]
    fetcher = GeoFetcher(regions=regions)
    records = fetcher.fetch()
    features = FeatureStore.build_geo_features(records)

    return features.to_dict(orient="records")