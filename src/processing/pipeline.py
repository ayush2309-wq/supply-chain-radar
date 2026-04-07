import logging
import os

import pandas as pd

from src.ingestion.pipeline import run_ingestion
from src.processing.feature_store import FeatureStore

logger = logging.getLogger(__name__)


def run_processing() -> pd.DataFrame:
    # Step 1 — pull raw data
    logger.info("Starting ingestion...")
    raw = run_ingestion()

    # Step 2 — build feature sets
    logger.info("Building features...")
    price_features = FeatureStore.build_price_features(raw["prices"])
    news_features  = FeatureStore.build_news_features(raw["news"])
    geo_features   = FeatureStore.build_geo_features(raw["geo"])

    # Step 3 — merge into unified matrix
    logger.info("Building unified feature matrix...")
    matrix = FeatureStore.build_unified_matrix(price_features, news_features, geo_features)

    # Step 4 — save to CSV (feature store on disk for now)
    os.makedirs("data/features", exist_ok=True)
    matrix.to_csv("data/features/feature_matrix.csv", index=False)
    logger.info(f"Feature matrix saved → data/features/feature_matrix.csv")

    return matrix


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    matrix = run_processing()
    print(f"\n── Feature Matrix ──")
    print(f"  Shape  : {matrix.shape}")
    print(f"  Columns: {list(matrix.columns)}")
    print(f"\n{matrix.tail(3).to_string()}")