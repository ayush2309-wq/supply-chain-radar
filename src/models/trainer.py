import logging
import numpy as np
import pandas as pd
from src.models.risk_classifier import RiskClassifier, FEATURE_COLS

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    n = n_samples

    data = {
        "pct_change_oil":       np.random.normal(0, 1.5, n),
        "pct_change_copper":    np.random.normal(0, 1.0, n),
        "pct_change_freight":   np.random.normal(0, 1.5, n),
        "pct_change_gas":       np.random.normal(0, 1.2, n),
        "pct_change_aluminum":  np.random.normal(0, 1.0, n),
        "rolling_7d_std_oil":      np.abs(np.random.normal(1.5, 1.0, n)),
        "rolling_7d_std_copper":   np.abs(np.random.normal(0.05, 0.03, n)),
        "rolling_7d_std_freight":  np.abs(np.random.normal(0.15, 0.1, n)),
        "volatility_flag_oil":      np.random.randint(0, 2, n).astype(float),
        "volatility_flag_copper":   np.random.randint(0, 2, n).astype(float),
        "volatility_flag_freight":  np.random.randint(0, 2, n).astype(float),
        "avg_urgency":    np.where(np.random.rand(n) < 0.6,
                            np.random.uniform(0, 0.3, n),
                            np.where(np.random.rand(n) < 0.6,
                                np.random.uniform(0.3, 0.7, n),
                                np.random.uniform(0.7, 1.0, n))),
        "max_urgency":    np.where(np.random.rand(n) < 0.6,
                            np.random.uniform(0, 0.4, n),
                            np.where(np.random.rand(n) < 0.6,
                                np.random.uniform(0.4, 0.8, n),
                                np.random.uniform(0.8, 1.0, n))),
        "article_count":  np.random.randint(0, 15, n).astype(float),
        "avg_sentiment":  np.where(np.random.rand(n) < 0.6,
                            np.random.uniform(0, 0.3, n),
                            np.where(np.random.rand(n) < 0.6,
                                np.random.uniform(0.3, 0.7, n),
                                np.random.uniform(0.7, 1.0, n))),
        "max_sentiment":  np.where(np.random.rand(n) < 0.6,
                            np.random.uniform(0, 0.4, n),
                            np.where(np.random.rand(n) < 0.6,
                                np.random.uniform(0.4, 0.8, n),
                                np.random.uniform(0.8, 1.0, n))),
        "avg_geo_risk":       np.random.uniform(20, 80, n),
        "max_geo_risk":       np.random.uniform(50, 100, n),
        "critical_countries": np.random.randint(0, 5, n).astype(float),
    }

    df = pd.DataFrame(data)

    risk_score = (
        np.abs(df["pct_change_oil"])     * 2.0 +
        np.abs(df["pct_change_freight"]) * 2.0 +
        df["avg_urgency"]                * 20  +
        df["avg_sentiment"]              * 25  +
        df["avg_geo_risk"]               * 0.2 +
        df["critical_countries"]         * 4   +
        df["volatility_flag_freight"]    * 8   +
        df["max_urgency"]                * 12
    )
    risk_score += np.random.normal(0, 2, n)

    labels = np.zeros(n, dtype=int)
    labels[risk_score > np.percentile(risk_score, 50)] = 1
    labels[risk_score > np.percentile(risk_score, 80)] = 2

    logger.info(f"Synthetic data: {n} samples | LOW:{(labels==0).sum()} MEDIUM:{(labels==1).sum()} HIGH:{(labels==2).sum()}")
    return df[FEATURE_COLS].values, labels


def run_training() -> dict:
    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=2000)

    classifier = RiskClassifier()

    logger.info("Training ensemble model...")
    metrics = classifier.train(X, y)

    logger.info("Saving models...")
    classifier.save()

    return {"classifier": classifier, "metrics": metrics}


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    result = run_training()
    print(f"\n── Training Results ──")
    print(f"  XGBoost  CV Accuracy : {result['metrics']['xgb_cv']:.4f}")
    print(f"  LightGBM CV Accuracy : {result['metrics']['lgbm_cv']:.4f}")
    print(f"  Models saved to      : data/models/")