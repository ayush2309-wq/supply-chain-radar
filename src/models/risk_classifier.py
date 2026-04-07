import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "pct_change_oil", "pct_change_copper", "pct_change_freight",
    "pct_change_gas", "pct_change_aluminum",
    "rolling_7d_std_oil", "rolling_7d_std_copper", "rolling_7d_std_freight",
    "volatility_flag_oil", "volatility_flag_copper", "volatility_flag_freight",
    "avg_urgency", "max_urgency", "article_count",
    "avg_sentiment", "max_sentiment",
    "avg_geo_risk", "max_geo_risk", "critical_countries",
]


class RiskClassifier:
    """
    Ensemble of XGBoost + LightGBM that predicts supply chain
    disruption risk as: LOW(0) / MEDIUM(1) / HIGH(2)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.lgbm = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self.is_trained = False
        self.feature_cols = FEATURE_COLS

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Select + scale features."""
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].copy()

        # Fill any missing feature columns with 0
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self.feature_cols]
        
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].map({"True": 1.0, "False": 0.0}).fillna(0.0)
        X = X.fillna(0).astype(float)
        return X.values

    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Training XGBoost...")
        self.xgb.fit(X_scaled, y)

        logger.info("Training LightGBM...")
        self.lgbm.fit(X_scaled, y)

        # Cross-val on both
        xgb_cv = cross_val_score(self.xgb, X_scaled, y, cv=5, scoring="accuracy").mean()
        lgbm_cv = cross_val_score(self.lgbm, X_scaled, y, cv=5, scoring="accuracy").mean()

        logger.info(f"XGBoost CV Accuracy : {xgb_cv:.4f}")
        logger.info(f"LightGBM CV Accuracy: {lgbm_cv:.4f}")

        self.is_trained = True
        return {"xgb_cv": xgb_cv, "lgbm_cv": lgbm_cv}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._prepare(df)
        X_scaled = self.scaler.transform(X)

        xgb_proba  = self.xgb.predict_proba(X_scaled)
        lgbm_proba = self.lgbm.predict_proba(X_scaled)

        # Ensemble: average probabilities
        avg_proba  = (xgb_proba + lgbm_proba) / 2
        prob_high   = avg_proba[:, 2]
        prob_medium = avg_proba[:, 1]
        predictions = np.where(prob_high > 0.003, 2,
              np.where(prob_medium > 0.015, 1, 0))

        label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        result = df[["date"]].copy() if "date" in df.columns else pd.DataFrame()
        result["risk_label"]       = [label_map[p] for p in predictions]
        result["risk_score"]       = avg_proba[:, 2] * 100  # HIGH risk probability %
        result["prob_low"]         = (avg_proba[:, 0] * 100).round(2)
        result["prob_medium"]      = (avg_proba[:, 1] * 100).round(2)
        result["prob_high"]        = (avg_proba[:, 2] * 100).round(2)

        return result

    def save(self, path: str = "data/models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.xgb,    f"{path}/xgb_model.pkl")
        joblib.dump(self.lgbm,   f"{path}/lgbm_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        logger.info(f"Models saved to {path}/")

    def load(self, path: str = "data/models"):
        self.xgb    = joblib.load(f"{path}/xgb_model.pkl")
        self.lgbm   = joblib.load(f"{path}/lgbm_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.is_trained = True
        logger.info(f"Models loaded from {path}/")