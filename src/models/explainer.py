import logging
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.models.risk_classifier import FEATURE_COLS

logger = logging.getLogger(__name__)


class RiskExplainer:
    """
    Uses SHAP to explain WHY the model predicted a given risk level.
    Critical for business trust — 'why is this route HIGH risk?'
    """

    def __init__(self, model, scaler):
        self.model   = model
        self.scaler  = scaler
        self.explainer = None

    def fit(self, X_background: np.ndarray):
        """Fit SHAP explainer on background data sample."""
        X_scaled = self.scaler.transform(X_background)
        background = shap.sample(X_scaled, min(100, len(X_scaled)))
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba, background
        )
        logger.info("SHAP explainer fitted.")

    def explain(self, df: pd.DataFrame) -> dict:
        """Return top factors driving risk for each row."""
        if self.explainer is None:
            raise RuntimeError("Call fit() before explain()")

        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].reindex(columns=FEATURE_COLS, fill_value=0).values
        X_scaled = self.scaler.transform(X)

        shap_values = self.explainer.shap_values(X_scaled, nsamples=100)

        results = []
        for i in range(len(df)):
            # shap_values[2] = HIGH risk class contributions
            vals = shap_values[2][i] if isinstance(shap_values, list) else shap_values[i]
            top_factors = sorted(
                zip(FEATURE_COLS, vals),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            results.append({
                "top_risk_drivers": [
                    {"feature": f, "impact": round(float(v), 4)}
                    for f, v in top_factors
                ]
            })

        return results

    def save_summary_plot(self, X: np.ndarray, path: str = "data/models/shap_summary.png"):
        """Save SHAP summary plot."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        X_scaled = self.scaler.transform(X)
        shap_values = self.explainer.shap_values(X_scaled, nsamples=50)
        shap.summary_plot(
            shap_values[2] if isinstance(shap_values, list) else shap_values,
            X_scaled,
            feature_names=FEATURE_COLS,
            show=False,
        )
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"SHAP summary plot saved → {path}")