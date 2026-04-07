import logging
import os
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class RiskForecaster:
    """
    Uses Facebook Prophet to forecast supply chain risk scores
    2-4 weeks into the future using historical price + sentiment signals.
    """

    def __init__(self, forecast_days: int = 28):
        self.forecast_days = forecast_days
        self.models        = {}   # one Prophet model per signal
        self.forecasts     = {}
        self.is_trained    = False

    # ── Prepare time series per signal ───────────────────────────────────────
    @staticmethod
    def _prep_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Prophet requires columns: ds (date) and y (value)."""
        series = df[["date", col]].copy()
        series.columns = ["ds", "y"]
        series["ds"] = pd.to_datetime(series["ds"])
        series = series.dropna().sort_values("ds")
        return series

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame):
        """
        Train one Prophet model per key signal column.
        """
        signals = [
            "close_oil", "close_freight", "close_copper",
            "pct_change_oil", "pct_change_freight",
            "avg_urgency", "avg_geo_risk",
        ]

        available = [s for s in signals if s in df.columns]
        logger.info(f"Training Prophet on {len(available)} signals...")

        for signal in available:
            series = self._prep_series(df, signal)

            if len(series) < 5:
                logger.warning(f"Not enough data for {signal}, skipping.")
                continue

            model = Prophet(
                changepoint_prior_scale=0.15,   # flexibility to catch sudden changes
                seasonality_prior_scale=10,
                seasonality_mode="multiplicative",
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.90,            # 90% confidence interval
            )

            # Add US market holidays as a regressor
            model.add_country_holidays(country_name="US")

            model.fit(series)
            self.models[signal] = model
            logger.info(f"  ✓ Trained Prophet for {signal}")

        self.is_trained = True
        logger.info(f"Forecasting trained on {len(self.models)} signals.")

    # ── Forecast ──────────────────────────────────────────────────────────────
    def forecast(self) -> dict[str, pd.DataFrame]:
        """
        Generate forecast for each signal for next N days.
        Returns dict of {signal: forecast_df}
        """
        if not self.is_trained:
            raise RuntimeError("Call train() first.")

        for signal, model in self.models.items():
            future = model.make_future_dataframe(
                periods=self.forecast_days,
                freq="B",          # Business days only
            )
            forecast = model.predict(future)
            self.forecasts[signal] = forecast[[
                "ds", "yhat", "yhat_lower", "yhat_upper",
                "trend", "weekly"
            ]].tail(self.forecast_days)  # only future rows

            logger.info(f"  ✓ Forecast ready for {signal} ({self.forecast_days} days)")

        return self.forecasts

    # ── Risk Forecast Score ───────────────────────────────────────────────────
    def build_risk_forecast(self) -> pd.DataFrame:
        """
        Combine all signal forecasts into a single daily risk score (0-100).
        Higher = more likely disruption.
        """
        if not self.forecasts:
            raise RuntimeError("Call forecast() first.")

        # Weights per signal for composite risk score
        weights = {
            "pct_change_oil":     0.25,
            "pct_change_freight": 0.25,
            "avg_urgency":        0.25,
            "avg_geo_risk":       0.15,
            "close_freight":      0.10,
        }

        # Build date index from oil forecast (always present)
        base_signal = "close_oil" if "close_oil" in self.forecasts else list(self.forecasts.keys())[0]
        dates = self.forecasts[base_signal]["ds"].values
        composite = np.zeros(len(dates))

        for signal, weight in weights.items():
            if signal not in self.forecasts:
                continue

            forecast_df = self.forecasts[signal]
            values      = forecast_df["yhat"].values

            # Normalize to 0-100 range
            v_min, v_max = values.min(), values.max()
            if v_max > v_min:
                normalized = (values - v_min) / (v_max - v_min) * 100
            else:
                normalized = np.full(len(values), 50.0)

            composite += normalized * weight

        # Classify composite score
        risk_labels = pd.cut(
            composite,
            bins=[0, 33, 66, 100],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )

        result = pd.DataFrame({
            "date":        pd.to_datetime(dates),
            "risk_score":  composite.round(2),
            "risk_label":  risk_labels,
        })

        return result

    # ── Save Plots ────────────────────────────────────────────────────────────
    def save_forecast_plots(self, path: str = "data/forecasts"):
        os.makedirs(path, exist_ok=True)

        for signal, model in self.models.items():
            forecast = model.predict(
                model.make_future_dataframe(periods=self.forecast_days, freq="B")
            )
            fig = model.plot(forecast)
            fig.suptitle(f"Forecast: {signal}", fontsize=14)
            fig.savefig(f"{path}/{signal}_forecast.png", bbox_inches="tight", dpi=120)
            plt.close(fig)
            logger.info(f"Plot saved → {path}/{signal}_forecast.png")


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Load feature matrix
    df = pd.read_csv("data/features/feature_matrix.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Train & forecast
    forecaster = RiskForecaster(forecast_days=28)
    forecaster.train(df)
    forecasts  = forecaster.forecast()

    # Build composite risk forecast
    risk_forecast = forecaster.build_risk_forecast()

    print("\n── 4-Week Supply Chain Risk Forecast ──")
    print(risk_forecast.to_string(index=False))

    # Save plots
    forecaster.save_forecast_plots()
    print("\n✓ Forecast plots saved to data/forecasts/")