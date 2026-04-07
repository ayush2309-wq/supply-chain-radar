import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from src.ingestion.news_fetcher import NewsArticle
from src.ingestion.price_fetcher import PriceRecord
from src.ingestion.geo_fetcher import GeoRiskRecord
from src.processing.validator import (
    ValidatedPriceFeature,
    ValidatedNewsFeature,
    ValidatedGeoFeature,
)
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

RISK_KEYWORDS = {
    "port strike":       0.9,
    "factory shutdown":  0.85,
    "trade sanctions":   0.85,
    "shipping delay":    0.75,
    "supply chain":      0.5,
    "freight rate":      0.6,
    "logistics":         0.4,
    "disruption":        0.7,
    "blockade":          0.95,
    "flood":             0.8,
    "earthquake":        0.8,
    "war":               1.0,
    "conflict":          0.9,
}


class FeatureStore:

    # ── Price Features ────────────────────────────────────────────────────────
    @staticmethod
    def build_price_features(records: list[PriceRecord]) -> pd.DataFrame:
        if not records:
            logger.warning("No price records to process.")
            return pd.DataFrame()

        df = pd.DataFrame([r.model_dump() for r in records])
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["label", "date"], inplace=True)

        features = []
        for label, group in df.groupby("label"):
            group = group.copy()
            group["rolling_7d_avg"]  = group["close"].rolling(7, min_periods=1).mean()
            group["rolling_7d_std"]  = group["close"].rolling(7, min_periods=1).std().fillna(0)
            group["volatility_flag"] = group["rolling_7d_std"] / group["rolling_7d_avg"] > 0.02

            for _, row in group.iterrows():
                try:
                    feat = ValidatedPriceFeature(
                        date=row["date"],
                        label=row["label"],
                        close=row["close"],
                        pct_change=row["pct_change"] if pd.notna(row["pct_change"]) else 0.0,
                        rolling_7d_avg=round(row["rolling_7d_avg"], 4),
                        rolling_7d_std=round(row["rolling_7d_std"], 4),
                        volatility_flag=bool(row["volatility_flag"]),
                    )
                    features.append(feat.model_dump())
                except Exception as e:
                    logger.warning(f"Skipping invalid price row: {e}")

        result = pd.DataFrame(features)
        logger.info(f"Built {len(result)} price feature rows across {result['label'].nunique()} commodities")
        return result

    # ── News Features ─────────────────────────────────────────────────────────
    @staticmethod
    def build_news_features(articles: list[NewsArticle], run_sentiment: bool = True) -> pd.DataFrame:
        if not articles:
            logger.warning("No articles to process.")
            return pd.DataFrame()

        features = []
        for article in articles:
            text = f"{article.title} {article.description or ''}".lower()

            found_keywords = []
            urgency = 0.0
            for kw, weight in RISK_KEYWORDS.items():
                if kw in text:
                    found_keywords.append(kw)
                    urgency = max(urgency, weight)

            try:
                feat = ValidatedNewsFeature(
                    published_at=article.published_at,
                    source=article.source,
                    title=article.title,
                    raw_sentiment=None,
                    risk_keywords_found=found_keywords,
                    urgency_score=round(urgency, 4),
                )
                features.append(feat.model_dump())
            except Exception as e:
                logger.warning(f"Skipping invalid article: {e}")

        result = pd.DataFrame(features)

        # FinBERT sentiment
        if run_sentiment and not result.empty:
            logger.info("Running FinBERT sentiment on headlines...")
            analyzer  = SentimentAnalyzer()
            extractor = EntityExtractor()

            titles     = result["title"].tolist()
            sentiments = analyzer.batch_analyze(titles)
            entities   = extractor.batch_extract(titles)

            result["sentiment"]       = [s["sentiment"]  for s in sentiments]
            result["sentiment_score"] = [s["risk_score"] for s in sentiments]
            result["confidence"]      = [s["confidence"] for s in sentiments]
            result["countries"]       = [e["countries"]  for e in entities]
            result["disruption_type"] = [e["disruption_type"] for e in entities]

        logger.info(f"Built {len(result)} news feature rows — avg urgency: {result['urgency_score'].mean():.2f}")
        return result

    # ── Geo Features ──────────────────────────────────────────────────────────
    @staticmethod
    def build_geo_features(records: list[GeoRiskRecord]) -> pd.DataFrame:
        if not records:
            logger.warning("No geo records to process.")
            return pd.DataFrame()

        features = []
        for record in records:
            try:
                feat = ValidatedGeoFeature(
                    country_code=record.country_code,
                    composite_score=record.composite_score,
                    risk_tier="",
                )
                features.append(feat.model_dump())
            except Exception as e:
                logger.warning(f"Skipping invalid geo record: {e}")

        result = pd.DataFrame(features)
        logger.info(f"Built {len(result)} geo feature rows")
        return result

    # ── Unified Feature Matrix ────────────────────────────────────────────────
    @staticmethod
    def build_unified_matrix(
        price_features: pd.DataFrame,
        news_features:  pd.DataFrame,
        geo_features:   pd.DataFrame,
    ) -> pd.DataFrame:
        if price_features.empty:
            logger.error("Cannot build matrix — price features are empty.")
            return pd.DataFrame()

        price_pivot = price_features.pivot_table(
            index="date",
            columns="label",
            values=["close", "pct_change", "rolling_7d_avg", "rolling_7d_std", "volatility_flag"],
            aggfunc="first",
        )
        price_pivot.columns = ["_".join(col) for col in price_pivot.columns]
        price_pivot.reset_index(inplace=True)

        if not news_features.empty:
            news_features["published_at"] = pd.to_datetime(news_features["published_at"]).dt.normalize()
            daily_news = news_features.groupby("published_at").agg(
                avg_urgency=("urgency_score",    "mean"),
                max_urgency=("urgency_score",    "max"),
                article_count=("title",          "count"),
                avg_sentiment=("sentiment_score","mean") if "sentiment_score" in news_features.columns else ("urgency_score", "mean"),
                max_sentiment=("sentiment_score","max")  if "sentiment_score" in news_features.columns else ("urgency_score", "max"),
            ).reset_index().rename(columns={"published_at": "date"})
        else:
            daily_news = pd.DataFrame(columns=["date","avg_urgency","max_urgency","article_count"])

        if not geo_features.empty:
            geo_summary = pd.DataFrame({
                "avg_geo_risk":        [geo_features["composite_score"].mean()],
                "max_geo_risk":        [geo_features["composite_score"].max()],
                "critical_countries":  [(geo_features["risk_tier"] == "CRITICAL").sum()],
            })
        else:
            geo_summary = pd.DataFrame()

        matrix = pd.merge(price_pivot, daily_news, on="date", how="left")

        if not geo_summary.empty:
            for col in geo_summary.columns:
                matrix[col] = geo_summary[col].iloc[0]

        matrix.fillna(0, inplace=True)
        matrix["fetched_at"] = datetime.utcnow()

        logger.info(f"Unified matrix: {matrix.shape[0]} rows × {matrix.shape[1]} columns")
        return matrix