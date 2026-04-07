import logging
import os
import time
from datetime import datetime
from typing import Optional

import requests
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Baseline weights — only used as floor, not ceiling ───────────────────────
# Structural risk that doesn't change from news alone
STRUCTURAL_RISK = {
    "CN": 30, "US": 15, "DE": 10, "IN": 20,
    "TW": 35, "SA": 25, "RU": 60, "UA": 65,
    "IR": 55,
}

COUNTRY_NAMES = {
    "CN": "China", "US": "United States", "DE": "Germany",
    "IN": "India",  "TW": "Taiwan",        "SA": "Saudi Arabia",
    "RU": "Russia", "UA": "Ukraine",       "IR": "Iran",
}

# Search queries per country — what to look for in news
COUNTRY_QUERIES = {
    "CN": "China trade supply chain tariff",
    "US": "United States trade war tariff supply chain",
    "DE": "Germany economy supply chain manufacturing",
    "IN": "India supply chain trade logistics",
    "TW": "Taiwan semiconductor supply chain strait",
    "SA": "Saudi Arabia oil supply OPEC",
    "RU": "Russia sanctions war supply chain",
    "UA": "Ukraine war conflict supply disruption",
    "IR": "Iran sanctions Hormuz oil supply",
}

# Keywords that signal supply chain disruption — weighted
DISRUPTION_KEYWORDS = {
    "war":          1.0,
    "conflict":     0.9,
    "sanctions":    0.9,
    "blockade":     1.0,
    "strike":       0.8,
    "shutdown":     0.8,
    "disruption":   0.75,
    "shortage":     0.75,
    "tariff":       0.7,
    "ban":          0.7,
    "attack":       0.9,
    "tension":      0.6,
    "protest":      0.5,
    "delay":        0.6,
    "crisis":       0.85,
    "embargo":      0.95,
    "explosion":    0.9,
    "flood":        0.7,
    "earthquake":   0.7,
}


# ── Schema ────────────────────────────────────────────────────────────────────
class GeoRiskRecord(BaseModel):
    country_code:     str
    country_name:     str
    base_risk_score:  float
    news_risk_score:  float        # derived from live news
    composite_score:  float        # final weighted score
    conflict_active:  bool
    sanctions_active: bool
    risk_tier:        str
    top_headlines:    list[str]    # what drove the score
    fetched_at:       datetime


# ── Fetcher ───────────────────────────────────────────────────────────────────
class GeoFetcher:
    BASE_URL = "https://newsdata.io/api/1/news"

    def __init__(self, regions: list[str]):
        self.regions = regions
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("NEWS_API_KEY not set in .env")

    def _fetch_country_news(self, country_code: str) -> list[dict]:
        """Fetch latest news for a specific country."""
        query = COUNTRY_QUERIES.get(country_code, country_code)
        params = {
            "apikey":   self.api_key,
            "q":        query,
            "language": "en",
            "size":     5,        # 5 articles per country
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                logger.warning(f"API error for {country_code}: {data.get('message')}")
                return []

            return data.get("results", [])

        except requests.Timeout:
            logger.warning(f"Timeout fetching news for {country_code}")
            return []
        except Exception as e:
            logger.warning(f"Failed fetching news for {country_code}: {e}")
            return []

    def _score_articles(self, articles: list[dict]) -> tuple[float, bool, bool, list[str]]:
        """
        Score news articles for a country.
        Returns: (news_risk_score, conflict_active, sanctions_active, top_headlines)
        """
        if not articles:
            return 0.0, False, False, []

        total_score    = 0.0
        conflict       = False
        sanctions      = False
        headlines      = []

        for article in articles:
            text = f"{article.get('title','')} {article.get('description','')}".lower()
            headlines.append(article.get("title", "")[:80])

            article_score = 0.0
            for keyword, weight in DISRUPTION_KEYWORDS.items():
                if keyword in text:
                    article_score = max(article_score, weight)
                    if keyword in ("war", "conflict", "attack", "explosion"):
                        conflict = True
                    if keyword in ("sanctions", "embargo", "ban"):
                        sanctions = True

            total_score += article_score

        # Average across articles, scale to 0–100
        news_risk = min((total_score / len(articles)) * 100, 100)
        return round(news_risk, 2), conflict, sanctions, headlines[:3]

    def _assign_tier(self, score: float) -> str:
        if score >= 80: return "CRITICAL"
        if score >= 60: return "HIGH"
        if score >= 40: return "MEDIUM"
        return "LOW"

    def fetch(self) -> list[GeoRiskRecord]:
        records = []

        for code in self.regions:
            logger.info(f"Fetching live news risk for {code} ({COUNTRY_NAMES.get(code, code)})...")

            # Fetch live news
            articles = self._fetch_country_news(code)

            # Score articles
            news_score, conflict, sanctions, headlines = self._score_articles(articles)

            # Structural baseline (floor)
            base = STRUCTURAL_RISK.get(code, 30)

            # Add sanctions/conflict bonuses on top of news score
            bonus = 0
            if conflict:  bonus += 15
            if sanctions: bonus += 10

            # Composite: 40% structural + 60% live news + bonuses
            composite = min(
                (base * 0.4) + (news_score * 0.6) + bonus,
                100
            )

            record = GeoRiskRecord(
                country_code=code,
                country_name=COUNTRY_NAMES.get(code, code),
                base_risk_score=base,
                news_risk_score=news_score,
                composite_score=round(composite, 2),
                conflict_active=conflict,
                sanctions_active=sanctions,
                risk_tier=self._assign_tier(composite),
                top_headlines=headlines,
                fetched_at=datetime.utcnow(),
            )
            records.append(record)

            logger.info(
                f"  {code} → base:{base} | news:{news_score} | "
                f"composite:{composite:.1f} | tier:{record.risk_tier}"
            )

            # Respect API rate limits — 1 request/second on free tier
            time.sleep(1)

        return records