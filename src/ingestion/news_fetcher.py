import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
from pydantic import BaseModel, validator
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Schema ───────────────────────────────────────────────────────────────────
class NewsArticle(BaseModel):
    source: str
    title: str
    description: Optional[str] = None
    url: str
    published_at: datetime
    content: Optional[str] = None

    @validator("published_at", pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, str):
            # newsdata.io format: "2024-01-15 10:30:00"
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
        return v


# ── Fetcher ──────────────────────────────────────────────────────────────────
class NewsFetcher:
    BASE_URL = "https://newsdata.io/api/1/news"

    def __init__(self, keywords: list[str], lookback_days: int = 7):
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("NEWS_API_KEY not set in .env file.")
        self.keywords = keywords
        self.lookback_days = lookback_days

    def _build_query(self) -> str:
        return " OR ".join(f'"{kw}"' for kw in self.keywords[:3])
        # newsdata.io free tier limits query complexity

    def fetch(self) -> list[NewsArticle]:
        params = {
            "apikey": self.api_key,
            "q": self._build_query(),
            "language": "en",
            "category": "business,world",
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                logger.error(f"newsdata.io error: {data.get('message')}")
                return []

            articles = []
            for item in data.get("results", []):
                try:
                    article = NewsArticle(
                        source=item.get("source_id", "unknown"),
                        title=item.get("title", ""),
                        description=item.get("description"),
                        url=item.get("link", ""),
                        published_at=item.get("pubDate", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                        content=item.get("content"),
                    )
                    articles.append(article)
                except Exception as e:
                    logger.warning(f"Skipping malformed article: {e}")

            logger.info(f"Fetched {len(articles)} articles from newsdata.io")
            return articles

        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except requests.Timeout:
            logger.error("Request timed out.")
            raise