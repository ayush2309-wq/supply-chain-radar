import logging
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ingestion.news_fetcher import NewsFetcher
from src.ingestion.price_fetcher import PriceFetcher
from src.ingestion.geo_fetcher import GeoFetcher

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_ingestion() -> dict:
    config  = load_config()
    ing     = config["ingestion"]

    fetchers = {
        "news": lambda: NewsFetcher(
            keywords=ing["news"]["keywords"],
            lookback_days=ing["news"]["lookback_days"],
        ).fetch(),
        "prices": lambda: PriceFetcher(
            tickers=ing["prices"]["tickers"],
            lookback_days=ing["prices"]["lookback_days"],
        ).fetch(),
        "geo": lambda: GeoFetcher(
            regions=ing["geo"]["regions"],
        ).fetch(),
    }

    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn): name for name, fn in fetchers.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                logger.info(f"✓ {name} → {len(results[name])} records")
            except Exception as e:
                logger.error(f"✗ {name} failed: {e}")
                results[name] = []

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    data = run_ingestion()
    print(f"\n── Ingestion Summary ──")
    for k, v in data.items():
        print(f"  {k:10s} → {len(v)} records")