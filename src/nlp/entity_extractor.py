import logging
import re

logger = logging.getLogger(__name__)

# Key supply chain entities to watch
COUNTRY_ALIASES = {
    "china": "CN", "chinese": "CN",
    "russia": "RU", "russian": "RU",
    "ukraine": "UA", "ukrainian": "UA",
    "taiwan": "TW", "taiwanese": "TW",
    "india": "IN", "indian": "IN",
    "germany": "DE", "german": "DE",
    "saudi": "SA", "saudi arabia": "SA",
    "united states": "US", "usa": "US", "american": "US",
}

KEY_PORTS = [
    "shanghai", "singapore", "rotterdam", "los angeles",
    "hamburg", "busan", "dubai", "hong kong", "suez",
]

KEY_COMMODITIES = [
    "oil", "gas", "copper", "aluminum", "steel",
    "semiconductor", "chip", "lithium", "wheat", "fertilizer",
]

DISRUPTION_TYPES = {
    "strike":    "LABOR",
    "flood":     "WEATHER",
    "hurricane": "WEATHER",
    "earthquake":"WEATHER",
    "sanctions": "GEOPOLITICAL",
    "war":       "GEOPOLITICAL",
    "conflict":  "GEOPOLITICAL",
    "shutdown":  "OPERATIONAL",
    "delay":     "OPERATIONAL",
    "shortage":  "SUPPLY",
    "ban":       "REGULATORY",
}


class EntityExtractor:

    @staticmethod
    def extract(text: str) -> dict:
        text_lower = text.lower()

        # Countries mentioned
        countries = []
        for alias, code in COUNTRY_ALIASES.items():
            if alias in text_lower and code not in countries:
                countries.append(code)

        # Ports mentioned
        ports = [port for port in KEY_PORTS if port in text_lower]

        # Commodities mentioned
        commodities = [c for c in KEY_COMMODITIES if c in text_lower]

        # Disruption type
        disruption_type = None
        for keyword, dtype in DISRUPTION_TYPES.items():
            if keyword in text_lower:
                disruption_type = dtype
                break

        return {
            "countries":       countries,
            "ports":           ports,
            "commodities":     commodities,
            "disruption_type": disruption_type,
        }

    @staticmethod
    def batch_extract(texts: list[str]) -> list[dict]:
        return [EntityExtractor.extract(t) for t in texts]