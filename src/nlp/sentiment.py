import logging
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"  # FinBERT — trained on financial news


class SentimentAnalyzer:
    """
    Uses FinBERT to classify news headlines as:
    positive / negative / neutral
    Then maps to a risk score: negative = high risk signal
    """

    def __init__(self):
        logger.info("Loading FinBERT model — first run downloads ~500MB...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.labels    = ["positive", "negative", "neutral"]
        logger.info("FinBERT loaded successfully.")

    def analyze(self, text: str) -> dict:
        """
        Returns sentiment label + confidence + risk_score (0–1).
        Negative sentiment = higher risk score.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs  = F.softmax(outputs.logits, dim=-1).squeeze()
            scores = {label: round(float(probs[i]), 4) for i, label in enumerate(self.labels)}

            predicted = max(scores, key=scores.get)

            # Risk score: negative news = 1.0, neutral = 0.5, positive = 0.0
            risk_map   = {"negative": 1.0, "neutral": 0.5, "positive": 0.0}
            risk_score = round(
                risk_map[predicted] * scores[predicted] +
                0.5 * scores["neutral"] * 0.3,   # partial weight for uncertainty
                4
            )

            return {
                "sentiment":  predicted,
                "confidence": scores[predicted],
                "risk_score": risk_score,
                "scores":     scores,
            }

        except Exception as e:
            logger.warning(f"Sentiment analysis failed for text: {e}")
            return {
                "sentiment":  "neutral",
                "confidence": 0.0,
                "risk_score": 0.5,
                "scores":     {},
            }

    def batch_analyze(self, texts: list[str]) -> list[dict]:
        """Analyze a list of headlines efficiently."""
        results = []
        for i, text in enumerate(texts):
            result = self.analyze(text)
            results.append(result)
            if (i + 1) % 5 == 0:
                logger.info(f"Analyzed {i + 1}/{len(texts)} headlines")
        return results