import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Schema ───────────────────────────────────────────────────────────────────
class PriceRecord(BaseModel):
    ticker: str
    label: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    pct_change: Optional[float] = None


# ── Fetcher ──────────────────────────────────────────────────────────────────
class PriceFetcher:
    def __init__(self, tickers: dict[str, str], lookback_days: int = 30):
        self.tickers = tickers
        self.lookback_days = lookback_days

    def fetch(self) -> list[PriceRecord]:
        start = (datetime.utcnow() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        records = []

        for label, ticker in self.tickers.items():
            try:
                df = yf.download(ticker, start=start, progress=False, auto_adjust=True)

                if df.empty:
                    logger.warning(f"No data for {ticker} ({label})")
                    continue

                # Fix for yfinance 1.2.0 — flatten MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df["pct_change"] = df["Close"].pct_change() * 100
                df.dropna(subset=["Close"], inplace=True)

                for date, row in df.iterrows():
                    records.append(
                        PriceRecord(
                            ticker=ticker,
                            label=label,
                            date=date.to_pydatetime(),
                            open=round(float(row["Open"]), 4),
                            high=round(float(row["High"]), 4),
                            low=round(float(row["Low"]), 4),
                            close=round(float(row["Close"]), 4),
                            volume=float(row["Volume"]) if pd.notna(row["Volume"]) else None,
                            pct_change=round(float(row["pct_change"]), 4) if pd.notna(row["pct_change"]) else None,
                        )
                    )

                logger.info(f"Fetched {len(df)} rows for {label} ({ticker})")

            except Exception as e:
                logger.error(f"Failed fetching {ticker}: {e}")

        return records