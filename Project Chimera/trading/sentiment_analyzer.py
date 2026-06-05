# trading/sentiment_analyzer.py
"""
Sentiment Analysis Engine — fetches news and social-media signals, scores
them, and provides an aggregate sentiment indicator per token.

Data sources (free-tier, no API key required for basic use)
-----------------------------------------------------------
* CryptoCompare News API   — crypto news headlines
* CoinGecko Trending       — trending tokens / market sentiment
* Keyword-based scoring    — configurable positive/negative word lists

The sentiment score ranges from -1.0 (extremely bearish) to +1.0
(extremely bullish).  A score of 0.0 is neutral.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import requests

import config
from trading.database import init_db, insert_sentiment, get_aggregate_sentiment

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Keyword dictionaries for simple sentiment scoring                            #
# --------------------------------------------------------------------------- #

_POSITIVE_KEYWORDS = {
    "bullish", "surge", "pump", "moon", "rally", "breakout", "soar", "gain",
    "profit", "uptrend", "buy", "long", "accumulate", "growth", "launch",
    "partnership", "listing", "upgrade", "adoption", "integration", "airdrop",
    "reward", "stake", "yield", "tvl increase", "ath", "all-time high",
    "outperform", "strong", "confidence", "recovery",
}

_NEGATIVE_KEYWORDS = {
    "bearish", "dump", "crash", "rug", "scam", "hack", "exploit", "sell",
    "short", "liquidation", "downtrend", "correction", "loss", "fraud",
    "sec", "lawsuit", "delisting", "vulnerability", "bug", "drain",
    "ponzi", "fake", "manipulation", "fud", "fear", "panic", "decline",
    "plunge", "collapse", "warning", "risk", "suspicious",
}


class SentimentAnalyzer:
    """
    Fetches and scores sentiment data from news and social sources.
    """

    def __init__(self):
        self.conn = init_db()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "ProjectPhoenix/1.0",
            "Accept": "application/json",
        })
        logger.info("SentimentAnalyzer initialised.")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def fetch_and_score_news(self, token_symbol: Optional[str] = None) -> list[dict]:
        """
        Fetch recent crypto news, score sentiment, and persist to DB.

        Parameters
        ----------
        token_symbol : str or None
            If provided, only news mentioning this symbol is scored with
            high relevance.  All news is still stored for general market
            sentiment.

        Returns a list of scored sentiment dicts.
        """
        articles = self._fetch_cryptocompare_news()
        scored: list[dict] = []

        for article in articles:
            title = article.get("title", "")
            body = article.get("body", "")
            text = f"{title} {body}"

            sentiment = self._score_text(text)
            relevance = self._compute_relevance(text, token_symbol)

            # Try to extract a token address context
            token_addr = None
            categories = article.get("categories", "")
            if token_symbol and token_symbol.upper() in text.upper():
                token_addr = token_symbol  # placeholder — real mapping done elsewhere

            record = {
                "token_address": token_addr,
                "source": "news",
                "headline": title[:500],
                "sentiment_score": sentiment,
                "relevance_score": relevance,
                "raw_text": text[:2000],
            }
            insert_sentiment(self.conn, record)
            scored.append(record)

        logger.info(
            "Fetched and scored %d news articles (filter=%s).",
            len(scored),
            token_symbol or "all",
        )
        return scored

    def fetch_trending_sentiment(self) -> list[dict]:
        """
        Fetch trending tokens from CoinGecko and infer market sentiment
        from trending momentum.

        Returns a list of sentiment records for trending tokens.
        """
        scored: list[dict] = []
        try:
            resp = self._session.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            coins = data.get("coins", [])
            for coin_wrapper in coins:
                coin = coin_wrapper.get("item", {})
                name = coin.get("name", "")
                symbol = coin.get("symbol", "")
                score = coin.get("score", 0)

                # Trending = positive momentum signal (mild bullish)
                # Clamp score to [0, 10] to avoid negative sentiment
                clamped_score = max(0, min(score, 10))
                sentiment = max(0.1, min(0.3 + (1.0 - clamped_score / 10.0) * 0.4, 1.0))

                record = {
                    "token_address": symbol,
                    "source": "trending",
                    "headline": f"{name} ({symbol}) is trending on CoinGecko",
                    "sentiment_score": sentiment,
                    "relevance_score": 0.5,
                    "raw_text": f"Trending rank: {score}",
                }
                insert_sentiment(self.conn, record)
                scored.append(record)

        except requests.RequestException as exc:
            logger.warning("Failed to fetch trending data: %s", exc)
        except (ValueError, KeyError) as exc:
            logger.warning("Error parsing trending data: %s", exc)

        logger.info("Fetched %d trending sentiment signals.", len(scored))
        return scored

    def get_token_sentiment(self, token_address: str) -> float:
        """
        Return the aggregate sentiment score for a token.

        Returns a float in [-1.0, 1.0].  0.0 if no data.
        """
        return get_aggregate_sentiment(self.conn, token_address)

    def get_market_sentiment(self) -> float:
        """
        Compute a general market sentiment score by averaging all recent
        sentiment records (not token-specific).

        Returns a float in [-1.0, 1.0].
        """
        row = self.conn.execute(
            """
            SELECT AVG(sentiment_score) AS avg_sentiment
            FROM (
                SELECT sentiment_score FROM sentiment_data
                ORDER BY fetched_at DESC
                LIMIT 100
            )
            """
        ).fetchone()
        return float(row["avg_sentiment"]) if row and row["avg_sentiment"] is not None else 0.0

    def should_trade(self, token_address: str) -> tuple[bool, str]:
        """
        Determine if sentiment supports trading a given token.

        Returns (should_trade: bool, reason: str).
        """
        token_sent = self.get_token_sentiment(token_address)
        market_sent = self.get_market_sentiment()

        # Block trading if token sentiment is strongly negative
        if token_sent < config.SENTIMENT_BEARISH_THRESHOLD:
            return False, f"Token sentiment too bearish ({token_sent:.2f})"

        # Block trading if overall market sentiment is very negative
        if market_sent < config.SENTIMENT_MARKET_FEAR_THRESHOLD:
            return False, f"Market-wide fear detected ({market_sent:.2f})"

        return True, f"Sentiment OK (token={token_sent:.2f}, market={market_sent:.2f})"

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _fetch_cryptocompare_news(self, limit: int = 50) -> list[dict]:
        """Fetch recent news articles from CryptoCompare (free tier)."""
        try:
            resp = self._session.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN", "sortOrder": "latest"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("Data", [])
            return articles[:limit]
        except requests.RequestException as exc:
            logger.warning("CryptoCompare news fetch failed: %s", exc)
            return []
        except (ValueError, KeyError) as exc:
            logger.warning("Error parsing CryptoCompare response: %s", exc)
            return []

    def _score_text(self, text: str) -> float:
        """
        Score a text string for sentiment using keyword matching.

        Returns a float in [-1.0, 1.0].
        """
        if not text:
            return 0.0

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        pos_count = len(words & _POSITIVE_KEYWORDS)
        neg_count = len(words & _NEGATIVE_KEYWORDS)

        # Also check multi-word phrases
        for phrase in _POSITIVE_KEYWORDS:
            if " " in phrase and phrase in text_lower:
                pos_count += 1
        for phrase in _NEGATIVE_KEYWORDS:
            if " " in phrase and phrase in text_lower:
                neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        # Normalise to [-1, 1]
        raw_score = (pos_count - neg_count) / total
        return max(-1.0, min(1.0, raw_score))

    def _compute_relevance(self, text: str, token_symbol: Optional[str]) -> float:
        """
        Compute how relevant a text is to a specific token.

        Returns 1.0 if the token symbol appears in the text,
        0.3 for general Solana mentions, 0.1 otherwise.
        """
        if not text:
            return 0.0

        text_lower = text.lower()

        if token_symbol and token_symbol.lower() in text_lower:
            return 1.0
        if "solana" in text_lower or "sol" in text_lower:
            return 0.3
        return 0.1
