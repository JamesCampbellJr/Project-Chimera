"""Unified sentiment scoring engine.

Combines news, social-media, and on-chain sentiment sources into a
single score with EMA smoothing, momentum tracking, contrarian signal
detection, and trend classification.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .news_analyzer import NewsAnalyzer
from .social_media import SocialMediaAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SentimentSnapshot:
    """Point-in-time sentiment reading for a single token."""
    timestamp: datetime
    token: str
    news_score: float       # -1.0 … +1.0
    social_score: float     # -1.0 … +1.0
    onchain_score: float    # -1.0 … +1.0
    unified_score: float    # -1.0 … +1.0
    momentum: float         # rate of change
    trend: str              # "bullish", "bearish", "neutral", "shifting"
    is_extreme: bool        # True when |unified_score| > 0.8


@dataclass(frozen=True)
class ContrarianSignal:
    """Signal generated when sentiment reaches an extreme reading."""
    token: str
    current_sentiment: float
    signal_type: str          # "overbought" or "oversold"
    confidence: float         # 0.0 … 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# SentimentScorer
# ---------------------------------------------------------------------------

class SentimentScorer:
    """Combine multiple sentiment sources into one unified score.

    Features:
    * Configurable source weights (news, social, on-chain).
    * EMA-smoothed sentiment with configurable period.
    * Momentum (rate of change of the smoothed score).
    * Contrarian signal detection at extreme readings.
    * Trend classification over a rolling history window.
    """

    def __init__(
        self,
        news_weight: float = 0.3,
        social_weight: float = 0.4,
        onchain_weight: float = 0.3,
        ema_period: int = 14,
        history_size: int = 100,
        extreme_threshold: float = 0.8,
        news_analyzer: NewsAnalyzer | None = None,
        social_analyzer: SocialMediaAnalyzer | None = None,
    ) -> None:
        """Initialise the SentimentScorer.

        Args:
            news_weight: Weight for news sentiment component.
            social_weight: Weight for social-media sentiment component.
            onchain_weight: Weight for on-chain sentiment component.
            ema_period: Number of observations for EMA smoothing.
            history_size: Maximum snapshots kept in the rolling history.
            extreme_threshold: Absolute score threshold to flag as extreme.
            news_analyzer: Optional pre-configured :class:`NewsAnalyzer`.
            social_analyzer: Optional pre-configured :class:`SocialMediaAnalyzer`.
        """
        total = news_weight + social_weight + onchain_weight
        if total == 0:
            raise ValueError("At least one weight must be non-zero.")
        self._news_weight = news_weight / total
        self._social_weight = social_weight / total
        self._onchain_weight = onchain_weight / total

        self._ema_period = max(1, ema_period)
        self._ema_multiplier = 2.0 / (self._ema_period + 1)
        self._extreme_threshold = extreme_threshold

        # Per-token state
        self._ema_values: dict[str, float] = {}
        self._history: dict[str, deque[SentimentSnapshot]] = {}
        self._history_size = history_size

        self._news_analyzer = news_analyzer
        self._social_analyzer = social_analyzer

    # -- core calculations --------------------------------------------------

    def calculate_unified_score(
        self,
        news: float,
        social: float,
        onchain: float,
    ) -> float:
        """Return a weighted unified sentiment score in **-1.0 … +1.0**.

        Args:
            news: News sentiment score (-1.0 … +1.0).
            social: Social-media sentiment score (-1.0 … +1.0).
            onchain: On-chain sentiment score (-1.0 … +1.0).

        Returns:
            Weighted composite score clamped to -1.0 … +1.0.
        """
        raw = (
            self._news_weight * _clamp(news)
            + self._social_weight * _clamp(social)
            + self._onchain_weight * _clamp(onchain)
        )
        return round(_clamp(raw), 4)

    def update_ema(self, new_score: float, token: str = "_default") -> float:
        """Update and return the EMA-smoothed score for *token*.

        On the first call the EMA is seeded with *new_score*.

        Args:
            new_score: Latest unified score observation.
            token: Token identifier (allows per-token tracking).

        Returns:
            Updated EMA value.
        """
        new_score = _clamp(new_score)
        prev = self._ema_values.get(token)
        if prev is None:
            ema = new_score
        else:
            ema = (new_score - prev) * self._ema_multiplier + prev
        self._ema_values[token] = round(ema, 6)
        return self._ema_values[token]

    def get_momentum(self, token: str = "_default") -> float:
        """Return the momentum (rate of change) of the smoothed score.

        Momentum is defined as the difference between the two most recent
        EMA-smoothed snapshots.  Positive → sentiment is improving.

        Args:
            token: Token identifier.

        Returns:
            Momentum value, or 0.0 if insufficient history.
        """
        history = self._history.get(token)
        if not history or len(history) < 2:
            return 0.0
        return round(history[-1].unified_score - history[-2].unified_score, 4)

    def detect_extremes(self, token: str = "_default") -> bool:
        """Return *True* if the latest unified score is at an extreme level.

        An extreme reading is one where ``|unified_score| >= extreme_threshold``.

        Args:
            token: Token identifier.
        """
        history = self._history.get(token)
        if not history:
            return False
        return abs(history[-1].unified_score) >= self._extreme_threshold

    def classify_trend(
        self,
        history: list[SentimentSnapshot] | None = None,
        token: str = "_default",
    ) -> str:
        """Classify the sentiment trend from recent history.

        Uses the slope of the last *N* unified scores to determine the
        trend direction.

        Args:
            history: Explicit snapshot list (overrides internal history).
            token: Token identifier used when *history* is *None*.

        Returns:
            One of ``"bullish"``, ``"bearish"``, ``"neutral"``, or
            ``"shifting"``.
        """
        snapshots = history or list(self._history.get(token, []))
        if len(snapshots) < 3:
            return "neutral"

        recent = [s.unified_score for s in snapshots[-10:]]
        slope = _linear_slope(recent)

        # Check for direction consistency
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        positive_diffs = sum(1 for d in diffs if d > 0)
        negative_diffs = sum(1 for d in diffs if d < 0)
        consistency = max(positive_diffs, negative_diffs) / len(diffs)

        if consistency < 0.6:
            return "shifting"

        avg = sum(recent) / len(recent)
        if slope > 0.01 and avg > 0.05:
            return "bullish"
        elif slope < -0.01 and avg < -0.05:
            return "bearish"
        elif abs(slope) <= 0.01:
            return "neutral"
        else:
            return "shifting"

    def get_contrarian_signal(
        self,
        token: str = "_default",
    ) -> ContrarianSignal | None:
        """Generate a contrarian signal if sentiment is at an extreme.

        Extreme bullish sentiment (>0.8) → *overbought* (consider selling).
        Extreme bearish sentiment (<-0.8) → *oversold* (consider buying).

        Args:
            token: Token identifier.

        Returns:
            :class:`ContrarianSignal` if an extreme is detected, else *None*.
        """
        history = self._history.get(token)
        if not history:
            return None

        latest = history[-1]
        if abs(latest.unified_score) < self._extreme_threshold:
            return None

        if latest.unified_score >= self._extreme_threshold:
            signal_type = "overbought"
        else:
            signal_type = "oversold"

        # Confidence rises with how far past the threshold we are, and with
        # the number of consistent recent readings.
        excess = abs(latest.unified_score) - self._extreme_threshold
        # Scale excess: 0 at threshold, 1.0 at ±1.0
        excess_confidence = min(1.0, excess / (1.0 - self._extreme_threshold + 1e-9))

        # Duration factor: how many of the last 5 readings are extreme?
        recent = list(history)[-5:]
        extreme_count = sum(
            1 for s in recent if abs(s.unified_score) >= self._extreme_threshold
        )
        duration_confidence = extreme_count / len(recent)

        confidence = round(0.5 * excess_confidence + 0.5 * duration_confidence, 4)

        signal = ContrarianSignal(
            token=token,
            current_sentiment=latest.unified_score,
            signal_type=signal_type,
            confidence=confidence,
        )
        logger.info("Contrarian signal: %s", signal)
        return signal

    # -- high-level orchestration -------------------------------------------

    async def get_token_sentiment(
        self,
        token: str,
        onchain_score: float = 0.0,
    ) -> SentimentSnapshot:
        """Compute a full sentiment snapshot for *token*.

        Fetches news and social sentiment (if analysers are configured),
        combines them with the supplied on-chain score, updates the EMA
        and history, and returns a :class:`SentimentSnapshot`.

        Args:
            token: Token name or ticker.
            onchain_score: Pre-computed on-chain sentiment (-1.0 … +1.0).

        Returns:
            The resulting :class:`SentimentSnapshot`.
        """
        # -- news sentiment --
        news_score = 0.0
        if self._news_analyzer:
            try:
                news_result = await self._news_analyzer.get_news_sentiment(token)
                news_score = news_result.score
            except Exception as exc:
                logger.error("News sentiment fetch failed for '%s': %s", token, exc)

        # -- social sentiment --
        social_score = 0.0
        if self._social_analyzer:
            try:
                posts = await self._social_analyzer.analyze_twitter(token, limit=100)
                if posts:
                    from .news_analyzer import NewsAnalyzer as _NA
                    _text_analyzer = _NA()
                    scores = [
                        _text_analyzer.analyze_text_sentiment(p.content).score
                        for p in posts
                    ]
                    social_score = sum(scores) / len(scores) if scores else 0.0
            except Exception as exc:
                logger.error("Social sentiment fetch failed for '%s': %s", token, exc)

        unified = self.calculate_unified_score(news_score, social_score, onchain_score)
        smoothed = self.update_ema(unified, token)
        momentum = self.get_momentum(token)
        is_extreme = abs(smoothed) >= self._extreme_threshold

        # Create snapshot and record
        snapshot = SentimentSnapshot(
            timestamp=datetime.now(timezone.utc),
            token=token,
            news_score=round(news_score, 4),
            social_score=round(social_score, 4),
            onchain_score=round(_clamp(onchain_score), 4),
            unified_score=smoothed,
            momentum=momentum,
            trend="neutral",  # placeholder — updated below
            is_extreme=is_extreme,
        )

        # Append to history
        if token not in self._history:
            self._history[token] = deque(maxlen=self._history_size)
        self._history[token].append(snapshot)

        # Classify trend using accumulated history
        trend = self.classify_trend(token=token)
        # Replace trend in snapshot (dataclass is mutable)
        snapshot.trend = trend

        logger.info(
            "Sentiment snapshot for %s: unified=%.4f ema=%.4f momentum=%.4f trend=%s",
            token, unified, smoothed, momentum, trend,
        )
        return snapshot

    # -- state accessors ----------------------------------------------------

    def get_history(self, token: str) -> list[SentimentSnapshot]:
        """Return the full snapshot history for *token*."""
        return list(self._history.get(token, []))

    def get_ema(self, token: str) -> float | None:
        """Return the current EMA value for *token*, or *None*."""
        return self._ema_values.get(token)

    def reset(self, token: str | None = None) -> None:
        """Clear internal state.

        Args:
            token: If provided, clear only that token's state.
                   Otherwise clear everything.
        """
        if token:
            self._ema_values.pop(token, None)
            self._history.pop(token, None)
        else:
            self._ema_values.clear()
            self._history.clear()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _linear_slope(values: list[float]) -> float:
    """Return the least-squares slope of *values* indexed 0 … n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator
