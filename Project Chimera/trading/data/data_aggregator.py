"""Data aggregation layer combining historical and real-time sources.

Provides multi-source price aggregation with outlier rejection, VWAP
calculation, time-series resampling/alignment, and a unified market
view for downstream strategy consumers.

Combines and normalizes data from multiple on-chain and off-chain feeds.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .historical_data import HistoricalDataPipeline, PriceCandle
from .realtime_streams import PriceUpdate, RealtimeDataStream

logger = logging.getLogger(__name__)


@dataclass
class AggregatedPrice:
    """Price consensus derived from multiple independent sources."""

    timestamp: float
    token: str
    price: float
    vwap: float
    spread: float
    sources_count: int
    confidence: float


@dataclass
class MarketView:
    """Holistic snapshot of a token's market state."""

    timestamp: float
    token: str
    price: float
    volume_24h: float
    liquidity: float
    price_change_pct: float
    sentiment_score: float


@dataclass
class _Trade:
    """Internal representation of a trade used for VWAP calculation."""

    price: float
    volume: float


class DataAggregator:
    """Combine and normalise data from historical and real-time sources.

    Usage::

        aggregator = DataAggregator(pipeline, stream)
        agg = aggregator.aggregate_prices(price_updates)
        vwap = aggregator.calculate_vwap(trades)
    """

    def __init__(
        self,
        historical_pipeline: HistoricalDataPipeline | None = None,
        realtime_stream: RealtimeDataStream | None = None,
    ) -> None:
        self._pipeline = historical_pipeline
        self._stream = realtime_stream

        # In-memory caches keyed by token id
        self._latest_prices: dict[str, list[PriceUpdate]] = {}
        self._candle_cache: dict[str, list[PriceCandle]] = {}

    # ------------------------------------------------------------------
    # Price aggregation
    # ------------------------------------------------------------------

    def aggregate_prices(self, prices: list[PriceUpdate]) -> AggregatedPrice | None:
        """Aggregate prices from multiple sources using median filtering.

        Outlier prices (beyond 2 × MAD from the median) are excluded.
        Confidence is derived from the agreement between remaining
        sources.

        Args:
            prices: Price updates to aggregate; should share the same token.

        Returns:
            An :class:`AggregatedPrice`, or ``None`` if the list is empty.
        """
        if not prices:
            return None

        token = prices[0].token
        values = np.array([p.price for p in prices], dtype=np.float64)
        volumes = np.array([p.volume for p in prices], dtype=np.float64)

        clean_values, clean_volumes = self._remove_outliers(values, volumes)
        if clean_values.size == 0:
            logger.warning("All prices rejected as outliers for %s", token)
            clean_values = values
            clean_volumes = volumes

        median_price = float(np.median(clean_values))
        spread = float(np.max(clean_values) - np.min(clean_values)) if clean_values.size > 1 else 0.0

        total_vol = float(np.sum(clean_volumes))
        vwap = float(np.average(clean_values, weights=clean_volumes)) if total_vol > 0 else median_price

        # Confidence: 1.0 when all sources agree perfectly, decreasing
        # with the coefficient of variation among clean values.
        if clean_values.size > 1 and median_price > 0:
            cv = float(np.std(clean_values) / median_price)
            confidence = max(0.0, min(1.0, 1.0 - cv * 10))
        else:
            confidence = 1.0 if clean_values.size == 1 else 0.0

        agg = AggregatedPrice(
            timestamp=time.time(),
            token=token,
            price=median_price,
            vwap=vwap,
            spread=spread,
            sources_count=int(clean_values.size),
            confidence=round(confidence, 4),
        )

        # Cache latest
        self._latest_prices.setdefault(token, []).extend(prices)
        # Keep only the most recent 500 updates per token
        self._latest_prices[token] = self._latest_prices[token][-500:]

        logger.debug(
            "Aggregated %s: price=%.6f vwap=%.6f conf=%.2f (%d sources)",
            token, agg.price, agg.vwap, agg.confidence, agg.sources_count,
        )
        return agg

    @staticmethod
    def _remove_outliers(
        values: np.ndarray, volumes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers using Median Absolute Deviation.

        Points beyond ``2 × MAD`` from the median are rejected.
        """
        if values.size <= 2:
            return values, volumes

        median = float(np.median(values))
        abs_dev = np.abs(values - median)
        mad = float(np.median(abs_dev))

        if mad == 0:
            return values, volumes

        threshold = 2.0 * mad
        mask = abs_dev <= threshold
        return values[mask], volumes[mask]

    # ------------------------------------------------------------------
    # VWAP
    # ------------------------------------------------------------------

    def calculate_vwap(self, trades: list[dict[str, Any]]) -> float:
        """Calculate Volume-Weighted Average Price from trade dicts.

        Each dict must contain ``"price"`` and ``"volume"`` keys.

        Args:
            trades: List of trade dictionaries.

        Returns:
            The VWAP as a float, or ``0.0`` if no valid trades.
        """
        if not trades:
            return 0.0

        prices = np.array(
            [float(t.get("price", 0)) for t in trades], dtype=np.float64
        )
        volumes = np.array(
            [float(t.get("volume", 0)) for t in trades], dtype=np.float64
        )

        total_volume = float(np.sum(volumes))
        if total_volume == 0:
            logger.warning("Total volume is zero; returning simple mean")
            return float(np.mean(prices))

        vwap = float(np.dot(prices, volumes) / total_volume)
        logger.debug("VWAP calculated: %.6f over %d trades", vwap, len(trades))
        return vwap

    # ------------------------------------------------------------------
    # Time-series helpers
    # ------------------------------------------------------------------

    def resample_timeseries(
        self,
        data: list[dict[str, Any]],
        interval: float,
    ) -> list[dict[str, Any]]:
        """Resample time-series data into uniform buckets.

        Each element in *data* must have a ``"timestamp"`` and a
        ``"value"`` key.  Values within the same bucket are averaged.

        Args:
            data: Input time-series (need not be sorted).
            interval: Bucket width in seconds.

        Returns:
            Sorted list of resampled data points.
        """
        if not data or interval <= 0:
            return []

        timestamps = np.array(
            [d["timestamp"] for d in data], dtype=np.float64
        )
        values = np.array(
            [float(d["value"]) for d in data], dtype=np.float64
        )

        t_min = float(np.min(timestamps))
        t_max = float(np.max(timestamps))

        buckets: dict[int, list[float]] = {}
        for ts, val in zip(timestamps, values):
            idx = int((ts - t_min) // interval)
            buckets.setdefault(idx, []).append(val)

        result: list[dict[str, Any]] = []
        for idx in sorted(buckets):
            bucket_vals = np.array(buckets[idx], dtype=np.float64)
            result.append(
                {
                    "timestamp": t_min + idx * interval,
                    "value": float(np.mean(bucket_vals)),
                    "count": len(buckets[idx]),
                }
            )

        logger.debug(
            "Resampled %d points into %d buckets (%.0fs interval)",
            len(data), len(result), interval,
        )
        return result

    def align_timeseries(
        self,
        series_list: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Align multiple time series onto a common timestamp grid.

        Each inner list must contain dicts with ``"timestamp"`` and
        ``"value"`` keys.  The output uses the union of all timestamps
        and forward-fills missing values.

        Args:
            series_list: Two or more time-series to align.

        Returns:
            Merged list sorted by timestamp with a ``"values"`` array.
        """
        if not series_list:
            return []

        # Gather all unique timestamps
        all_ts: set[float] = set()
        for series in series_list:
            for pt in series:
                all_ts.add(pt["timestamp"])

        sorted_ts = sorted(all_ts)

        # Build lookup for each series: ts -> value
        lookups: list[dict[float, float]] = []
        for series in series_list:
            mapping: dict[float, float] = {}
            for pt in series:
                mapping[pt["timestamp"]] = float(pt["value"])
            lookups.append(mapping)

        result: list[dict[str, Any]] = []
        prev_values = [0.0] * len(series_list)

        for ts in sorted_ts:
            current: list[float] = []
            for i, lookup in enumerate(lookups):
                val = lookup.get(ts)
                if val is not None:
                    prev_values[i] = val
                    current.append(val)
                else:
                    current.append(prev_values[i])
            result.append({"timestamp": ts, "values": current})

        logger.debug(
            "Aligned %d series onto %d timestamps", len(series_list), len(sorted_ts),
        )
        return result

    # ------------------------------------------------------------------
    # Market view
    # ------------------------------------------------------------------

    async def get_market_view(self, token: str) -> MarketView | None:
        """Build a comprehensive market view for a token.

        Combines cached real-time prices, historical candle data, and
        volume information into a single :class:`MarketView` snapshot.

        Args:
            token: CoinGecko token identifier (e.g. ``"solana"``).

        Returns:
            A :class:`MarketView`, or ``None`` if insufficient data.
        """
        logger.info("Building market view for %s", token)

        # Current price from cached real-time data
        cached = self._latest_prices.get(token, [])
        if cached:
            recent = cached[-10:]
            prices_arr = np.array([p.price for p in recent], dtype=np.float64)
            current_price = float(np.median(prices_arr))
            volume_24h = float(max(p.volume for p in recent))
        elif self._pipeline:
            try:
                candles = await self._pipeline.fetch_price_history(token, 1)
                if candles:
                    current_price = candles[-1].close
                    volume_24h = sum(c.volume for c in candles)
                else:
                    logger.warning("No data available for %s", token)
                    return None
            except Exception:
                logger.exception("Pipeline fetch failed for %s", token)
                return None
        else:
            logger.warning("No data sources available for %s", token)
            return None

        # Price change percentage from candle history
        price_change_pct = 0.0
        candles = self._candle_cache.get(token)
        if not candles and self._pipeline:
            try:
                candles = await self._pipeline.fetch_price_history(token, 1)
                if candles:
                    self._candle_cache[token] = candles
            except Exception:
                logger.warning("Could not fetch candle history for %s", token)

        if candles and len(candles) >= 2:
            old_price = candles[0].close
            if old_price > 0:
                price_change_pct = ((current_price - old_price) / old_price) * 100.0

        # Liquidity placeholder — a full implementation would query pools
        liquidity = 0.0

        # Sentiment heuristic: normalise price change to [-1, 1]
        sentiment_score = float(np.clip(price_change_pct / 10.0, -1.0, 1.0))

        return MarketView(
            timestamp=time.time(),
            token=token,
            price=current_price,
            volume_24h=volume_24h,
            liquidity=liquidity,
            price_change_pct=round(price_change_pct, 4),
            sentiment_score=round(sentiment_score, 4),
        )

    # ------------------------------------------------------------------
    # Outlier detection (public)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_price_outliers(
        prices: list[float], threshold_factor: float = 2.0
    ) -> list[int]:
        """Identify outlier indices in a list of prices.

        Uses Median Absolute Deviation (MAD).  Indices whose value
        deviates more than ``threshold_factor × MAD`` from the median
        are returned.

        Args:
            prices: Raw price observations.
            threshold_factor: Sensitivity multiplier (default 2.0).

        Returns:
            List of integer indices flagged as outliers.
        """
        if len(prices) < 3:
            return []

        arr = np.array(prices, dtype=np.float64)
        median = float(np.median(arr))
        abs_dev = np.abs(arr - median)
        mad = float(np.median(abs_dev))

        if mad == 0:
            return []

        outlier_indices: list[int] = []
        for i, dev in enumerate(abs_dev):
            if dev > threshold_factor * mad:
                outlier_indices.append(i)

        if outlier_indices:
            logger.info(
                "Detected %d outlier(s) in %d prices (MAD=%.6f)",
                len(outlier_indices), len(prices), mad,
            )
        return outlier_indices
