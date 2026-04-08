"""Metrics collection for trading bot infrastructure monitoring.

Collects trading metrics (PnL, win rate, Sharpe ratio, drawdown) and system
metrics (API latency, error rates, model inference time). Stores time-series
data in memory with configurable retention and exports in Prometheus format.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Supported metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricsSummary:
    """Summary of trading metrics over a given period."""
    period: str
    total_trades: int
    win_rate: float
    pnl: float
    sharpe: float
    max_drawdown: float
    avg_latency: float
    error_rate: float


class MetricsCollector:
    """Collects, stores, and exports trading and system metrics.

    Supports counter, gauge, and histogram metric types with configurable
    retention periods and Prometheus-compatible export.

    Args:
        retention_seconds: How long to keep metric data points in memory.
            Defaults to 86400 (24 hours).
    """

    def __init__(self, retention_seconds: int = 86400) -> None:
        self._retention_seconds = retention_seconds
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._histogram_buckets: Dict[str, List[float]] = {}
        self._default_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        logger.info("MetricsCollector initialised with %ds retention", retention_seconds)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE,
    ) -> Metric:
        """Record a new metric data point.

        Args:
            name: Metric name (e.g. ``trading_pnl``).
            value: Numeric value of the metric.
            labels: Optional key-value labels for the metric.
            metric_type: One of COUNTER, GAUGE, HISTOGRAM.

        Returns:
            The recorded ``Metric`` instance.
        """
        labels = labels or {}
        now = time.time()
        metric = Metric(name=name, value=value, timestamp=now, labels=labels, metric_type=metric_type)

        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
            metric = Metric(name=name, value=self._counters[name], timestamp=now, labels=labels, metric_type=metric_type)
        elif metric_type == MetricType.HISTOGRAM:
            if name not in self._histogram_buckets:
                self._histogram_buckets[name] = list(self._default_buckets)

        self._metrics[name].append(metric)
        self._prune(name)
        logger.debug("Recorded metric %s = %s", name, value)
        return metric

    def get_metric_history(
        self, name: str, period: Optional[float] = None
    ) -> List[Metric]:
        """Return metric history for *name* within the given *period* (seconds).

        Args:
            name: Metric name.
            period: Look-back window in seconds. ``None`` returns all stored data.

        Returns:
            List of ``Metric`` data points sorted by timestamp ascending.
        """
        history = self._metrics.get(name, [])
        if period is not None:
            cutoff = time.time() - period
            history = [m for m in history if m.timestamp >= cutoff]
        return sorted(history, key=lambda m: m.timestamp)

    def calculate_summary(self, period: float = 3600.0) -> MetricsSummary:
        """Calculate a trading metrics summary for the given period.

        Args:
            period: Look-back window in seconds. Defaults to 1 hour.

        Returns:
            A ``MetricsSummary`` with aggregated statistics.
        """
        cutoff = time.time() - period

        # --- trades ---
        trades = [m for m in self._metrics.get("trade_pnl", []) if m.timestamp >= cutoff]
        total_trades = len(trades)
        pnl_values = np.array([t.value for t in trades]) if trades else np.array([0.0])
        win_rate = float(np.mean(pnl_values > 0)) if total_trades > 0 else 0.0
        pnl = float(np.sum(pnl_values))

        # --- sharpe ---
        sharpe = self._calculate_sharpe(pnl_values) if total_trades > 1 else 0.0

        # --- drawdown ---
        max_drawdown = self._calculate_max_drawdown(pnl_values) if total_trades > 0 else 0.0

        # --- latency ---
        latencies = [m.value for m in self._metrics.get("api_latency", []) if m.timestamp >= cutoff]
        avg_latency = float(np.mean(latencies)) if latencies else 0.0

        # --- error rate ---
        errors = [m for m in self._metrics.get("error_count", []) if m.timestamp >= cutoff]
        requests = [m for m in self._metrics.get("request_count", []) if m.timestamp >= cutoff]
        total_errors = sum(m.value for m in errors)
        total_requests = sum(m.value for m in requests) if requests else 1.0
        error_rate = total_errors / max(total_requests, 1.0)

        period_label = f"{period}s"
        summary = MetricsSummary(
            period=period_label,
            total_trades=total_trades,
            win_rate=win_rate,
            pnl=pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_latency=avg_latency,
            error_rate=error_rate,
        )
        logger.info("Summary for %s: trades=%d pnl=%.4f sharpe=%.4f", period_label, total_trades, pnl, sharpe)
        return summary

    def export_prometheus_format(self) -> str:
        """Export all current metrics in Prometheus text exposition format.

        Returns:
            A multi-line string suitable for a ``/metrics`` HTTP endpoint.
        """
        lines: List[str] = []
        seen_help: set[str] = set()

        for name, datapoints in self._metrics.items():
            if not datapoints:
                continue
            latest = datapoints[-1]
            safe_name = name.replace(".", "_").replace("-", "_")

            if safe_name not in seen_help:
                lines.append(f"# HELP {safe_name} {safe_name}")
                lines.append(f"# TYPE {safe_name} {latest.metric_type.value}")
                seen_help.add(safe_name)

            if latest.metric_type == MetricType.HISTOGRAM:
                self._export_histogram(lines, safe_name, name)
            else:
                label_str = self._format_labels(latest.labels)
                lines.append(f"{safe_name}{label_str} {latest.value}")

        return "\n".join(lines) + "\n"

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Return a snapshot of key metrics suitable for a dashboard UI.

        Returns:
            Dictionary with ``summary``, ``recent_trades``, ``system``, and
            ``metric_names`` keys.
        """
        summary = self.calculate_summary(period=3600.0)
        recent_trades = self.get_metric_history("trade_pnl", period=3600.0)

        system_metrics: Dict[str, Any] = {}
        for sys_name in ("api_latency", "error_count", "model_inference_time"):
            history = self.get_metric_history(sys_name, period=3600.0)
            if history:
                values = [m.value for m in history]
                system_metrics[sys_name] = {
                    "current": values[-1],
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "min": float(np.min(values)),
                    "count": len(values),
                }

        return {
            "summary": summary,
            "recent_trades": [{"value": m.value, "timestamp": m.timestamp} for m in recent_trades[-50:]],
            "system": system_metrics,
            "metric_names": list(self._metrics.keys()),
        }

    def reset_metrics(self) -> None:
        """Clear all stored metric data."""
        self._metrics.clear()
        self._counters.clear()
        self._histogram_buckets.clear()
        logger.info("All metrics have been reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune(self, name: str) -> None:
        """Remove data points older than the retention window."""
        cutoff = time.time() - self._retention_seconds
        self._metrics[name] = [m for m in self._metrics[name] if m.timestamp >= cutoff]

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio (assuming ~252 trading days)."""
        excess = returns - risk_free
        std = float(np.std(excess, ddof=1))
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @staticmethod
    def _calculate_max_drawdown(pnl_series: np.ndarray) -> float:
        """Maximum drawdown from cumulative PnL."""
        cumulative = np.cumsum(pnl_series)
        if len(cumulative) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    @staticmethod
    def _format_labels(labels: Dict[str, str]) -> str:
        if not labels:
            return ""
        pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return "{" + pairs + "}"

    def _export_histogram(self, lines: List[str], safe_name: str, raw_name: str) -> None:
        """Append Prometheus histogram lines for *raw_name*."""
        values = [m.value for m in self._metrics.get(raw_name, [])]
        if not values:
            return
        arr = np.array(values)
        buckets = self._histogram_buckets.get(raw_name, self._default_buckets)
        for b in buckets:
            count = int(np.sum(arr <= b))
            lines.append(f'{safe_name}_bucket{{le="{b}"}} {count}')
        lines.append(f'{safe_name}_bucket{{le="+Inf"}} {len(values)}')
        lines.append(f"{safe_name}_sum {float(np.sum(arr))}")
        lines.append(f"{safe_name}_count {len(values)}")
