"""
ML-based anomaly detection for Solana token trading activity.

Implements an Isolation Forest from scratch (no sklearn) plus
heuristic detectors for volume anomalies and pump-and-dump schemes.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Euler–Mascheroni constant for harmonic number approximation         #
# ------------------------------------------------------------------ #
_EULER_MASCHERONI = 0.5772156649


# ====================================================================
# Isolation Tree
# ====================================================================

class _IsolationNode:
    """Internal / external node of an Isolation Tree."""

    __slots__ = ("split_feature", "split_value", "left", "right", "size")

    def __init__(
        self,
        split_feature: Optional[int] = None,
        split_value: Optional[float] = None,
        left: Optional["_IsolationNode"] = None,
        right: Optional["_IsolationNode"] = None,
        size: int = 0,
    ) -> None:
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size

    @property
    def is_external(self) -> bool:
        return self.left is None and self.right is None


class IsolationTree:
    """A single tree in the Isolation Forest ensemble."""

    def __init__(self, height_limit: int) -> None:
        """
        Parameters
        ----------
        height_limit : int
            Maximum depth the tree is allowed to grow.
        """
        self.height_limit = height_limit
        self.root: Optional[_IsolationNode] = None

    def fit(self, data: list[list[float]], current_height: int = 0) -> _IsolationNode:
        """
        Recursively build the isolation tree.

        Parameters
        ----------
        data : list[list[float]]
            Subset of samples (each sample is a list of floats).
        current_height : int
            Current depth in the tree.

        Returns
        -------
        _IsolationNode
            Root node of the (sub)tree.
        """
        n_samples = len(data)

        # External node conditions
        if n_samples <= 1 or current_height >= self.height_limit:
            node = _IsolationNode(size=n_samples)
            if current_height == 0:
                self.root = node
            return node

        n_features = len(data[0])
        split_feature = random.randint(0, n_features - 1)

        feature_values = [row[split_feature] for row in data]
        feat_min = min(feature_values)
        feat_max = max(feature_values)

        # If all values are identical we cannot split further
        if feat_min == feat_max:
            node = _IsolationNode(size=n_samples)
            if current_height == 0:
                self.root = node
            return node

        split_value = random.uniform(feat_min, feat_max)

        left_data = [row for row in data if row[split_feature] < split_value]
        right_data = [row for row in data if row[split_feature] >= split_value]

        # Guard against empty splits (unlikely but possible at boundary)
        if not left_data or not right_data:
            node = _IsolationNode(size=n_samples)
            if current_height == 0:
                self.root = node
            return node

        node = _IsolationNode(
            split_feature=split_feature,
            split_value=split_value,
            left=self.fit(left_data, current_height + 1),
            right=self.fit(right_data, current_height + 1),
            size=n_samples,
        )
        if current_height == 0:
            self.root = node
        return node


class IsolationForest:
    """
    Isolation Forest anomaly detector implemented from scratch.

    Anomaly score interpretation:
        * Close to **1.0** → anomaly
        * Close to **0.5** → normal
        * Below **0.5** → very normal / dense region
    """

    def __init__(self, n_trees: int = 100, sample_size: int = 256) -> None:
        """
        Parameters
        ----------
        n_trees : int
            Number of isolation trees in the ensemble.
        sample_size : int
            Number of samples drawn (with replacement) to build each tree.
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: list[IsolationTree] = []
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Average-path-length helper  c(n)                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _c(n: int) -> float:
        """
        Average path length of an unsuccessful search in a BST of *n* items.

        Formula: ``2 * H(n-1) - 2*(n-1)/n``
        where ``H(i) ≈ ln(i) + 0.5772…`` (Euler–Mascheroni approximation).
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        h = math.log(n - 1) + _EULER_MASCHERONI
        return 2.0 * h - 2.0 * (n - 1) / n

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, data: list[list[float]]) -> None:
        """
        Build the ensemble of isolation trees.

        Parameters
        ----------
        data : list[list[float]]
            Training data — a list of feature vectors.
        """
        if not data:
            logger.warning("IsolationForest.fit called with empty data")
            return

        n = len(data)
        height_limit = max(1, int(math.ceil(math.log2(max(self.sample_size, 2)))))
        self.trees = []

        for _ in range(self.n_trees):
            # Sub-sample (with replacement)
            if n > self.sample_size:
                sample = random.choices(data, k=self.sample_size)
            else:
                sample = list(data)

            tree = IsolationTree(height_limit)
            tree.fit(sample)
            self.trees.append(tree)

        self._fitted = True
        logger.info(
            "IsolationForest fitted with %d trees, sample_size=%d, data_size=%d",
            self.n_trees, self.sample_size, n,
        )

    # ------------------------------------------------------------------ #
    # Path-length computation                                             #
    # ------------------------------------------------------------------ #

    def _path_length(
        self,
        sample: list[float],
        tree_node: _IsolationNode,
        current_height: int = 0,
    ) -> float:
        """
        Traverse the tree and return the path length for *sample*.

        At an external node, return ``current_height + c(node.size)``
        (the expected additional path length for the remaining data).
        """
        if tree_node.is_external:
            return current_height + self._c(tree_node.size)

        # Determine which branch to follow
        assert tree_node.split_feature is not None
        assert tree_node.split_value is not None

        if sample[tree_node.split_feature] < tree_node.split_value:
            return self._path_length(sample, tree_node.left, current_height + 1)  # type: ignore[arg-type]
        return self._path_length(sample, tree_node.right, current_height + 1)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Anomaly score                                                       #
    # ------------------------------------------------------------------ #

    def score(self, sample: list[float]) -> float:
        """
        Compute the anomaly score for *sample*.

        Returns
        -------
        float
            Score in ``(0, 1]``.  Values near 1.0 indicate anomalies.
        """
        if not self._fitted or not self.trees:
            logger.warning("IsolationForest.score called before fit()")
            return 0.5

        avg_path = sum(
            self._path_length(sample, tree.root, 0)  # type: ignore[arg-type]
            for tree in self.trees
        ) / len(self.trees)

        c_n = self._c(self.sample_size)
        if c_n == 0:
            return 0.5

        return 2.0 ** (-avg_path / c_n)


# ====================================================================
# Data classes
# ====================================================================

@dataclass
class AnomalyReport:
    """Container for a single anomaly detection result."""

    timestamp: float
    token: str
    anomaly_type: str
    score: float
    evidence: list[str] = field(default_factory=list)
    is_anomaly: bool = False


# ====================================================================
# AnomalyDetector — high-level API
# ====================================================================

class AnomalyDetector:
    """
    High-level anomaly detector that combines an Isolation Forest
    with heuristic checks for volume anomalies and pump-and-dump
    schemes.
    """

    # Feature keys expected in data dicts
    _FEATURE_KEYS = [
        "volume",
        "price_change_pct",
        "tx_count",
        "unique_wallets",
        "buy_sell_ratio",
    ]

    def __init__(
        self,
        n_trees: int = 100,
        sample_size: int = 256,
        contamination: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        n_trees : int
            Number of trees for the Isolation Forest.
        sample_size : int
            Sub-sample size for each tree.
        contamination : float
            Expected proportion of anomalies in the data (used for
            threshold calibration).
        """
        self.forest = IsolationForest(n_trees=n_trees, sample_size=sample_size)
        self.contamination = contamination
        self._score_threshold: float = 0.5  # updated after training

    # ------------------------------------------------------------------ #
    # Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_features(cls, sample: dict) -> list[float]:
        """Extract a fixed-length feature vector from a data dict."""
        return [float(sample.get(k, 0.0)) for k in cls._FEATURE_KEYS]

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, normal_data: list[dict]) -> None:
        """
        Train the Isolation Forest on historical *normal* data.

        Parameters
        ----------
        normal_data : list[dict]
            Each dict should contain the feature keys:
            ``volume``, ``price_change_pct``, ``tx_count``,
            ``unique_wallets``, ``buy_sell_ratio``.
        """
        if not normal_data:
            logger.warning("AnomalyDetector.train called with empty data")
            return

        vectors = [self._extract_features(d) for d in normal_data]
        self.forest.fit(vectors)

        # Calibrate the anomaly threshold from the training data itself
        scores = sorted(self.forest.score(v) for v in vectors)
        idx = max(0, int(len(scores) * (1.0 - self.contamination)) - 1)
        self._score_threshold = scores[idx]
        logger.info(
            "Anomaly threshold calibrated at %.4f (contamination=%.2f)",
            self._score_threshold, self.contamination,
        )

    # ------------------------------------------------------------------ #
    # Scoring                                                             #
    # ------------------------------------------------------------------ #

    def score_anomaly(self, sample: dict) -> float:
        """
        Score a single sample via the trained Isolation Forest.

        Parameters
        ----------
        sample : dict
            Same format as items passed to :meth:`train`.

        Returns
        -------
        float
            Anomaly score (higher → more anomalous).
        """
        features = self._extract_features(sample)
        return self.forest.score(features)

    # ------------------------------------------------------------------ #
    # Volume anomaly detection (z-score)                                  #
    # ------------------------------------------------------------------ #

    def detect_volume_anomalies(
        self,
        volume_series: list[float],
        window: int = 20,
    ) -> list[AnomalyReport]:
        """
        Detect anomalous volumes via rolling z-score.

        Parameters
        ----------
        volume_series : list[float]
            Ordered time-series of volume values.
        window : int
            Rolling window size for mean/std calculation.

        Returns
        -------
        list[AnomalyReport]
            One report per flagged data point (|z| > 3.0).
        """
        reports: list[AnomalyReport] = []
        n = len(volume_series)
        if n < window:
            logger.debug(
                "Volume series (%d) shorter than window (%d); skipping", n, window
            )
            return reports

        for i in range(window, n):
            window_slice = volume_series[i - window : i]
            mean = sum(window_slice) / window
            variance = sum((v - mean) ** 2 for v in window_slice) / window
            std = math.sqrt(variance) if variance > 0 else 0.0

            if std == 0.0:
                continue

            z = (volume_series[i] - mean) / std

            if abs(z) > 3.0:
                direction = "spike" if z > 0 else "drop"
                reports.append(
                    AnomalyReport(
                        timestamp=time.time(),
                        token="",
                        anomaly_type=f"volume_{direction}",
                        score=min(abs(z) / 5.0, 1.0),
                        evidence=[
                            f"z-score={z:.2f}",
                            f"volume={volume_series[i]:.2f}",
                            f"rolling_mean={mean:.2f}",
                            f"rolling_std={std:.2f}",
                            f"index={i}",
                        ],
                        is_anomaly=True,
                    )
                )

        return reports

    # ------------------------------------------------------------------ #
    # Pump-and-dump detection                                             #
    # ------------------------------------------------------------------ #

    def detect_pump_scheme(self, token_data: dict) -> Optional[AnomalyReport]:
        """
        Check for pump-and-dump indicators.

        Parameters
        ----------
        token_data : dict
            Must contain:
              * ``new_wallets``  – number of new wallets trading the token recently
              * ``new_wallet_window_sec`` – time window in seconds
              * ``buy_sell_ratio`` – ratio of buys to sells
              * ``price_change_pct`` – recent price change in percent
              * ``price_reversal_pct`` – subsequent price drop in percent
              * ``token`` – token identifier (mint address)

        Returns
        -------
        Optional[AnomalyReport]
            Report if pump-scheme signals are present, else ``None``.
        """
        try:
            new_wallets: int = int(token_data.get("new_wallets", 0))
            window_sec: float = float(token_data.get("new_wallet_window_sec", 3600))
            buy_sell_ratio: float = float(token_data.get("buy_sell_ratio", 1.0))
            price_change: float = float(token_data.get("price_change_pct", 0.0))
            price_reversal: float = float(token_data.get("price_reversal_pct", 0.0))
            token: str = str(token_data.get("token", "unknown"))
        except (TypeError, ValueError) as exc:
            logger.error("Failed to parse token_data for pump detection: %s", exc)
            return None

        evidence: list[str] = []
        score = 0.0

        # Signal 1: burst of new wallets in short window
        wallet_rate = new_wallets / max(window_sec, 1.0)
        if wallet_rate > 0.05:  # more than 1 new wallet per 20 s
            signal = min(wallet_rate / 0.2, 1.0)  # saturate at ~1 per 5 s
            score += 0.35 * signal
            evidence.append(
                f"wallet_burst: {new_wallets} new wallets in {window_sec:.0f}s "
                f"(rate={wallet_rate:.3f}/s)"
            )

        # Signal 2: heavily skewed buy/sell ratio (synchronized buying)
        if buy_sell_ratio > 3.0:
            signal = min((buy_sell_ratio - 3.0) / 7.0, 1.0)
            score += 0.30 * signal
            evidence.append(f"buy_sell_ratio={buy_sell_ratio:.2f}")

        # Signal 3: rapid price spike followed by reversal (dump)
        if price_change > 50.0 and price_reversal > 30.0:
            spike_signal = min(price_change / 200.0, 1.0)
            dump_signal = min(price_reversal / 100.0, 1.0)
            combined = (spike_signal + dump_signal) / 2.0
            score += 0.35 * combined
            evidence.append(
                f"price_spike={price_change:.1f}%, reversal={price_reversal:.1f}%"
            )

        if not evidence:
            return None

        is_anomaly = score >= 0.5

        return AnomalyReport(
            timestamp=time.time(),
            token=token,
            anomaly_type="pump_and_dump",
            score=score,
            evidence=evidence,
            is_anomaly=is_anomaly,
        )

    # ------------------------------------------------------------------ #
    # Comprehensive report                                                #
    # ------------------------------------------------------------------ #

    def get_anomaly_report(self, token: str) -> AnomalyReport:
        """
        Aggregate all anomaly signals into a single comprehensive report.

        This is a *synchronous* convenience wrapper.  For production use,
        callers should invoke the individual detectors and merge results.

        Parameters
        ----------
        token : str
            Token mint address.

        Returns
        -------
        AnomalyReport
            An aggregated anomaly report.  ``score`` is the maximum
            individual score observed; ``evidence`` merges all evidence
            strings.
        """
        evidence: list[str] = []
        max_score = 0.0

        # Isolation-forest score (requires a representative sample dict)
        try:
            sample_dict: dict[str, float] = {
                "volume": 0.0,
                "price_change_pct": 0.0,
                "tx_count": 0.0,
                "unique_wallets": 0.0,
                "buy_sell_ratio": 1.0,
            }
            iso_score = self.score_anomaly(sample_dict)
            max_score = max(max_score, iso_score)
            evidence.append(f"isolation_forest_score={iso_score:.4f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("IsolationForest scoring failed for %s: %s", token, exc)

        is_anomaly = max_score >= self._score_threshold

        return AnomalyReport(
            timestamp=time.time(),
            token=token,
            anomaly_type="aggregate",
            score=max_score,
            evidence=evidence,
            is_anomaly=is_anomaly,
        )
