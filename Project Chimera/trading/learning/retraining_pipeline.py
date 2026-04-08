"""Model retraining pipeline with drift detection, A/B testing, and versioning.

Supports scheduled retraining, Population Stability Index (PSI) and
Kolmogorov-Smirnov drift detection, champion/challenger A/B testing,
random-search hyperparameter tuning, and model version management with
rollback capabilities.
"""

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols – minimal interface that "model" objects must satisfy
# ---------------------------------------------------------------------------

class TrainableModel(Protocol):
    """Protocol describing what the pipeline expects from a model object."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def get_params(self) -> Dict[str, Any]: ...
    def set_params(self, **params: Any) -> None: ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    """Snapshot of a trained model version."""
    name: str
    version: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trained_at: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)
    is_champion: bool = False
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    model_object: Any = None


@dataclass
class DriftReport:
    """Result of a concept-drift detection analysis."""
    model_name: str
    psi_score: float
    ks_statistic: float
    is_drifted: bool
    recommendation: str


@dataclass
class ABTestResult:
    """Outcome of an A/B (champion vs challenger) test."""
    champion_version: str
    challenger_version: str
    champion_metric: float
    challenger_metric: float
    winner: str
    confidence: float


# ---------------------------------------------------------------------------
# Retraining pipeline
# ---------------------------------------------------------------------------

class RetrainingPipeline:
    """End-to-end model retraining, evaluation, and deployment pipeline.

    Args:
        metric_fn: Callable ``(y_true, y_pred) -> float`` used to evaluate
            model performance.  Higher is better by default.
        psi_threshold: PSI value above which drift is declared.
        ks_threshold: KS-statistic value above which drift is declared.
    """

    def __init__(
        self,
        metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
    ) -> None:
        self._versions: Dict[str, List[ModelVersion]] = {}
        self._champion: Dict[str, ModelVersion] = {}
        self._metric_fn = metric_fn or self._default_accuracy
        self._psi_threshold = psi_threshold
        self._ks_threshold = ks_threshold
        self._schedule: Dict[str, Dict[str, Any]] = {}
        logger.info("RetrainingPipeline initialised (PSI=%.2f, KS=%.2f)",
                     psi_threshold, ks_threshold)

    # ------------------------------------------------------------------
    # Model retraining
    # ------------------------------------------------------------------

    async def retrain_model(
        self,
        model: Any,
        new_data: Dict[str, np.ndarray],
    ) -> ModelVersion:
        """Retrain *model* on *new_data* and register a new version.

        Args:
            model: An object satisfying :class:`TrainableModel`.
            new_data: Dictionary with ``X_train``, ``y_train``, and optionally
                ``X_val``, ``y_val`` arrays.

        Returns:
            The newly created ``ModelVersion``.
        """
        model_name = getattr(model, "name", "unnamed")
        logger.info("Starting retraining for model '%s'", model_name)
        start = time.time()

        try:
            X_train = new_data["X_train"]
            y_train = new_data["y_train"]
            model.fit(X_train, y_train)

            metrics: Dict[str, float] = {}
            if "X_val" in new_data and "y_val" in new_data:
                preds = model.predict(new_data["X_val"])
                metrics["validation_score"] = float(self._metric_fn(new_data["y_val"], preds))

            metrics["training_time"] = time.time() - start
            metrics["training_samples"] = len(X_train)

            version = ModelVersion(
                name=model_name,
                metrics=metrics,
                hyperparams=model.get_params() if hasattr(model, "get_params") else {},
                model_object=copy.deepcopy(model),
            )
            self._versions.setdefault(model_name, []).append(version)

            if model_name not in self._champion:
                version.is_champion = True
                self._champion[model_name] = version

            logger.info("Model '%s' retrained (v=%s) in %.2fs – metrics: %s",
                        model_name, version.version, metrics.get("training_time", 0), metrics)
            return version

        except Exception as exc:
            logger.error("Retraining failed for '%s': %s", model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        model_name: str = "unnamed",
    ) -> DriftReport:
        """Detect concept drift between *reference_data* and *current_data*.

        Uses Population Stability Index (PSI) and the two-sample
        Kolmogorov-Smirnov test.

        Args:
            reference_data: 1-D array of reference (training) distribution values.
            current_data: 1-D array of recent production distribution values.
            model_name: Name to include in the report.

        Returns:
            A ``DriftReport`` with PSI score, KS statistic, and recommendation.
        """
        psi = self._calculate_psi(reference_data, current_data)
        ks = self._calculate_ks(reference_data, current_data)
        is_drifted = psi > self._psi_threshold or ks > self._ks_threshold

        if is_drifted:
            recommendation = "Retrain model – significant drift detected"
        elif psi > self._psi_threshold * 0.5 or ks > self._ks_threshold * 0.5:
            recommendation = "Monitor closely – moderate drift detected"
        else:
            recommendation = "No action required – distribution stable"

        report = DriftReport(
            model_name=model_name,
            psi_score=psi,
            ks_statistic=ks,
            is_drifted=is_drifted,
            recommendation=recommendation,
        )
        logger.info("Drift detection for '%s': PSI=%.4f KS=%.4f drifted=%s",
                     model_name, psi, ks, is_drifted)
        return report

    # ------------------------------------------------------------------
    # A/B testing
    # ------------------------------------------------------------------

    def run_ab_test(
        self,
        champion: Any,
        challenger: Any,
        test_data: Dict[str, np.ndarray],
        n_bootstrap: int = 1000,
    ) -> ABTestResult:
        """Run an A/B test comparing *champion* with *challenger*.

        Uses bootstrap resampling to estimate confidence.

        Args:
            champion: Champion model object.
            challenger: Challenger model object.
            test_data: Dictionary with ``X_test`` and ``y_test``.
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            An ``ABTestResult`` with winning model and confidence.
        """
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

        champ_preds = champion.predict(X_test)
        chall_preds = challenger.predict(X_test)

        champ_score = float(self._metric_fn(y_test, champ_preds))
        chall_score = float(self._metric_fn(y_test, chall_preds))

        # Bootstrap confidence
        challenger_wins = 0
        n = len(y_test)
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            bs_champ = float(self._metric_fn(y_test[idx], champ_preds[idx]))
            bs_chall = float(self._metric_fn(y_test[idx], chall_preds[idx]))
            if bs_chall > bs_champ:
                challenger_wins += 1

        confidence = challenger_wins / n_bootstrap
        winner_name = getattr(challenger, "name", "challenger") if confidence > 0.5 else getattr(champion, "name", "champion")
        winner_version = (
            getattr(challenger, "version", "unknown") if confidence > 0.5
            else getattr(champion, "version", "unknown")
        )

        result = ABTestResult(
            champion_version=getattr(champion, "version", "unknown"),
            challenger_version=getattr(challenger, "version", "unknown"),
            champion_metric=champ_score,
            challenger_metric=chall_score,
            winner=winner_version,
            confidence=confidence if confidence > 0.5 else 1.0 - confidence,
        )
        logger.info("A/B test result: champion=%.4f challenger=%.4f winner=%s confidence=%.2f",
                     champ_score, chall_score, result.winner, result.confidence)
        return result

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune_hyperparams(
        self,
        model: Any,
        param_ranges: Dict[str, Tuple[float, float]],
        data: Dict[str, np.ndarray],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Random-search hyperparameter tuning.

        Args:
            model: Model object with ``set_params``, ``fit``, ``predict``.
            param_ranges: Mapping of parameter name → ``(low, high)`` bounds.
            data: Dict with ``X_train``, ``y_train``, ``X_val``, ``y_val``.
            n_trials: Number of random configurations to try.

        Returns:
            Dictionary with ``best_params``, ``best_score``, and ``all_trials``.
        """
        best_score = -np.inf
        best_params: Dict[str, Any] = {}
        all_trials: List[Dict[str, Any]] = []

        for i in range(n_trials):
            params = {
                k: float(np.random.uniform(lo, hi))
                for k, (lo, hi) in param_ranges.items()
            }
            try:
                model.set_params(**params)
                model.fit(data["X_train"], data["y_train"])
                preds = model.predict(data["X_val"])
                score = float(self._metric_fn(data["y_val"], preds))
            except Exception as exc:
                logger.debug("Trial %d failed: %s", i, exc)
                score = -np.inf

            trial = {"trial": i, "params": params, "score": score}
            all_trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = params

        logger.info("Hyperparameter tuning done (%d trials): best_score=%.4f", n_trials, best_score)
        return {"best_params": best_params, "best_score": best_score, "all_trials": all_trials}

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def promote_model(self, version: ModelVersion) -> None:
        """Promote *version* to champion status.

        Args:
            version: The ``ModelVersion`` to promote.
        """
        name = version.name
        if name in self._champion:
            self._champion[name].is_champion = False
        version.is_champion = True
        self._champion[name] = version
        logger.info("Model '%s' version %s promoted to champion", name, version.version)

    def rollback_model(self, to_version: str, model_name: str = "unnamed") -> Optional[ModelVersion]:
        """Rollback to a previous model version.

        Args:
            to_version: Version ID string to revert to.
            model_name: Name of the model lineage.

        Returns:
            The rolled-back ``ModelVersion``, or ``None`` if not found.
        """
        versions = self._versions.get(model_name, [])
        target = next((v for v in versions if v.version == to_version), None)
        if target is None:
            logger.error("Version %s not found for model '%s'", to_version, model_name)
            return None
        self.promote_model(target)
        logger.info("Rolled back model '%s' to version %s", model_name, to_version)
        return target

    def get_retraining_schedule(self) -> Dict[str, Dict[str, Any]]:
        """Return the current retraining schedule for all models.

        Returns:
            Dictionary mapping model names to schedule configuration.
        """
        if not self._schedule:
            return {
                name: {"frequency": "weekly", "last_trained": vers[-1].trained_at if vers else None}
                for name, vers in self._versions.items()
            }
        return dict(self._schedule)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple accuracy metric."""
        return float(np.mean(np.round(y_pred) == y_true))

    @staticmethod
    def _calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index between two distributions."""
        ref = np.array(reference, dtype=float).ravel()
        cur = np.array(current, dtype=float).ravel()
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        breakpoints = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()) + 1e-10,
            bins + 1,
        )
        ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
        cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

        # Avoid zero counts
        ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
        cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    @staticmethod
    def _calculate_ks(reference: np.ndarray, current: np.ndarray) -> float:
        """Two-sample Kolmogorov-Smirnov statistic."""
        ref = np.sort(np.array(reference, dtype=float).ravel())
        cur = np.sort(np.array(current, dtype=float).ravel())
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        all_values = np.sort(np.concatenate([ref, cur]))
        cdf_ref = np.searchsorted(ref, all_values, side="right") / len(ref)
        cdf_cur = np.searchsorted(cur, all_values, side="right") / len(cur)
        return float(np.max(np.abs(cdf_ref - cdf_cur)))
