"""
Multi-model ensemble for unified trading signal generation.

Aggregates predictions from heterogeneous models (LSTM, tree-based,
transformer, GNN, RL) via weighted averaging with dynamic weight
adjustment, Platt-scaling confidence calibration, and disagreement
detection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal enum
# ---------------------------------------------------------------------------

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


SIGNAL_NUMERIC: Dict[Signal, float] = {Signal.BUY: 1.0, Signal.SELL: -1.0, Signal.HOLD: 0.0}
NUMERIC_SIGNAL: Dict[int, Signal] = {1: Signal.BUY, -1: Signal.SELL, 0: Signal.HOLD}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """Single model's prediction."""
    model_name: str
    signal: Signal
    confidence: float
    timestamp: float = field(default_factory=time.time)

    @property
    def numeric(self) -> float:
        return SIGNAL_NUMERIC[self.signal] * self.confidence


@dataclass
class EnsembleSignal:
    """Aggregated ensemble output."""
    signal: Signal
    confidence: float
    model_agreement: float
    individual_predictions: List[ModelPrediction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model protocol
# ---------------------------------------------------------------------------

class PredictorProtocol(Protocol):
    """Minimal interface a model must satisfy to participate in the ensemble."""

    def predict(self, features: Any) -> float:
        """Return a numeric prediction (positive = buy, negative = sell)."""
        ...


# ---------------------------------------------------------------------------
# Platt scaling (sigmoid calibration)
# ---------------------------------------------------------------------------

def _platt_fit(raw_scores: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> Tuple[float, float]:
    """Fit Platt scaling parameters A, B via Newton's method.

    Minimises NLL of ``P(y=1 | f) = 1 / (1 + exp(A*f + B))``.
    """
    A, B = 0.0, 0.0
    n_pos = float(np.sum(labels == 1))
    n_neg = float(len(labels) - n_pos)
    # Target probabilities (Platt's smoothing)
    t = np.where(labels == 1, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))
    lr = 1e-2

    for _ in range(max_iter):
        fApB = A * raw_scores + B
        # Numerically stable sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(fApB, -500, 500)))
        p = np.clip(p, 1e-12, 1 - 1e-12)

        d1 = t - p
        dA = float(np.dot(d1, raw_scores))
        dB = float(np.sum(d1))
        A += lr * dA
        B += lr * dB

    return A, B


def _platt_predict(score: float, A: float, B: float) -> float:
    """Apply Platt scaling to a raw score, returning calibrated probability."""
    return 1.0 / (1.0 + np.exp(-np.clip(A * score + B, -500, 500)))


# ---------------------------------------------------------------------------
# Model Ensemble
# ---------------------------------------------------------------------------

@dataclass
class ModelEnsemble:
    """Multi-model ensemble with dynamic weighting and calibration.

    Parameters
    ----------
    confidence_threshold : float
        Minimum ensemble confidence to emit a non-HOLD signal.
    agreement_threshold : float
        Minimum model agreement fraction required for a signal.
    window : int
        Number of recent predictions to consider for weight adaptation.
    """

    confidence_threshold: float = 0.5
    agreement_threshold: float = 0.5
    window: int = 50

    _models: Dict[str, Any] = field(default_factory=dict, repr=False)
    _predict_fns: Dict[str, Callable] = field(default_factory=dict, repr=False)
    _weights: Dict[str, float] = field(default_factory=dict, repr=False)
    _recent_accuracy: Dict[str, List[float]] = field(default_factory=dict, repr=False)
    _platt_params: Dict[str, Tuple[float, float]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    # Model registration
    # ------------------------------------------------------------------ #
    def add_model(
        self,
        name: str,
        model: Any,
        predict_fn: Optional[Callable] = None,
        initial_weight: float = 1.0,
    ) -> None:
        """Register a model in the ensemble.

        Parameters
        ----------
        predict_fn : callable, optional
            Custom ``fn(model, features) -> float`` wrapper.  If *None*,
            ``model.predict(features)`` is called directly.
        """
        self._models[name] = model
        self._predict_fns[name] = predict_fn or (lambda m, f: float(m.predict(f)))
        self._weights[name] = initial_weight
        self._recent_accuracy[name] = []
        self._platt_params[name] = (1.0, 0.0)  # identity
        logger.info("Ensemble: registered model '%s' (weight=%.2f)", name, initial_weight)

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        for store in (self._models, self._predict_fns, self._weights,
                      self._recent_accuracy, self._platt_params):
            store.pop(name, None)
        logger.info("Ensemble: removed model '%s'", name)

    # ------------------------------------------------------------------ #
    # Individual model prediction
    # ------------------------------------------------------------------ #
    def _get_model_prediction(
        self, name: str, features: Any,
    ) -> ModelPrediction:
        model = self._models[name]
        fn = self._predict_fns[name]
        try:
            raw = fn(model, features)
        except Exception:
            logger.exception("Model '%s' prediction failed; defaulting to HOLD", name)
            return ModelPrediction(model_name=name, signal=Signal.HOLD, confidence=0.0)

        # Calibrate
        A, B = self._platt_params[name]
        calibrated_conf = _platt_predict(abs(raw), A, B)
        calibrated_conf = float(np.clip(calibrated_conf, 0.0, 1.0))

        if raw > 0:
            signal = Signal.BUY
        elif raw < 0:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        return ModelPrediction(
            model_name=name, signal=signal, confidence=calibrated_conf,
        )

    # ------------------------------------------------------------------ #
    # Ensemble prediction
    # ------------------------------------------------------------------ #
    def get_ensemble_prediction(self, features: Any) -> EnsembleSignal:
        """Query all registered models and combine their predictions."""
        if not self._models:
            return EnsembleSignal(
                signal=Signal.HOLD, confidence=0.0, model_agreement=0.0,
            )

        predictions: List[ModelPrediction] = []
        for name in self._models:
            pred = self._get_model_prediction(name, features)
            predictions.append(pred)

        return self._aggregate(predictions)

    def _aggregate(self, predictions: List[ModelPrediction]) -> EnsembleSignal:
        """Weighted combination of individual predictions."""
        total_weight = 0.0
        weighted_score = 0.0

        for pred in predictions:
            w = self._weights.get(pred.model_name, 1.0)
            weighted_score += pred.numeric * w
            total_weight += w

        if total_weight == 0:
            return EnsembleSignal(
                signal=Signal.HOLD, confidence=0.0, model_agreement=0.0,
                individual_predictions=predictions,
            )

        avg_score = weighted_score / total_weight
        confidence = float(min(abs(avg_score), 1.0))

        if avg_score > 0:
            raw_signal = Signal.BUY
        elif avg_score < 0:
            raw_signal = Signal.SELL
        else:
            raw_signal = Signal.HOLD

        # Agreement: fraction of models that agree with the ensemble signal
        agree_count = sum(1 for p in predictions if p.signal == raw_signal)
        agreement = agree_count / len(predictions)

        # Apply thresholds
        if confidence < self.confidence_threshold or agreement < self.agreement_threshold:
            final_signal = Signal.HOLD
        else:
            final_signal = raw_signal

        return EnsembleSignal(
            signal=final_signal,
            confidence=confidence,
            model_agreement=agreement,
            individual_predictions=predictions,
        )

    # ------------------------------------------------------------------ #
    # Weight calibration
    # ------------------------------------------------------------------ #
    def calibrate_weights(
        self,
        validation_data: List[Tuple[Any, Signal]],
    ) -> Dict[str, float]:
        """Adjust model weights based on accuracy on labelled validation data.

        Parameters
        ----------
        validation_data : list[(features, true_signal)]

        Returns
        -------
        dict
            Updated weight per model.
        """
        if not validation_data:
            return dict(self._weights)

        for name in self._models:
            correct = 0
            raw_scores: List[float] = []
            binary_labels: List[int] = []

            for features, true_signal in validation_data:
                pred = self._get_model_prediction(name, features)
                if pred.signal == true_signal:
                    correct += 1

                # Collect data for Platt scaling re-fit
                fn = self._predict_fns[name]
                try:
                    raw = fn(self._models[name], features)
                except Exception:
                    raw = 0.0
                raw_scores.append(abs(raw))
                binary_labels.append(1 if true_signal != Signal.HOLD else 0)

            accuracy = correct / len(validation_data)
            self._recent_accuracy[name].append(accuracy)
            # Keep only recent window
            self._recent_accuracy[name] = self._recent_accuracy[name][-self.window:]
            # Weight = mean recent accuracy (floor at 0.01 to avoid zero)
            self._weights[name] = max(float(np.mean(self._recent_accuracy[name])), 0.01)

            # Re-fit Platt scaling
            if len(set(binary_labels)) > 1:
                arr_scores = np.array(raw_scores)
                arr_labels = np.array(binary_labels)
                A, B = _platt_fit(arr_scores, arr_labels)
                self._platt_params[name] = (A, B)

            logger.info(
                "Calibrated '%s': accuracy=%.3f  weight=%.3f",
                name, accuracy, self._weights[name],
            )

        return dict(self._weights)

    # ------------------------------------------------------------------ #
    # Disagreement detection
    # ------------------------------------------------------------------ #
    def detect_disagreement(self, features: Any) -> float:
        """Return entropy of model predictions (higher = more disagreement).

        Range: 0 (full agreement) to log2(3) ≈ 1.585 (uniform over 3 classes).
        """
        if not self._models:
            return 0.0

        signal_counts: Dict[Signal, float] = {s: 0.0 for s in Signal}
        total_w = 0.0
        for name in self._models:
            pred = self._get_model_prediction(name, features)
            w = self._weights.get(name, 1.0)
            signal_counts[pred.signal] += w
            total_w += w

        if total_w == 0:
            return 0.0

        probs = np.array([signal_counts[s] / total_w for s in Signal])
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        return entropy

    # ------------------------------------------------------------------ #
    # Unified signal (convenience)
    # ------------------------------------------------------------------ #
    def get_signal(self, market_data: Any) -> EnsembleSignal:
        """Convenience alias matching the spec; delegates to ``get_ensemble_prediction``."""
        return self.get_ensemble_prediction(market_data)
