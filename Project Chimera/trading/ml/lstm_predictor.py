"""
LSTM / GRU price-prediction models built on PyTorch.

Provides two recurrent architectures (``PriceLSTM`` and ``PriceGRU``) wrapped
by a convenience class (``LSTMPredictor``) that handles training, evaluation,
Monte-Carlo-dropout confidence intervals, and model persistence.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recurrent modules
# ---------------------------------------------------------------------------

class PriceLSTM(nn.Module):
    """Multi-layer LSTM with optional bidirectionality and dropout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (batch, seq_len, input_size)."""
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # last time-step
        return self.fc(out)


class PriceGRU(nn.Module):
    """Drop-in replacement using GRU cells."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


# ---------------------------------------------------------------------------
# Sliding-window helper
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray,
    targets: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences from time-series data.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix of shape ``(T, F)``.
    targets : np.ndarray
        Target values of shape ``(T,)`` or ``(T, 1)``.
    window_size : int
        Number of time steps per sequence.

    Returns
    -------
    X : np.ndarray – (N, window_size, F)
    y : np.ndarray – (N,) or (N, D)
    """
    targets = targets.reshape(len(targets), -1) if targets.ndim == 1 else targets
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size: i])
        y.append(targets[i])
    return np.array(X), np.squeeze(np.array(y))


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------

@dataclass
class LSTMPredictor:
    """High-level wrapper for training / inference with recurrent models.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    hidden_size : int
        LSTM / GRU hidden dimension.
    num_layers : int
        Number of stacked recurrent layers.
    dropout : float
        Dropout probability.
    bidirectional : bool
        Use bidirectional recurrence.
    window_size : int
        Sliding-window length for sequence creation.
    cell_type : str
        ``"lstm"`` or ``"gru"``.
    device : str
        PyTorch device string.
    """

    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    window_size: int = 30
    cell_type: str = "lstm"
    device: str = "cpu"

    # internal state (populated after __post_init__)
    _model: nn.Module = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)
    _trained: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        cls = PriceLSTM if self.cell_type == "lstm" else PriceGRU
        self._model = cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        ).to(self._device)
        logger.info(
            "Initialised %s predictor (%s params)",
            self.cell_type.upper(),
            sum(p.numel() for p in self._model.parameters()),
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 10,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...]}``.
        """
        X, y = create_sequences(data, targets, self.window_size)
        split = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = AdamW(self._model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = math.inf
        wait = 0
        best_state = self._model.state_dict()

        for epoch in range(1, epochs + 1):
            # --- train ---
            self._model.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                pred = self._model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            scheduler.step()

            # --- validate ---
            val_loss = self._evaluate_loss(val_dl, criterion)
            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 10 == 0 or wait == 0:
                logger.info(
                    "Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
                    epoch, epochs, epoch_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )
            if wait >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

        self._model.load_state_dict(best_state)
        self._trained = True
        return history

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, sequence: np.ndarray) -> float:
        """Point prediction for a single sequence.

        Parameters
        ----------
        sequence : np.ndarray
            Shape ``(window_size, input_size)`` or ``(1, window_size, input_size)``.
        """
        self._model.eval()
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]
        with torch.no_grad():
            x = torch.tensor(sequence, dtype=torch.float32).to(self._device)
            return self._model(x).squeeze().item()

    def predict_with_confidence(
        self,
        sequence: np.ndarray,
        n_samples: int = 50,
    ) -> Tuple[float, float, float]:
        """Monte-Carlo dropout prediction with confidence interval.

        Returns ``(mean, lower_95, upper_95)``.
        """
        self._model.train()  # keep dropout active
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]
        x = torch.tensor(sequence, dtype=torch.float32).to(self._device)

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self._model(x).squeeze().item())
        arr = np.array(preds)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return mean, mean - 1.96 * std, mean + 1.96 * std

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, test_data: np.ndarray, test_targets: np.ndarray) -> Dict[str, float]:
        """Evaluate on held-out test data.  Returns MSE, MAE, RMSE."""
        X, y = create_sequences(test_data, test_targets, self.window_size)
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=128)

        self._model.eval()
        all_preds: List[float] = []
        all_true: List[float] = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self._device)
                pred = self._model(xb).squeeze(-1)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_true.extend(yb.numpy().tolist())

        preds_arr = np.array(all_preds)
        true_arr = np.array(all_true)
        mse = float(np.mean((preds_arr - true_arr) ** 2))
        mae = float(np.mean(np.abs(preds_arr - true_arr)))
        rmse = float(np.sqrt(mse))
        logger.info("Evaluation  MSE=%.6f  MAE=%.6f  RMSE=%.6f", mse, mae, rmse)
        return {"mse": mse, "mae": mae, "rmse": rmse}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist model weights and config."""
        state = {
            "model_state": self._model.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "window_size": self.window_size,
                "cell_type": self.cell_type,
            },
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LSTMPredictor":
        """Load a previously saved predictor."""
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = state["config"]
        predictor = cls(**cfg, device=device)
        predictor._model.load_state_dict(state["model_state"])
        predictor._trained = True
        logger.info("Model loaded from %s", path)
        return predictor

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _evaluate_loss(self, dl: DataLoader, criterion: nn.Module) -> float:
        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                pred = self._model(xb).squeeze(-1)
                total += criterion(pred, yb).item() * len(xb)
                count += len(xb)
        return total / max(count, 1)
