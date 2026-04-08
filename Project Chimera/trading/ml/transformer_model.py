"""
Transformer-based sentiment analysis model (PyTorch).

Provides a lightweight Transformer encoder with a simple word-level tokenizer
for classifying social-media text into **positive / negative / neutral**
sentiment categories.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Sentiment label mapping
SENTIMENT_LABELS: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_IDX: Dict[str, int] = {v: k for k, v in SENTIMENT_LABELS.items()}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in *Attention Is All You Need*."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Encoder
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module):
    """Thin wrapper around ``nn.TransformerEncoder`` for classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-init_range, init_range)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (batch, seq_len) int tensor of token IDs
        src_key_padding_mask : (batch, seq_len) bool – True where padded

        Returns
        -------
        logits : (batch, num_classes)
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # Mean-pool over non-padded positions
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, S, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)

    def get_attention_weights(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Extract attention weight matrices from each layer.

        Returns a list of tensors each shaped ``(batch, nhead, seq, seq)``.
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        weights: List[torch.Tensor] = []
        for layer in self.transformer_encoder.layers:
            # Use the self-attention sub-layer directly
            attn_out, attn_w = layer.self_attn(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            weights.append(attn_w.detach())
            # Reproduce the rest of the encoder layer forward pass
            x = layer.norm1(x + layer.dropout1(attn_out))
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(x2))
        return weights


# ---------------------------------------------------------------------------
# Simple tokenizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z0-9$#@]+|[!?.,;:]")


def _tokenize(text: str) -> List[str]:
    """Lowercase word-level tokenisation."""
    return [t.lower() for t in _WORD_RE.findall(text)]


# ---------------------------------------------------------------------------
# Sentiment Transformer (public API)
# ---------------------------------------------------------------------------

@dataclass
class SentimentTransformer:
    """Transformer-based sentiment classifier.

    Parameters
    ----------
    d_model : int
        Embedding / transformer hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feed-forward intermediate dimension.
    dropout : float
        Dropout probability.
    max_len : int
        Maximum token sequence length (truncation / padding boundary).
    min_freq : int
        Minimum word frequency to include in vocabulary.
    device : str
        PyTorch device string.
    """

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = 128
    min_freq: int = 2
    device: str = "cpu"

    _vocab: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _idx2word: Dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _model: Optional[_TransformerEncoder] = field(default=None, init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        self._vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self._idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    # ------------------------------------------------------------------ #
    # Vocabulary
    # ------------------------------------------------------------------ #
    def build_vocab(self, texts: List[str]) -> int:
        """Build vocabulary from a corpus.  Returns vocab size."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(_tokenize(text))

        idx = len(self._vocab)
        for word, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if word not in self._vocab:
                self._vocab[word] = idx
                self._idx2word[idx] = word
                idx += 1

        logger.info("Vocabulary built: %d tokens (min_freq=%d)", len(self._vocab), self.min_freq)
        self._init_model()
        return len(self._vocab)

    def _init_model(self) -> None:
        self._model = _TransformerEncoder(
            vocab_size=len(self._vocab),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_len=self.max_len,
            num_classes=3,
        ).to(self._device)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode_text(self, text: str) -> List[int]:
        """Convert raw text to a list of token IDs."""
        tokens = _tokenize(text)
        return [self._vocab.get(t, UNK_IDX) for t in tokens[: self.max_len]]

    def _encode_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode and pad a batch of texts.

        Returns (ids, padding_mask) tensors.
        """
        encoded = [self.encode_text(t) for t in texts]
        max_len = max(len(e) for e in encoded) if encoded else 1
        padded = np.full((len(encoded), max_len), PAD_IDX, dtype=np.int64)
        mask = np.ones((len(encoded), max_len), dtype=bool)
        for i, seq in enumerate(encoded):
            padded[i, : len(seq)] = seq
            mask[i, : len(seq)] = False
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 5,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train the sentiment classifier.

        Parameters
        ----------
        labels : list[str]
            ``"positive"``, ``"negative"``, or ``"neutral"`` per sample.

        Returns
        -------
        dict  with ``train_loss`` and ``val_loss`` lists.
        """
        if self._model is None:
            self.build_vocab(texts)
        assert self._model is not None

        label_ids = np.array([LABEL_TO_IDX[l] for l in labels], dtype=np.int64)

        # Encode all texts
        ids_tensor, mask_tensor = self._encode_batch(texts)
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        split = int(len(texts) * (1 - validation_split))
        train_ds = TensorDataset(ids_tensor[:split], mask_tensor[:split], label_tensor[:split])
        val_ds = TensorDataset(ids_tensor[split:], mask_tensor[split:], label_tensor[split:])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = AdamW(self._model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = math.inf
        wait = 0
        best_state = self._model.state_dict()

        for epoch in range(1, epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            for ids_b, mask_b, lab_b in train_dl:
                ids_b = ids_b.to(self._device)
                mask_b = mask_b.to(self._device)
                lab_b = lab_b.to(self._device)
                optimizer.zero_grad()
                logits = self._model(ids_b, src_key_padding_mask=mask_b)
                loss = criterion(logits, lab_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(ids_b)
            epoch_loss /= max(len(train_ds), 1)
            scheduler.step()

            # Validation
            val_loss = self._eval_loss(val_dl, criterion)
            history["train_loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 5 == 0 or wait == 0:
                logger.info(
                    "Epoch %d/%d  train=%.4f  val=%.4f", epoch, epochs, epoch_loss, val_loss,
                )
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        self._model.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Return ``(label, confidence)`` for a single text."""
        if self._model is None:
            raise RuntimeError("Model has not been trained.")
        self._model.eval()
        ids, mask = self._encode_batch([text])
        ids, mask = ids.to(self._device), mask.to(self._device)
        with torch.no_grad():
            logits = self._model(ids, src_key_padding_mask=mask)
            probs = F.softmax(logits, dim=-1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        return SENTIMENT_LABELS[idx], float(probs[idx].item())

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for multiple texts."""
        if self._model is None:
            raise RuntimeError("Model has not been trained.")
        self._model.eval()
        ids, mask = self._encode_batch(texts)
        ids, mask = ids.to(self._device), mask.to(self._device)
        with torch.no_grad():
            logits = self._model(ids, src_key_padding_mask=mask)
            probs = F.softmax(logits, dim=-1)
        results: List[Tuple[str, float]] = []
        for i in range(len(texts)):
            idx = int(torch.argmax(probs[i]).item())
            results.append((SENTIMENT_LABELS[idx], float(probs[i, idx].item())))
        return results

    # ------------------------------------------------------------------ #
    # Attention inspection
    # ------------------------------------------------------------------ #
    def get_attention_weights(self, text: str) -> List[np.ndarray]:
        """Return per-layer attention weights for interpretability.

        Returns a list of arrays each shaped ``(nhead, seq, seq)``.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained.")
        self._model.eval()
        ids, mask = self._encode_batch([text])
        ids, mask = ids.to(self._device), mask.to(self._device)
        weights = self._model.get_attention_weights(ids, src_key_padding_mask=mask)
        return [w.squeeze(0).cpu().numpy() for w in weights]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save model weights and vocabulary."""
        if self._model is None:
            raise RuntimeError("Model has not been trained.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "vocab": self._vocab,
                "config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "max_len": self.max_len,
                },
            },
            path,
        )
        logger.info("SentimentTransformer saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SentimentTransformer":
        """Load a saved model."""
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        obj = cls(**cfg, device=device)
        obj._vocab = data["vocab"]
        obj._idx2word = {v: k for k, v in obj._vocab.items()}
        obj._init_model()
        assert obj._model is not None
        obj._model.load_state_dict(data["model_state"])
        logger.info("SentimentTransformer loaded from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _eval_loss(self, dl: DataLoader, criterion: nn.Module) -> float:
        assert self._model is not None
        self._model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for ids_b, mask_b, lab_b in dl:
                ids_b = ids_b.to(self._device)
                mask_b = mask_b.to(self._device)
                lab_b = lab_b.to(self._device)
                logits = self._model(ids_b, src_key_padding_mask=mask_b)
                total += criterion(logits, lab_b).item() * len(ids_b)
                count += len(ids_b)
        return total / max(count, 1)
