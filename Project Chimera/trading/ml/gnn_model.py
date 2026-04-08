"""
Graph Neural Network for wallet behaviour classification (PyTorch).

Models Solana wallet interactions as a graph and uses Graph Convolutional
Network (GCN) layers to classify wallets as **whale**, **bot**, **retail**,
or **smart_money**.  Also supports link-prediction for relationship
forecasting and embedding extraction.
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
import torch.nn.functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)

# Node-type labels
NODE_TYPES: Dict[int, str] = {0: "whale", 1: "bot", 2: "retail", 3: "smart_money"}
TYPE_TO_IDX: Dict[str, int] = {v: k for k, v in NODE_TYPES.items()}
NUM_NODE_TYPES = len(NODE_TYPES)

# ---------------------------------------------------------------------------
# GCN Layer
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Single Graph Convolutional layer.

    Implements the propagation rule::

        H' = σ( D̃⁻¹/² Ã D̃⁻¹/²  H  W )

    where Ã = A + I  and  D̃ is the degree matrix of Ã.
    Supports both dense and ``torch.sparse`` adjacency matrices.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (N, in_features)  – node feature matrix
        adj : (N, N)            – normalised adjacency (dense or sparse)
        """
        support = x @ self.weight  # (N, out)
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = adj @ support
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Multi-layer GCN module
# ---------------------------------------------------------------------------

class _GCNClassifier(nn.Module):
    """Stack of ``GCNLayer`` layers ending in a classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_classes: int = NUM_NODE_TYPES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[GCNLayer] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(GCNLayer(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, adj)  # logits

    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward through all but the last layer to get node embeddings."""
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj))
        return x


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _normalise_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: D̃⁻¹/² Ã D̃⁻¹/² where Ã = A + I."""
    n = adj.size(0)
    # Add self-loops
    if adj.is_sparse:
        indices = torch.arange(n, device=adj.device)
        self_loops = torch.sparse_coo_tensor(
            torch.stack([indices, indices]),
            torch.ones(n, device=adj.device),
            size=(n, n),
        )
        adj = adj + self_loops
        # Convert to dense for degree calculation then back
        adj_dense = adj.to_dense()
    else:
        adj_dense = adj + torch.eye(n, device=adj.device)

    deg = adj_dense.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt.unsqueeze(1) * adj_dense * deg_inv_sqrt.unsqueeze(0)
    return norm


def build_graph_from_transactions(
    wallets: List[Dict],
    transactions: List[Dict],
    feature_keys: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct adjacency and feature matrices from wallet / transaction data.

    Parameters
    ----------
    wallets : list[dict]
        Each dict must contain ``"address"`` and numeric feature fields.
    transactions : list[dict]
        Each dict must contain ``"from"`` and ``"to"`` address strings and
        an optional ``"amount"`` float.
    feature_keys : list[str] | None
        Keys to extract from each wallet dict as node features.
        Defaults to ``["balance", "tx_count", "age_days"]``.

    Returns
    -------
    (adj, features) : (N, N) float tensor, (N, F) float tensor
    """
    if feature_keys is None:
        feature_keys = ["balance", "tx_count", "age_days"]

    addr_to_idx: Dict[str, int] = {}
    for i, w in enumerate(wallets):
        addr_to_idx[w["address"]] = i

    n = len(wallets)
    adj = torch.zeros(n, n)
    for tx in transactions:
        src = addr_to_idx.get(tx.get("from", ""))
        dst = addr_to_idx.get(tx.get("to", ""))
        if src is not None and dst is not None:
            weight = float(tx.get("amount", 1.0))
            adj[src, dst] += weight
            adj[dst, src] += weight  # undirected

    features = torch.zeros(n, len(feature_keys))
    for i, w in enumerate(wallets):
        for j, key in enumerate(feature_keys):
            features[i, j] = float(w.get(key, 0.0))

    return adj, features


# ---------------------------------------------------------------------------
# WalletGNN (public API)
# ---------------------------------------------------------------------------

@dataclass
class WalletGNN:
    """Graph Neural Network for Solana wallet classification.

    Parameters
    ----------
    input_dim : int
        Number of features per node.
    hidden_dim : int
        Hidden dimension of GCN layers.
    num_layers : int
        Number of GCN layers.
    dropout : float
        Dropout probability.
    device : str
        PyTorch device string.
    """

    input_dim: int = 3
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    device: str = "cpu"

    _model: Optional[_GCNClassifier] = field(default=None, init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        self._model = _GCNClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self._device)
        logger.info(
            "WalletGNN initialised: input=%d  hidden=%d  layers=%d  params=%d",
            self.input_dim, self.hidden_dim, self.num_layers,
            sum(p.numel() for p in self._model.parameters()),
        )

    # ------------------------------------------------------------------ #
    # Graph building convenience
    # ------------------------------------------------------------------ #
    def build_graph(
        self,
        wallets: List[Dict],
        transactions: List[Dict],
        feature_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build normalised adjacency and feature matrices.

        Returns ``(adj_norm, features)`` on ``self._device``.
        """
        adj, features = build_graph_from_transactions(wallets, transactions, feature_keys)
        adj_norm = _normalise_adjacency(adj)
        return adj_norm.to(self._device), features.to(self._device)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
        labels: np.ndarray,
        train_mask: Optional[np.ndarray] = None,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 5e-4,
        patience: int = 20,
    ) -> Dict[str, List[float]]:
        """Semi-supervised training on labelled nodes.

        Parameters
        ----------
        labels : (N,) int array with values in ``{0,1,2,3}`` or ``-1``
            for unlabelled.
        train_mask : (N,) bool array – nodes to use for loss.  If *None*
            all nodes with ``label >= 0`` are used.
        """
        assert self._model is not None
        adj = adj.to(self._device)
        features = features.to(self._device)
        label_t = torch.tensor(labels, dtype=torch.long, device=self._device)

        if train_mask is None:
            train_mask = labels >= 0
        mask_t = torch.tensor(train_mask, dtype=torch.bool, device=self._device)

        optimizer = Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}
        best_loss = math.inf
        wait = 0
        best_state = self._model.state_dict()

        for epoch in range(1, epochs + 1):
            self._model.train()
            optimizer.zero_grad()
            logits = self._model(features, adj)
            loss = criterion(logits[mask_t], label_t[mask_t])
            loss.backward()
            optimizer.step()

            # Accuracy on labelled nodes
            with torch.no_grad():
                preds = logits[mask_t].argmax(dim=-1)
                acc = float((preds == label_t[mask_t]).float().mean().item())
            history["loss"].append(loss.item())
            history["accuracy"].append(acc)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if epoch % 50 == 0:
                logger.info(
                    "GNN epoch %d/%d  loss=%.4f  acc=%.3f", epoch, epochs, loss.item(), acc,
                )
            if wait >= patience:
                logger.info("GNN early stopping at epoch %d", epoch)
                break

        self._model.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_node_type(
        self, adj: torch.Tensor, features: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        """Classify each node.  Returns ``[(label, confidence), ...]``."""
        assert self._model is not None
        self._model.eval()
        adj, features = adj.to(self._device), features.to(self._device)
        with torch.no_grad():
            logits = self._model(features, adj)
            probs = F.softmax(logits, dim=-1)
        results: List[Tuple[str, float]] = []
        for i in range(probs.size(0)):
            idx = int(torch.argmax(probs[i]).item())
            results.append((NODE_TYPES[idx], float(probs[i, idx].item())))
        return results

    # ------------------------------------------------------------------ #
    # Link prediction
    # ------------------------------------------------------------------ #
    def predict_links(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
        node_pairs: List[Tuple[int, int]],
    ) -> List[float]:
        """Predict link probability for given node pairs using dot-product
        similarity of learned embeddings."""
        embeddings = self.get_embeddings(adj, features)
        scores: List[float] = []
        for i, j in node_pairs:
            sim = float(F.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0),
            ).item())
            scores.append((sim + 1.0) / 2.0)  # map to [0, 1]
        return scores

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def get_embeddings(
        self, adj: torch.Tensor, features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract node embeddings from the penultimate GCN layer."""
        assert self._model is not None
        self._model.eval()
        adj, features = adj.to(self._device), features.to(self._device)
        with torch.no_grad():
            return self._model.get_embeddings(features, adj)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Save model weights and config."""
        assert self._model is not None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
            },
            path,
        )
        logger.info("WalletGNN saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "WalletGNN":
        """Load a saved model."""
        data = torch.load(path, map_location=device, weights_only=False)
        cfg = data["config"]
        obj = cls(**cfg, device=device)
        assert obj._model is not None
        obj._model.load_state_dict(data["model_state"])
        logger.info("WalletGNN loaded from %s", path)
        return obj
