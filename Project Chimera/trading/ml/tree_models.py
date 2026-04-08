"""
Tree-based models implemented in pure NumPy (no sklearn / xgboost).

Provides:
* ``DecisionTree`` – CART-style decision tree (classification & regression).
* ``RandomForestModel`` – bagged ensemble of ``DecisionTree`` instances.
* ``GradientBoostedTrees`` – sequential residual fitting with L2
  regularisation and shrinkage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decision Tree (CART)
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    """Internal representation of a tree node."""

    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    value: Optional[float] = None  # leaf prediction


class DecisionTree:
    """CART decision tree supporting both classification (Gini) and
    regression (MSE) splits.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    min_samples_leaf : int
        Minimum number of samples required at a leaf node.
    criterion : str
        ``"gini"`` for classification or ``"mse"`` for regression.
    max_features : int | None
        If set, randomly sample this many candidate features per split.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 2,
        criterion: str = "mse",
        max_features: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self._root: Optional[_TreeNode] = None
        self._n_features: int = 0
        self._split_counts: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # Impurity measures
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - np.mean(y)) ** 2))

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self.criterion == "gini" else self._mse(y)

    # ------------------------------------------------------------------ #
    # Best split search
    # ------------------------------------------------------------------ #
    def _best_split(
        self, X: np.ndarray, y: np.ndarray,
    ) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feat: Optional[int] = None
        best_thr: Optional[float] = None
        best_left: Optional[np.ndarray] = None
        best_right: Optional[np.ndarray] = None
        parent_impurity = self._impurity(y)

        feature_indices = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(
                n_features, self.max_features, replace=False,
            )

        for feat in feature_indices:
            values = np.unique(X[:, feat])
            if len(values) <= 1:
                continue
            # Evaluate midpoints between consecutive unique values
            thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                left_y, right_y = y[left_mask], y[right_mask]
                n_l, n_r = len(left_y), len(right_y)
                gain = parent_impurity - (
                    (n_l / n_samples) * self._impurity(left_y)
                    + (n_r / n_samples) * self._impurity(right_y)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(feat)
                    best_thr = float(thr)
                    best_left = left_mask
                    best_right = right_mask

        return best_feat, best_thr, best_left, best_right

    # ------------------------------------------------------------------ #
    # Recursive builder
    # ------------------------------------------------------------------ #
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        # Leaf conditions
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return _TreeNode(value=self._leaf_value(y))

        feat, thr, left_mask, right_mask = self._best_split(X, y)
        if feat is None:
            return _TreeNode(value=self._leaf_value(y))

        self._split_counts[feat] = self._split_counts.get(feat, 0) + 1
        left_node = self._build(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(feature_idx=feat, threshold=thr, left=left_node, right=right_node)

    def _leaf_value(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            values, counts = np.unique(y, return_counts=True)
            return float(values[np.argmax(counts)])
        return float(np.mean(y))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the decision tree on ``X`` (N, F) and ``y`` (N,)."""
        self._n_features = X.shape[1]
        self._split_counts = {}
        self._root = self._build(X, y, depth=0)
        logger.info(
            "DecisionTree trained: depth≤%d  features=%d  criterion=%s",
            self.max_depth, self._n_features, self.criterion,
        )

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:  # type: ignore[arg-type]
            return self._predict_one(x, node.left)  # type: ignore[arg-type]
        return self._predict_one(x, node.right)  # type: ignore[arg-type]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for ``X`` (N, F)."""
        if self._root is None:
            raise RuntimeError("Model has not been trained.")
        return np.array([self._predict_one(x, self._root) for x in X])

    def get_feature_importance(self) -> Dict[int, float]:
        """Feature importance based on split frequency."""
        total = sum(self._split_counts.values()) or 1
        return {k: v / total for k, v in sorted(self._split_counts.items())}


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel:
    """Bagged ensemble of ``DecisionTree`` instances.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int
        Maximum depth of each tree.
    min_samples_leaf : int
        Minimum samples per leaf.
    max_features_ratio : float
        Fraction of features to consider at each split.
    criterion : str
        ``"gini"`` or ``"mse"``.
    bootstrap_ratio : float
        Fraction of the dataset sampled (with replacement) per tree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 2,
        max_features_ratio: float = 0.7,
        criterion: str = "mse",
        bootstrap_ratio: float = 1.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_ratio = max_features_ratio
        self.criterion = criterion
        self.bootstrap_ratio = bootstrap_ratio
        self._trees: List[DecisionTree] = []

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all trees on bootstrap samples of ``(X, y)``."""
        n_samples, n_features = X.shape
        max_feats = max(1, int(n_features * self.max_features_ratio))
        sample_size = max(1, int(n_samples * self.bootstrap_ratio))
        self._trees = []

        for i in range(self.n_estimators):
            idx = np.random.choice(n_samples, size=sample_size, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_feats,
            )
            tree.train(X[idx], y[idx])
            self._trees.append(tree)

        logger.info(
            "RandomForest trained: %d trees  max_depth=%d",
            self.n_estimators, self.max_depth,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aggregate tree predictions (mean for regression, majority vote for classification)."""
        if not self._trees:
            raise RuntimeError("Model has not been trained.")
        preds = np.array([t.predict(X) for t in self._trees])  # (T, N)
        if self.criterion == "gini":
            # Majority vote per sample
            result = np.zeros(X.shape[0])
            for j in range(X.shape[0]):
                vals, counts = np.unique(preds[:, j], return_counts=True)
                result[j] = vals[np.argmax(counts)]
            return result
        return np.mean(preds, axis=0)

    def get_feature_importance(self) -> Dict[int, float]:
        """Average feature importance across all trees."""
        combined: Dict[int, float] = {}
        for tree in self._trees:
            for k, v in tree.get_feature_importance().items():
                combined[k] = combined.get(k, 0.0) + v
        n = len(self._trees) or 1
        return {k: v / n for k, v in sorted(combined.items())}

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, k_folds: int = 5,
    ) -> Dict[str, float]:
        """K-fold cross-validation.  Returns mean & std of error metric."""
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k_folds
        scores: List[float] = []

        for k in range(k_folds):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = np.concatenate([indices[: k * fold_size], indices[(k + 1) * fold_size:]])
            self.train(X[train_idx], y[train_idx])
            preds = self.predict(X[val_idx])
            if self.criterion == "gini":
                score = float(np.mean(preds == y[val_idx]))
            else:
                score = float(np.mean((preds - y[val_idx]) ** 2))
            scores.append(score)

        metric_name = "accuracy" if self.criterion == "gini" else "mse"
        result = {
            f"mean_{metric_name}": float(np.mean(scores)),
            f"std_{metric_name}": float(np.std(scores)),
        }
        logger.info("CV (%d-fold): %s", k_folds, result)
        return result


# ---------------------------------------------------------------------------
# Gradient Boosted Trees
# ---------------------------------------------------------------------------

class GradientBoostedTrees:
    """Gradient-boosted decision-tree ensemble (regression).

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum depth of each base learner.
    learning_rate : float
        Shrinkage factor applied to each tree's contribution.
    min_samples_leaf : int
        Minimum samples per leaf of base learners.
    l2_reg : float
        L2 regularisation strength applied as additional weight to leaf
        values (analogous to XGBoost ``lambda``).
    subsample : float
        Fraction of training rows used per round (stochastic gradient
        boosting).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 5,
        l2_reg: float = 1.0,
        subsample: float = 0.8,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg = l2_reg
        self.subsample = subsample

        self._trees: List[DecisionTree] = []
        self._init_pred: float = 0.0

    # ------------------------------------------------------------------ #
    # L2-regularised leaf value helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _regularised_leaf(y: np.ndarray, l2: float) -> float:
        """Optimal leaf value: sum(residuals) / (count + lambda)."""
        return float(np.sum(y) / (len(y) + l2))

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Fit the boosted ensemble.  Returns per-round training loss."""
        self._init_pred = float(np.mean(y))
        current_pred = np.full(len(y), self._init_pred)
        self._trees = []
        n_samples = len(X)
        sample_size = max(1, int(n_samples * self.subsample))
        losses: List[float] = []

        for i in range(self.n_estimators):
            residuals = y - current_pred

            # Stochastic subsample
            idx = np.random.choice(n_samples, size=sample_size, replace=False)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion="mse",
            )
            tree.train(X[idx], residuals[idx])

            # Apply L2 regularisation to leaf values
            self._regularise_tree(tree._root, self.l2_reg)

            update = tree.predict(X)
            current_pred += self.learning_rate * update
            self._trees.append(tree)

            mse = float(np.mean((y - current_pred) ** 2))
            losses.append(mse)
            if (i + 1) % 20 == 0:
                logger.info("GBT round %d/%d  MSE=%.6f", i + 1, self.n_estimators, mse)

        logger.info(
            "GradientBoostedTrees trained: %d rounds  lr=%.3f  final_MSE=%.6f",
            self.n_estimators, self.learning_rate, losses[-1] if losses else 0.0,
        )
        return losses

    def _regularise_tree(self, node: Optional[_TreeNode], l2: float) -> None:
        """Recursively scale leaf values by 1/(1 + lambda/n)."""
        if node is None:
            return
        if node.value is not None:
            # Approximate regularisation: shrink toward zero
            node.value = node.value / (1.0 + l2)
        else:
            self._regularise_tree(node.left, l2)
            self._regularise_tree(node.right, l2)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions by summing base-learner contributions."""
        if not self._trees:
            raise RuntimeError("Model has not been trained.")
        preds = np.full(X.shape[0], self._init_pred)
        for tree in self._trees:
            preds += self.learning_rate * tree.predict(X)
        return preds

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #
    def get_feature_importance(self) -> Dict[int, float]:
        """Average split-frequency importance across all weak learners."""
        combined: Dict[int, float] = {}
        for tree in self._trees:
            for k, v in tree.get_feature_importance().items():
                combined[k] = combined.get(k, 0.0) + v
        n = len(self._trees) or 1
        return {k: v / n for k, v in sorted(combined.items())}

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #
    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, k_folds: int = 5,
    ) -> Dict[str, float]:
        """K-fold cross-validation returning mean and std MSE."""
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k_folds
        scores: List[float] = []

        for k in range(k_folds):
            val_idx = indices[k * fold_size: (k + 1) * fold_size]
            train_idx = np.concatenate([indices[: k * fold_size], indices[(k + 1) * fold_size:]])
            self.train(X[train_idx], y[train_idx])
            preds = self.predict(X[val_idx])
            scores.append(float(np.mean((preds - y[val_idx]) ** 2)))

        result = {
            "mean_mse": float(np.mean(scores)),
            "std_mse": float(np.std(scores)),
        }
        logger.info("GBT CV (%d-fold): %s", k_folds, result)
        return result
