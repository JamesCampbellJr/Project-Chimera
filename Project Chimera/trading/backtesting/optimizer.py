"""Walk-forward optimization and hyperparameter search."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .backtester import BacktestConfig, BacktestResult, Backtester, Strategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single train/test split in walk-forward optimization."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class OptimizationResult:
    """Results from a walk-forward or grid/random search optimization."""
    best_params: Dict[str, Any]
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    stability_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)


class WalkForwardOptimizer:
    """Walk-forward optimizer with grid/random search and overfitting detection.

    Splits data into rolling train/test windows, optimizes parameters
    on the training portion, and validates on the subsequent test portion.
    Aggregated out-of-sample performance provides a realistic estimate of
    live performance.
    """

    def __init__(self, metric: str = "sharpe_ratio") -> None:
        """
        Args:
            metric: The metric name (from BacktestResult.metrics) to
                optimize. Higher is better.
        """
        self._metric = metric
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def walk_forward_optimize(
        self,
        strategy: Strategy,
        data: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        train_ratio: float = 0.7,
        num_windows: int = 5,
    ) -> OptimizationResult:
        """Run walk-forward optimization.

        The timeline is divided into *num_windows* overlapping
        train/test splits.  For each window the best parameters are
        found via grid search on the training set, then evaluated on
        the test set.

        Args:
            strategy: Strategy instance (must support set_parameters).
            data: Historical data dict (see Backtester.run).
            param_grid: Mapping of parameter names to lists of values.
            train_ratio: Fraction of each window used for training.
            num_windows: Number of walk-forward windows.

        Returns:
            OptimizationResult aggregated across all windows.
        """
        timestamps: List[datetime] = data["timestamps"]
        if len(timestamps) < 2:
            raise ValueError("Insufficient data for walk-forward optimization")

        windows = self._create_windows(timestamps, train_ratio, num_windows)
        logger.info("Walk-forward: %d windows, %d param combos", len(windows), self._grid_size(param_grid))

        all_window_results: List[Dict[str, Any]] = []
        oos_metrics_list: List[Dict[str, float]] = []
        best_params_per_window: List[Dict[str, Any]] = []

        for i, window in enumerate(windows):
            logger.info("Window %d/%d: train %s→%s  test %s→%s", i + 1, len(windows),
                        window.train_start.isoformat(), window.train_end.isoformat(),
                        window.test_start.isoformat(), window.test_end.isoformat())

            train_config = BacktestConfig(
                start_date=window.train_start, end_date=window.train_end,
                initial_capital=10_000,
            )
            best_params, best_is_metric = self._search_best(strategy, data, param_grid, train_config)
            best_params_per_window.append(best_params)

            # Out-of-sample evaluation
            strategy.set_parameters(best_params)
            test_config = BacktestConfig(
                start_date=window.test_start, end_date=window.test_end,
                initial_capital=10_000,
            )
            bt = Backtester()
            oos_result = bt.run(strategy, data, test_config)
            oos_metrics = oos_result.metrics

            oos_metrics_list.append(oos_metrics)
            all_window_results.append({
                "window": i,
                "best_params": best_params,
                "in_sample_metric": best_is_metric,
                "out_of_sample_metrics": oos_metrics,
            })

        # Aggregate
        agg_is = float(np.mean([r["in_sample_metric"] for r in all_window_results]))
        agg_oos: Dict[str, float] = {}
        for key in oos_metrics_list[0]:
            agg_oos[key] = float(np.mean([m.get(key, 0.0) for m in oos_metrics_list]))

        stability = self.analyze_stability(best_params_per_window)
        overfitting = self.detect_overfitting(agg_is, agg_oos.get(self._metric, 0.0))

        logger.info("Walk-forward complete – IS=%.4f  OOS=%.4f  stability=%.4f  overfit=%s",
                     agg_is, agg_oos.get(self._metric, 0.0), stability, overfitting)

        # Pick the most frequently chosen params
        best_overall = self._most_common_params(best_params_per_window)

        return OptimizationResult(
            best_params=best_overall,
            in_sample_metrics={self._metric: agg_is},
            out_of_sample_metrics=agg_oos,
            stability_score=stability,
            all_results=all_window_results,
        )

    def grid_search(
        self,
        strategy: Strategy,
        data: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        config: Optional[BacktestConfig] = None,
    ) -> OptimizationResult:
        """Exhaustive grid search over all parameter combinations.

        Args:
            strategy: Strategy instance.
            data: Historical data dict.
            param_grid: Parameter name → list of values.
            config: Backtest configuration. Uses full data range if None.

        Returns:
            OptimizationResult with the best parameter set.
        """
        if config is None:
            timestamps = data["timestamps"]
            config = BacktestConfig(start_date=timestamps[0], end_date=timestamps[-1])

        combos = self._param_combinations(param_grid)
        logger.info("Grid search: evaluating %d combinations", len(combos))

        results: List[Dict[str, Any]] = []
        best_metric = -np.inf
        best_params: Dict[str, Any] = {}

        for params in combos:
            strategy.set_parameters(params)
            bt = Backtester()
            res = bt.run(strategy, data, config)
            metric_val = res.metrics.get(self._metric, -np.inf)
            results.append({"params": params, "metric": metric_val, "metrics": res.metrics})
            if metric_val > best_metric:
                best_metric = metric_val
                best_params = params

        return OptimizationResult(
            best_params=best_params,
            in_sample_metrics={self._metric: float(best_metric)},
            out_of_sample_metrics={},
            stability_score=self.analyze_stability([r["params"] for r in results[:5]]),
            all_results=results,
        )

    def random_search(
        self,
        strategy: Strategy,
        data: Dict[str, Any],
        param_ranges: Dict[str, Tuple[float, float]],
        n_iter: int = 50,
        config: Optional[BacktestConfig] = None,
    ) -> OptimizationResult:
        """Random search sampling from continuous parameter ranges.

        Args:
            strategy: Strategy instance.
            data: Historical data dict.
            param_ranges: Parameter name → (low, high) bounds.
            n_iter: Number of random samples.
            config: Backtest configuration.

        Returns:
            OptimizationResult with the best parameter set.
        """
        if config is None:
            timestamps = data["timestamps"]
            config = BacktestConfig(start_date=timestamps[0], end_date=timestamps[-1])

        logger.info("Random search: %d iterations", n_iter)

        results: List[Dict[str, Any]] = []
        best_metric = -np.inf
        best_params: Dict[str, Any] = {}

        for _ in range(n_iter):
            params = {
                name: float(self._rng.uniform(lo, hi))
                for name, (lo, hi) in param_ranges.items()
            }
            strategy.set_parameters(params)
            bt = Backtester()
            res = bt.run(strategy, data, config)
            metric_val = res.metrics.get(self._metric, -np.inf)
            results.append({"params": params, "metric": metric_val, "metrics": res.metrics})
            if metric_val > best_metric:
                best_metric = metric_val
                best_params = params

        return OptimizationResult(
            best_params=best_params,
            in_sample_metrics={self._metric: float(best_metric)},
            out_of_sample_metrics={},
            stability_score=self.analyze_stability([r["params"] for r in results[:10]]),
            all_results=results,
        )

    @staticmethod
    def detect_overfitting(
        in_sample: float,
        out_of_sample: float,
        threshold: float = 0.5,
    ) -> bool:
        """Detect overfitting by comparing in-sample vs out-of-sample performance.

        If out-of-sample metric is less than *threshold* fraction of
        in-sample metric, overfitting is likely.

        Args:
            in_sample: Aggregated in-sample metric value.
            out_of_sample: Aggregated out-of-sample metric value.
            threshold: Ratio below which overfitting is flagged.

        Returns:
            True if overfitting is detected.
        """
        if abs(in_sample) < 1e-12:
            return False
        ratio = out_of_sample / in_sample
        is_overfit = ratio < threshold
        if is_overfit:
            logger.warning(
                "Overfitting detected: OOS/IS ratio=%.4f (threshold=%.2f)",
                ratio, threshold,
            )
        return is_overfit

    @staticmethod
    def analyze_stability(param_results: List[Dict[str, Any]]) -> float:
        """Analyse how stable chosen parameters are across windows.

        Returns a score in [0, 1] where 1 means perfectly stable
        (same params every window).
        """
        if len(param_results) < 2:
            return 1.0

        # Compute coefficient of variation per numeric parameter
        param_names = set()
        for pr in param_results:
            param_names.update(pr.keys())

        cvs: List[float] = []
        for name in param_names:
            vals = []
            for pr in param_results:
                v = pr.get(name)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if len(vals) >= 2:
                std = float(np.std(vals))
                mean = float(np.mean(vals))
                cv = std / max(abs(mean), 1e-12)
                cvs.append(cv)

        if not cvs:
            # All categorical — check equality
            first = param_results[0]
            same = sum(1 for pr in param_results[1:] if pr == first)
            return (same + 1) / len(param_results)

        avg_cv = float(np.mean(cvs))
        # Map CV to a 0-1 stability score (lower CV = higher stability)
        stability = float(np.exp(-avg_cv))
        return max(0.0, min(1.0, stability))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_windows(
        timestamps: List[datetime],
        train_ratio: float,
        num_windows: int,
    ) -> List[WalkForwardWindow]:
        n = len(timestamps)
        window_size = n // num_windows
        if window_size < 4:
            raise ValueError("Not enough data points for the requested number of windows")

        train_size = max(2, int(window_size * train_ratio))
        test_size = max(1, window_size - train_size)

        windows: List[WalkForwardWindow] = []
        for i in range(num_windows):
            start = i * test_size
            train_end_idx = min(start + train_size - 1, n - 1)
            test_start_idx = min(train_end_idx + 1, n - 1)
            test_end_idx = min(test_start_idx + test_size - 1, n - 1)
            if test_start_idx >= n or train_end_idx >= n:
                break
            windows.append(WalkForwardWindow(
                train_start=timestamps[start],
                train_end=timestamps[train_end_idx],
                test_start=timestamps[test_start_idx],
                test_end=timestamps[test_end_idx],
            ))
        return windows

    def _search_best(
        self,
        strategy: Strategy,
        data: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        config: BacktestConfig,
    ) -> Tuple[Dict[str, Any], float]:
        combos = self._param_combinations(param_grid)
        best_metric = -np.inf
        best_params: Dict[str, Any] = {}
        for params in combos:
            strategy.set_parameters(params)
            bt = Backtester()
            res = bt.run(strategy, data, config)
            val = res.metrics.get(self._metric, -np.inf)
            if val > best_metric:
                best_metric = val
                best_params = params
        return best_params, float(best_metric)

    @staticmethod
    def _param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = sorted(param_grid.keys())
        values = [param_grid[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @staticmethod
    def _grid_size(param_grid: Dict[str, List[Any]]) -> int:
        size = 1
        for v in param_grid.values():
            size *= len(v)
        return size

    @staticmethod
    def _most_common_params(params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not params_list:
            return {}
        # For numeric params take median; for others take mode
        keys = set()
        for p in params_list:
            keys.update(p.keys())

        result: Dict[str, Any] = {}
        for key in keys:
            vals = [p[key] for p in params_list if key in p]
            if all(isinstance(v, (int, float)) for v in vals):
                result[key] = float(np.median(vals))
            else:
                # mode
                from collections import Counter
                result[key] = Counter(vals).most_common(1)[0][0]
        return result
