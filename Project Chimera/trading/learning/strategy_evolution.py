"""Strategy evolution using genetic algorithms and market-regime detection.

Provides genetic-algorithm–based parameter optimisation with crossover and
mutation operators, fitness evaluation based on Sharpe ratio and drawdown,
market regime detection (trending / ranging / volatile), and automated
strategy rotation based on detected regime.
"""

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class RegimeType(Enum):
    """Market regime categories."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class StrategyGene:
    """A strategy genome encoding its parameters and fitness."""
    params: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    lineage: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class MarketRegime:
    """Detected market regime with confidence score."""
    regime_type: RegimeType = RegimeType.UNKNOWN
    confidence: float = 0.0
    start_time: float = field(default_factory=time.time)
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Performance metrics for a named strategy over a period."""
    strategy_name: str = ""
    period: str = ""
    sharpe: float = 0.0
    drawdown: float = 0.0
    win_rate: float = 0.0
    regime: RegimeType = RegimeType.UNKNOWN
    is_decaying: bool = False


# ---------------------------------------------------------------------------
# Strategy evolver
# ---------------------------------------------------------------------------

class StrategyEvolver:
    """Evolves trading-strategy parameters via genetic algorithms and
    performs market-regime detection for automated strategy rotation.

    Args:
        fitness_fn: Optional custom fitness function ``(returns) -> float``.
            Defaults to a Sharpe-ratio & drawdown composite.
        mutation_rate: Default probability of mutating each gene.
        crossover_rate: Probability of performing crossover vs cloning.
        elitism_ratio: Fraction of top individuals preserved unchanged.
    """

    def __init__(
        self,
        fitness_fn: Optional[Callable[[np.ndarray], float]] = None,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism_ratio: float = 0.1,
    ) -> None:
        self._fitness_fn = fitness_fn or self._default_fitness
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elitism_ratio = elitism_ratio
        self._population: List[StrategyGene] = []
        self._generation = 0
        self._performance_history: Dict[str, List[StrategyPerformance]] = {}
        self._regime_history: List[MarketRegime] = []
        logger.info(
            "StrategyEvolver initialised (mutation=%.2f crossover=%.2f elitism=%.2f)",
            mutation_rate, crossover_rate, elitism_ratio,
        )

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def initialize_population(
        self,
        base_params: Dict[str, Tuple[float, float]],
        pop_size: int = 50,
    ) -> List[StrategyGene]:
        """Create an initial random population around *base_params*.

        Args:
            base_params: Mapping of parameter name → ``(min, max)`` range.
            pop_size: Number of individuals.

        Returns:
            The generated population list.
        """
        self._population = []
        for _ in range(pop_size):
            params = {
                k: float(np.random.uniform(lo, hi))
                for k, (lo, hi) in base_params.items()
            }
            gene = StrategyGene(params=params, generation=0, lineage=["genesis"])
            self._population.append(gene)

        self._generation = 0
        logger.info("Population initialised: %d individuals, %d params",
                     pop_size, len(base_params))
        return list(self._population)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_fitness(
        self,
        strategy: StrategyGene,
        data: np.ndarray,
    ) -> float:
        """Evaluate the fitness of *strategy* on historical return *data*.

        Args:
            strategy: The strategy gene to evaluate.
            data: 1-D array of period returns.

        Returns:
            Scalar fitness score (higher is better).
        """
        try:
            score = float(self._fitness_fn(data))
            strategy.fitness = score
            return score
        except Exception as exc:
            logger.error("Fitness evaluation failed: %s", exc)
            strategy.fitness = -np.inf
            return -np.inf

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def crossover(self, parent_a: StrategyGene, parent_b: StrategyGene) -> StrategyGene:
        """Uniform crossover of two parents.

        Each parameter is inherited from one parent with equal probability.

        Args:
            parent_a: First parent.
            parent_b: Second parent.

        Returns:
            A new child ``StrategyGene``.
        """
        child_params: Dict[str, float] = {}
        all_keys = set(parent_a.params) | set(parent_b.params)
        for key in all_keys:
            if np.random.random() < 0.5:
                child_params[key] = parent_a.params.get(key, parent_b.params.get(key, 0.0))
            else:
                child_params[key] = parent_b.params.get(key, parent_a.params.get(key, 0.0))

        child = StrategyGene(
            params=child_params,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            lineage=[parent_a.id, parent_b.id],
        )
        return child

    def mutate(self, gene: StrategyGene, mutation_rate: Optional[float] = None) -> StrategyGene:
        """Mutate a strategy gene in-place.

        Each parameter is perturbed with probability *mutation_rate* using
        Gaussian noise proportional to its absolute value.

        Args:
            gene: The gene to mutate.
            mutation_rate: Override the default mutation rate.

        Returns:
            The (possibly mutated) gene.
        """
        rate = mutation_rate if mutation_rate is not None else self._mutation_rate
        for key in gene.params:
            if np.random.random() < rate:
                scale = abs(gene.params[key]) * 0.1 + 1e-8
                gene.params[key] += float(np.random.normal(0, scale))
        return gene

    # ------------------------------------------------------------------
    # Evolution loop
    # ------------------------------------------------------------------

    def evolve_generation(
        self,
        population: List[StrategyGene],
        data: np.ndarray,
    ) -> List[StrategyGene]:
        """Evolve the population by one generation.

        Steps: evaluate fitness → elitism selection → tournament selection →
        crossover → mutation.

        Args:
            population: Current generation.
            data: 1-D return array used for fitness evaluation.

        Returns:
            The next generation of ``StrategyGene`` instances.
        """
        # Evaluate fitness
        for gene in population:
            self.evaluate_fitness(gene, data)

        # Sort descending by fitness
        ranked = sorted(population, key=lambda g: g.fitness, reverse=True)
        pop_size = len(ranked)

        # Elitism
        n_elite = max(1, int(pop_size * self._elitism_ratio))
        next_gen: List[StrategyGene] = [copy.deepcopy(g) for g in ranked[:n_elite]]

        # Fill the rest via tournament selection + crossover + mutation
        while len(next_gen) < pop_size:
            if np.random.random() < self._crossover_rate:
                p_a = self._tournament_select(ranked)
                p_b = self._tournament_select(ranked)
                child = self.crossover(p_a, p_b)
            else:
                child = copy.deepcopy(self._tournament_select(ranked))
            child = self.mutate(child)
            child.generation = self._generation + 1
            next_gen.append(child)

        self._generation += 1
        self._population = next_gen
        best = max(next_gen, key=lambda g: g.fitness)
        logger.info("Generation %d evolved: best_fitness=%.4f", self._generation, best.fitness)
        return next_gen

    # ------------------------------------------------------------------
    # Market regime detection
    # ------------------------------------------------------------------

    def detect_market_regime(self, market_data: np.ndarray) -> MarketRegime:
        """Detect the current market regime from price / return data.

        Uses volatility, trend strength (linear regression slope), and
        mean-reversion statistics to classify the regime.

        Args:
            market_data: 1-D array of prices or returns.

        Returns:
            A ``MarketRegime`` describing the detected regime.
        """
        data = np.array(market_data, dtype=float).ravel()
        if len(data) < 5:
            return MarketRegime(regime_type=RegimeType.UNKNOWN, confidence=0.0)

        returns = np.diff(data) / (np.abs(data[:-1]) + 1e-10)
        volatility = float(np.std(returns))

        # Trend strength via OLS slope
        x = np.arange(len(data), dtype=float)
        slope, _ = np.polyfit(x, data, 1)
        normalised_slope = abs(slope) / (np.std(data) + 1e-10)

        # Mean-reversion: Hurst-like measure
        mean_rev = float(np.mean(np.abs(returns)))

        features = {
            "volatility": volatility,
            "trend_strength": float(normalised_slope),
            "mean_reversion": mean_rev,
        }

        # Classification
        vol_threshold = 0.03
        trend_threshold = 0.05

        if volatility > vol_threshold:
            regime = RegimeType.VOLATILE
            confidence = min(1.0, volatility / vol_threshold)
        elif normalised_slope > trend_threshold:
            regime = RegimeType.TRENDING
            confidence = min(1.0, normalised_slope / trend_threshold)
        else:
            regime = RegimeType.RANGING
            confidence = 1.0 - normalised_slope / (trend_threshold + 1e-10)

        result = MarketRegime(
            regime_type=regime,
            confidence=float(np.clip(confidence, 0, 1)),
            features=features,
        )
        self._regime_history.append(result)
        logger.info("Market regime detected: %s (confidence=%.2f)", regime.value, result.confidence)
        return result

    # ------------------------------------------------------------------
    # Strategy selection & performance
    # ------------------------------------------------------------------

    def select_strategy_for_regime(
        self,
        regime: MarketRegime,
        available_strategies: Dict[str, StrategyGene],
    ) -> Optional[str]:
        """Select the best strategy for the given market *regime*.

        Chooses the strategy with the highest historical fitness during the
        same regime type.  Falls back to overall best if no history exists.

        Args:
            regime: The current ``MarketRegime``.
            available_strategies: Mapping of strategy name → gene.

        Returns:
            Name of the selected strategy, or ``None`` if no strategies available.
        """
        if not available_strategies:
            return None

        best_name: Optional[str] = None
        best_score = -np.inf

        for name, gene in available_strategies.items():
            perf_list = self._performance_history.get(name, [])
            regime_perfs = [p for p in perf_list if p.regime == regime.regime_type]

            if regime_perfs:
                score = np.mean([p.sharpe for p in regime_perfs])
            else:
                score = gene.fitness

            if score > best_score:
                best_score = score
                best_name = name

        logger.info("Selected strategy '%s' for regime %s (score=%.4f)",
                     best_name, regime.regime_type.value, best_score)
        return best_name

    def track_performance_decay(
        self,
        strategy: str,
        window: int = 20,
    ) -> StrategyPerformance:
        """Check whether *strategy*'s performance is decaying.

        Compares the most recent *window* periods to the preceding *window*.

        Args:
            strategy: Strategy name.
            window: Number of recent periods to compare.

        Returns:
            A ``StrategyPerformance`` snapshot with ``is_decaying`` set.
        """
        history = self._performance_history.get(strategy, [])

        if len(history) < window * 2:
            recent_sharpe = np.mean([p.sharpe for p in history[-window:]]) if history else 0.0
            return StrategyPerformance(
                strategy_name=strategy,
                period=f"last_{window}",
                sharpe=float(recent_sharpe),
                is_decaying=False,
            )

        recent = history[-window:]
        previous = history[-2 * window:-window]

        recent_sharpe = float(np.mean([p.sharpe for p in recent]))
        prev_sharpe = float(np.mean([p.sharpe for p in previous]))
        recent_dd = float(np.mean([p.drawdown for p in recent]))
        recent_wr = float(np.mean([p.win_rate for p in recent]))

        # Decay if recent Sharpe dropped by >20 % relative to previous
        decay_threshold = 0.2
        is_decaying = (prev_sharpe - recent_sharpe) / (abs(prev_sharpe) + 1e-10) > decay_threshold

        current_regime = self._regime_history[-1].regime_type if self._regime_history else RegimeType.UNKNOWN

        perf = StrategyPerformance(
            strategy_name=strategy,
            period=f"last_{window}",
            sharpe=recent_sharpe,
            drawdown=recent_dd,
            win_rate=recent_wr,
            regime=current_regime,
            is_decaying=is_decaying,
        )

        if is_decaying:
            logger.warning("Performance decay detected for '%s': sharpe %.4f → %.4f",
                           strategy, prev_sharpe, recent_sharpe)
        return perf

    def add_performance_record(self, strategy: str, perf: StrategyPerformance) -> None:
        """Append a performance record for future decay analysis.

        Args:
            strategy: Strategy name.
            perf: The ``StrategyPerformance`` to record.
        """
        self._performance_history.setdefault(strategy, []).append(perf)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_fitness(returns: np.ndarray) -> float:
        """Composite fitness: Sharpe ratio penalised by max drawdown."""
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0

        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Penalise drawdown: fitness = sharpe - 0.5 * max_dd
        return sharpe - 0.5 * max_dd

    @staticmethod
    def _tournament_select(ranked: List[StrategyGene], k: int = 3) -> StrategyGene:
        """Tournament selection: pick the best of *k* random individuals."""
        contestants = [ranked[i] for i in np.random.randint(0, len(ranked), size=k)]
        return max(contestants, key=lambda g: g.fitness)
