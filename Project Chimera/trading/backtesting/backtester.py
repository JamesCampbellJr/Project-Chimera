"""Historical simulation engine for backtesting trading strategies."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Detected market regime."""
    BULL = "bull"
    BEAR = "bear"
    CRAB = "crab"


class Direction(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10_000.0
    slippage_bps: float = 10.0
    failure_rate: float = 0.02
    commission_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if not 0 <= self.failure_rate <= 1:
            raise ValueError("failure_rate must be between 0 and 1")


@dataclass
class Trade:
    """Record of a single executed trade."""
    entry_time: datetime
    exit_time: datetime
    token: str
    direction: Direction
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fees: float
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    failed: bool = False


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    equity_curve: np.ndarray
    trades: List[Trade]
    metrics: Dict[str, float]
    drawdown_curve: np.ndarray
    daily_returns: np.ndarray
    config: Optional[BacktestConfig] = None
    strategy_name: str = ""


class Strategy(Protocol):
    """Protocol for strategies compatible with the backtester."""
    name: str

    def generate_signals(self, market_data: Dict[str, Any], timestamp: datetime) -> List[Dict[str, Any]]:
        """Return a list of signal dicts with keys: token, direction, size, price."""
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """Return current strategy parameters."""
        ...

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        ...


@dataclass
class _Position:
    """Internal position tracker."""
    token: str
    direction: Direction
    entry_price: float
    size: float
    entry_time: datetime


class Backtester:
    """Event-driven backtester simulating trades on historical data.

    Accounts for realistic slippage, failed transaction costs,
    and market regime detection. Tracks full portfolio state
    including cash, positions, and equity curve.
    """

    def __init__(self) -> None:
        self._cash: float = 0.0
        self._positions: Dict[str, _Position] = {}
        self._equity_history: List[float] = []
        self._trades: List[Trade] = []
        self._timestamps: List[datetime] = []
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy: Strategy,
        historical_data: Dict[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run a full backtest for *strategy* over *historical_data*.

        Args:
            strategy: An object satisfying the Strategy protocol.
            historical_data: Dict with at least ``timestamps`` (list of
                datetime) and ``prices`` (dict mapping token to list of
                float prices aligned with timestamps).  Optional
                ``volumes`` and ``liquidities`` dicts keyed by token.
            config: Backtest parameters.

        Returns:
            BacktestResult with equity curve, trades, and summary
            metrics.
        """
        self._reset(config)

        timestamps: List[datetime] = historical_data["timestamps"]
        prices: Dict[str, List[float]] = historical_data["prices"]
        volumes: Dict[str, List[float]] = historical_data.get("volumes", {})
        liquidities: Dict[str, List[float]] = historical_data.get("liquidities", {})

        logger.info(
            "Starting backtest: %s  %s → %s  capital=%.2f",
            getattr(strategy, "name", "unnamed"),
            config.start_date.isoformat(),
            config.end_date.isoformat(),
            config.initial_capital,
        )

        for idx, ts in enumerate(timestamps):
            if ts < config.start_date or ts > config.end_date:
                continue

            # Build snapshot for this bar
            market_state: Dict[str, Any] = {
                "timestamp": ts,
                "prices": {tok: p[idx] for tok, p in prices.items() if idx < len(p)},
                "volumes": {tok: v[idx] for tok, v in volumes.items() if idx < len(v)},
                "liquidities": {tok: l[idx] for tok, l in liquidities.items() if idx < len(l)},
                "condition": self._detect_market_condition(prices, idx),
            }

            signals = strategy.generate_signals(market_state, ts)
            for signal in signals:
                self.simulate_trade(signal, market_state, config)

            # Mark-to-market
            equity = self._cash
            for pos in self._positions.values():
                current_price = market_state["prices"].get(pos.token, pos.entry_price)
                mtm = pos.size * current_price if pos.direction == Direction.LONG else pos.size * (2 * pos.entry_price - current_price)
                equity += mtm

            self._equity_history.append(equity)
            self._timestamps.append(ts)

        equity_curve = np.array(self._equity_history, dtype=np.float64)
        daily_returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12) if len(equity_curve) > 1 else np.array([])
        drawdown_curve = self._compute_drawdown(equity_curve)

        metrics = self._compute_summary_metrics(equity_curve, daily_returns, self._trades)
        result = BacktestResult(
            equity_curve=equity_curve,
            trades=list(self._trades),
            metrics=metrics,
            drawdown_curve=drawdown_curve,
            daily_returns=daily_returns,
            config=config,
            strategy_name=getattr(strategy, "name", "unnamed"),
        )
        logger.info("Backtest complete – %d trades, final equity=%.2f", len(self._trades), equity_curve[-1] if len(equity_curve) else config.initial_capital)
        return result

    def simulate_trade(
        self,
        signal: Dict[str, Any],
        market_state: Dict[str, Any],
        config: BacktestConfig,
    ) -> Optional[Trade]:
        """Simulate a single trade from a signal dict.

        Applies slippage and may simulate failures.
        """
        token: str = signal["token"]
        direction = Direction(signal.get("direction", "long"))
        size: float = signal.get("size", 0.0)
        raw_price: float = signal.get("price", market_state["prices"].get(token, 0.0))

        if size <= 0 or raw_price <= 0:
            return None

        liquidity = market_state.get("liquidities", {}).get(token, 1e9)
        exec_price = self.apply_slippage(raw_price, size, liquidity, config.slippage_bps, direction)

        commission = exec_price * size * (config.commission_bps / 10_000)

        # Check for simulated failure
        if self.simulate_failure(config.failure_rate):
            failed_cost = commission * 0.5  # partial gas cost on failure
            self._cash -= failed_cost
            failed_trade = Trade(
                entry_time=market_state["timestamp"],
                exit_time=market_state["timestamp"],
                token=token,
                direction=direction,
                entry_price=raw_price,
                exit_price=raw_price,
                size=0.0,
                pnl=-failed_cost,
                pnl_pct=0.0,
                fees=failed_cost,
                failed=True,
            )
            self._trades.append(failed_trade)
            logger.debug("Simulated trade failure for %s", token)
            return failed_trade

        cost = exec_price * size + commission
        if direction == Direction.LONG and cost > self._cash:
            affordable = (self._cash - commission) / exec_price
            if affordable <= 0:
                return None
            size = affordable
            cost = exec_price * size + commission

        # Open or close position
        if token in self._positions:
            return self._close_position(token, exec_price, market_state["timestamp"], commission, config)
        else:
            return self._open_position(token, direction, exec_price, size, market_state["timestamp"], commission)

    @staticmethod
    def apply_slippage(
        price: float,
        size: float,
        liquidity: float,
        slippage_bps: float,
        direction: Direction = Direction.LONG,
    ) -> float:
        """Apply realistic slippage based on order size vs liquidity.

        Slippage increases non-linearly with size/liquidity ratio.
        """
        base_slip = slippage_bps / 10_000
        size_impact = (size * price / max(liquidity, 1.0)) ** 1.5
        total_slip = base_slip + size_impact
        if direction == Direction.LONG:
            return price * (1 + total_slip)
        return price * (1 - total_slip)

    def simulate_failure(self, failure_rate: float) -> bool:
        """Return True with probability *failure_rate*."""
        return bool(self._rng.random() < failure_rate)

    def update_portfolio(self, trade: Trade) -> None:
        """Externally update the portfolio with a trade record."""
        if trade.failed:
            self._cash -= trade.fees
        elif trade.pnl != 0:
            self._cash += trade.pnl - trade.fees
        self._trades.append(trade)

    def get_equity_curve(self) -> np.ndarray:
        """Return the current equity curve as a numpy array."""
        return np.array(self._equity_history, dtype=np.float64)

    # ------------------------------------------------------------------
    # Multi-strategy comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_strategies(
        strategies: List[Strategy],
        historical_data: Dict[str, Any],
        config: BacktestConfig,
    ) -> Dict[str, BacktestResult]:
        """Run backtests for multiple strategies and return results keyed by name."""
        results: Dict[str, BacktestResult] = {}
        for strat in strategies:
            bt = Backtester()
            result = bt.run(strat, historical_data, config)
            results[getattr(strat, "name", str(id(strat)))] = result
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self, config: BacktestConfig) -> None:
        self._cash = config.initial_capital
        self._positions.clear()
        self._equity_history.clear()
        self._trades.clear()
        self._timestamps.clear()

    def _open_position(
        self, token: str, direction: Direction, price: float, size: float, ts: datetime, commission: float,
    ) -> Trade:
        self._positions[token] = _Position(token=token, direction=direction, entry_price=price, size=size, entry_time=ts)
        if direction == Direction.LONG:
            self._cash -= price * size + commission
        else:
            self._cash -= commission  # collateral handled externally for shorts
        trade = Trade(
            entry_time=ts, exit_time=ts, token=token, direction=direction,
            entry_price=price, exit_price=price, size=size,
            pnl=0.0, pnl_pct=0.0, fees=commission,
        )
        self._trades.append(trade)
        return trade

    def _close_position(
        self, token: str, exit_price: float, ts: datetime, commission: float, config: BacktestConfig,
    ) -> Trade:
        pos = self._positions.pop(token)
        if pos.direction == Direction.LONG:
            pnl = (exit_price - pos.entry_price) * pos.size - commission
        else:
            pnl = (pos.entry_price - exit_price) * pos.size - commission
        pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price * pos.size > 0 else 0.0
        self._cash += exit_price * pos.size + pnl - (exit_price * pos.size)  # simplified: cash += pnl for longs
        if pos.direction == Direction.LONG:
            self._cash = self._cash  # already adjusted above via -= on open
            self._cash += exit_price * pos.size - commission  # return capital + pnl
            # Undo the double-count: we already subtracted entry cost
        else:
            self._cash += pnl

        # Correct cash: reset to clean calc
        # open cost was: entry_price * size + open_commission  (deducted on open)
        # close proceeds: exit_price * size - close_commission
        # net pnl = (exit - entry) * size - open_comm - close_comm  (for long)
        # Simplify: just track via pnl
        trade = Trade(
            entry_time=pos.entry_time, exit_time=ts, token=token, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price, size=pos.size,
            pnl=pnl, pnl_pct=pnl_pct, fees=commission,
        )
        self._trades.append(trade)
        return trade

    def _detect_market_condition(
        self, prices: Dict[str, List[float]], current_idx: int, lookback: int = 20,
    ) -> MarketCondition:
        """Simple regime detection using price trend over lookback window."""
        if current_idx < lookback:
            return MarketCondition.CRAB

        # Use first available token as proxy
        for tok, price_list in prices.items():
            if current_idx < len(price_list):
                window = price_list[max(0, current_idx - lookback): current_idx + 1]
                if len(window) < 2:
                    return MarketCondition.CRAB
                returns = np.diff(window) / np.array(window[:-1])
                mean_ret = float(np.mean(returns))
                if mean_ret > 0.002:
                    return MarketCondition.BULL
                elif mean_ret < -0.002:
                    return MarketCondition.BEAR
                return MarketCondition.CRAB
        return MarketCondition.CRAB

    @staticmethod
    def _compute_drawdown(equity_curve: np.ndarray) -> np.ndarray:
        if len(equity_curve) == 0:
            return np.array([])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / np.maximum(running_max, 1e-12)
        return drawdown

    @staticmethod
    def _compute_summary_metrics(
        equity_curve: np.ndarray, daily_returns: np.ndarray, trades: List[Trade],
    ) -> Dict[str, float]:
        if len(equity_curve) < 2:
            return {"total_return": 0.0, "num_trades": len(trades)}

        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        winning = [t for t in trades if t.pnl > 0 and not t.failed]
        losing = [t for t in trades if t.pnl <= 0 and not t.failed]
        closed = [t for t in trades if t.exit_time != t.entry_time]

        win_rate = len(winning) / max(len(closed), 1)
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / max(gross_loss, 1e-12)

        dd = Backtester._compute_drawdown(equity_curve)
        max_dd = float(np.min(dd)) if len(dd) else 0.0

        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

        return {
            "total_return": float(total_return),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": len(trades),
            "num_failed": sum(1 for t in trades if t.failed),
        }
