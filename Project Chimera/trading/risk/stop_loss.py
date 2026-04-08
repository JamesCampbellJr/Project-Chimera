# trading/risk/stop_loss.py
"""
Stop Loss & Take Profit Engine — manages protective exits using ATR-based
stops, trailing mechanisms, time-based exits, and circuit breakers.

Implements the Chandelier Exit, ratcheting trailing stops, and flash-crash
detection to protect open positions under all market conditions.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Defaults                                                               #
# --------------------------------------------------------------------- #
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULTIPLIER = 2.0
DEFAULT_TRAIL_PCT = 0.05            # 5 % trailing stop
DEFAULT_MAX_HOLD_HOURS = 48.0       # mean-reversion time exit
CIRCUIT_BREAKER_DROP_PCT = 0.05     # 5 % drop in 1 minute
CIRCUIT_BREAKER_WINDOW_SEC = 60
CHANDELIER_MULTIPLIER = 3.0


# --------------------------------------------------------------------- #
# Data classes                                                           #
# --------------------------------------------------------------------- #
class StopType(str, Enum):
    ATR = "atr"
    TRAILING_PCT = "trailing_pct"
    TRAILING_ATR = "trailing_atr"
    TIME = "time"
    CIRCUIT_BREAKER = "circuit_breaker"
    CHANDELIER = "chandelier"
    TAKE_PROFIT = "take_profit"


class ExitUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StopLevel:
    """Represents a single stop-loss or take-profit level."""

    type: StopType
    price: float
    distance_pct: float
    triggered: bool = False

    def __post_init__(self) -> None:
        self.price = round(self.price, 8)
        self.distance_pct = round(self.distance_pct, 6)


@dataclass
class ExitSignal:
    """Emitted when a position should be closed."""

    reason: str
    urgency: ExitUrgency
    price_target: float

    def __post_init__(self) -> None:
        self.price_target = round(self.price_target, 8)


# --------------------------------------------------------------------- #
# Stop-Loss Manager                                                      #
# --------------------------------------------------------------------- #
class StopLossManager:
    """
    Evaluates multiple exit strategies for a position and returns the
    most urgent exit signal (if any).
    """

    def __init__(
        self,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        trail_pct: float = DEFAULT_TRAIL_PCT,
        max_hold_hours: float = DEFAULT_MAX_HOLD_HOURS,
        circuit_breaker_drop: float = CIRCUIT_BREAKER_DROP_PCT,
        circuit_breaker_window: int = CIRCUIT_BREAKER_WINDOW_SEC,
        chandelier_multiplier: float = CHANDELIER_MULTIPLIER,
        take_profit_pct: Optional[float] = 0.20,
    ) -> None:
        self.atr_multiplier = atr_multiplier
        self.trail_pct = trail_pct
        self.max_hold_hours = max_hold_hours
        self.circuit_breaker_drop = circuit_breaker_drop
        self.circuit_breaker_window = circuit_breaker_window
        self.chandelier_multiplier = chandelier_multiplier
        self.take_profit_pct = take_profit_pct

        # Trailing stop tracking: token → highest seen price
        self._trailing_highs: Dict[str, float] = {}

        logger.info(
            "StopLossManager ready  atr_mult=%.1f  trail=%.1f%%  "
            "max_hold=%.0fh  cb_drop=%.1f%%",
            self.atr_multiplier, self.trail_pct * 100,
            self.max_hold_hours, self.circuit_breaker_drop * 100,
        )

    # ------------------------------------------------------------------ #
    # ATR                                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_atr(prices: List[float], period: int = DEFAULT_ATR_PERIOD) -> float:
        """
        Average True Range from a list of close prices.

        For a close-only series the true range simplifies to
        abs(close[i] − close[i−1]).  Returns 0 if insufficient data.
        """
        if len(prices) < 2:
            return 0.0
        true_ranges: List[float] = []
        for i in range(1, len(prices)):
            true_ranges.append(abs(prices[i] - prices[i - 1]))
        if not true_ranges:
            return 0.0
        window = true_ranges[-period:]
        return sum(window) / len(window)

    @staticmethod
    def atr_stop(entry_price: float, atr: float, multiplier: float) -> StopLevel:
        """
        Fixed ATR-based stop placed *multiplier × ATR* below entry.
        """
        if atr <= 0 or entry_price <= 0:
            return StopLevel(StopType.ATR, 0.0, 0.0)
        stop_price = entry_price - multiplier * atr
        stop_price = max(stop_price, 0.0)
        distance = (entry_price - stop_price) / entry_price
        return StopLevel(
            type=StopType.ATR,
            price=stop_price,
            distance_pct=distance,
        )

    # ------------------------------------------------------------------ #
    # Trailing stops                                                       #
    # ------------------------------------------------------------------ #

    def trailing_stop_update(
        self,
        current_price: float,
        current_stop: float,
        trail_pct: Optional[float] = None,
    ) -> float:
        """
        Ratchet a percentage-based trailing stop upward.

        Returns the new stop level — never lower than the existing stop
        (the ratchet only moves up).
        """
        pct = trail_pct if trail_pct is not None else self.trail_pct
        if current_price <= 0:
            return current_stop
        new_stop = current_price * (1.0 - pct)
        return max(current_stop, new_stop)

    def _get_trailing_stop(
        self,
        token: str,
        current_price: float,
        trail_pct: Optional[float] = None,
    ) -> StopLevel:
        """
        Maintain a per-token trailing-stop high-water mark and return the
        current stop level.
        """
        pct = trail_pct if trail_pct is not None else self.trail_pct
        prev_high = self._trailing_highs.get(token, current_price)
        new_high = max(prev_high, current_price)
        self._trailing_highs[token] = new_high

        stop_price = new_high * (1.0 - pct)
        distance = (new_high - stop_price) / new_high if new_high > 0 else 0.0
        triggered = current_price <= stop_price
        return StopLevel(
            type=StopType.TRAILING_PCT,
            price=stop_price,
            distance_pct=distance,
            triggered=triggered,
        )

    # ------------------------------------------------------------------ #
    # Time-based exit                                                      #
    # ------------------------------------------------------------------ #

    def check_time_exit(
        self,
        entry_time: float,
        max_hold_hours: Optional[float] = None,
    ) -> Optional[ExitSignal]:
        """
        Force exit if the position has been held beyond *max_hold_hours*.

        Parameters
        ----------
        entry_time : float
            UNIX timestamp when the position was opened.
        max_hold_hours : float or None
            Override the instance default.
        """
        hours = max_hold_hours if max_hold_hours is not None else self.max_hold_hours
        elapsed_sec = time.time() - entry_time
        elapsed_hours = elapsed_sec / 3600.0
        if elapsed_hours >= hours:
            logger.info(
                "Time exit triggered: held %.1f h (limit %.1f h).",
                elapsed_hours, hours,
            )
            return ExitSignal(
                reason=f"Time exit after {elapsed_hours:.1f}h (limit {hours:.0f}h)",
                urgency=ExitUrgency.MEDIUM,
                price_target=0.0,  # market order
            )
        return None

    # ------------------------------------------------------------------ #
    # Circuit breaker (flash-crash detection)                              #
    # ------------------------------------------------------------------ #

    def check_circuit_breaker(
        self,
        recent_prices: List[Tuple[float, float]],
    ) -> Optional[ExitSignal]:
        """
        Detect a flash crash — a drop exceeding *circuit_breaker_drop*
        within *circuit_breaker_window* seconds.

        Parameters
        ----------
        recent_prices : list of (timestamp, price) tuples
            Must be sorted ascending by timestamp.

        Returns
        -------
        ExitSignal or None
        """
        if len(recent_prices) < 2:
            return None

        now_ts = recent_prices[-1][0]
        window_start = now_ts - self.circuit_breaker_window

        # Find the highest price in the window
        prices_in_window = [
            p for ts, p in recent_prices if ts >= window_start
        ]
        if len(prices_in_window) < 2:
            return None

        high = max(prices_in_window)
        current = prices_in_window[-1]

        if high <= 0:
            return None

        drop_pct = (high - current) / high

        if drop_pct >= self.circuit_breaker_drop:
            logger.warning(
                "CIRCUIT BREAKER: %.1f%% drop in %ds window (threshold %.1f%%).",
                drop_pct * 100, self.circuit_breaker_window,
                self.circuit_breaker_drop * 100,
            )
            return ExitSignal(
                reason=f"Flash crash: {drop_pct:.1%} drop in "
                       f"{self.circuit_breaker_window}s",
                urgency=ExitUrgency.CRITICAL,
                price_target=current,
            )
        return None

    # ------------------------------------------------------------------ #
    # Chandelier Exit                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def chandelier_exit(
        highs: List[float],
        atr: float,
        multiplier: float = CHANDELIER_MULTIPLIER,
    ) -> StopLevel:
        """
        Chandelier Exit: stop at highest-high minus multiplier × ATR.
        """
        if not highs or atr <= 0:
            return StopLevel(StopType.CHANDELIER, 0.0, 0.0)
        highest = max(highs)
        stop_price = highest - multiplier * atr
        stop_price = max(stop_price, 0.0)
        distance = (highest - stop_price) / highest if highest > 0 else 0.0
        return StopLevel(
            type=StopType.CHANDELIER,
            price=stop_price,
            distance_pct=distance,
        )

    # ------------------------------------------------------------------ #
    # Take-profit check                                                    #
    # ------------------------------------------------------------------ #

    def _check_take_profit(
        self,
        entry_price: float,
        current_price: float,
    ) -> Optional[ExitSignal]:
        """Return an exit signal if the take-profit threshold is reached."""
        if self.take_profit_pct is None or entry_price <= 0:
            return None
        gain_pct = (current_price - entry_price) / entry_price
        if gain_pct >= self.take_profit_pct:
            logger.info(
                "Take profit triggered: gain %.1f%% >= %.1f%%.",
                gain_pct * 100, self.take_profit_pct * 100,
            )
            return ExitSignal(
                reason=f"Take profit at {gain_pct:.1%} gain",
                urgency=ExitUrgency.LOW,
                price_target=current_price,
            )
        return None

    # ------------------------------------------------------------------ #
    # Master exit evaluator                                                #
    # ------------------------------------------------------------------ #

    def should_exit(
        self,
        position: Dict,
        current_price: float,
        current_time: Optional[float] = None,
    ) -> Optional[ExitSignal]:
        """
        Evaluate all exit strategies for a position and return the most
        urgent signal, or None if no exit is warranted.

        Parameters
        ----------
        position : dict
            Must contain:
              - token (str)
              - entry_price (float)
              - entry_time (float): UNIX timestamp
              - prices (List[float]): recent close prices for ATR
              - recent_ticks (List[Tuple[float, float]]): (ts, price)
                    for circuit-breaker evaluation
              - highs (List[float]): recent period highs (optional)
        current_price : float
        current_time : float or None
            UNIX timestamp; defaults to now.
        """
        if current_time is None:
            current_time = time.time()

        token: str = position["token"]
        entry_price: float = position["entry_price"]
        entry_time: float = position["entry_time"]
        prices: List[float] = position.get("prices", [])
        recent_ticks = position.get("recent_ticks", [])
        highs: List[float] = position.get("highs", prices[-20:] if prices else [])

        signals: List[ExitSignal] = []

        # 1. Circuit breaker (highest priority)
        cb_signal = self.check_circuit_breaker(recent_ticks)
        if cb_signal:
            signals.append(cb_signal)

        # 2. ATR stop
        if prices:
            atr = self.calculate_atr(prices)
            atr_level = self.atr_stop(entry_price, atr, self.atr_multiplier)
            if current_price <= atr_level.price and atr_level.price > 0:
                signals.append(ExitSignal(
                    reason=f"ATR stop hit at ${atr_level.price:.4f}",
                    urgency=ExitUrgency.HIGH,
                    price_target=atr_level.price,
                ))

        # 3. Trailing stop
        trail_level = self._get_trailing_stop(token, current_price)
        if trail_level.triggered:
            signals.append(ExitSignal(
                reason=f"Trailing stop hit at ${trail_level.price:.4f}",
                urgency=ExitUrgency.HIGH,
                price_target=trail_level.price,
            ))

        # 4. Chandelier exit
        if highs and prices:
            atr_val = self.calculate_atr(prices)
            ch_level = self.chandelier_exit(
                highs, atr_val, self.chandelier_multiplier
            )
            if ch_level.price > 0 and current_price <= ch_level.price:
                signals.append(ExitSignal(
                    reason=f"Chandelier exit at ${ch_level.price:.4f}",
                    urgency=ExitUrgency.MEDIUM,
                    price_target=ch_level.price,
                ))

        # 5. Time-based exit
        time_signal = self.check_time_exit(entry_time, self.max_hold_hours)
        if time_signal:
            signals.append(time_signal)

        # 6. Take profit
        tp_signal = self._check_take_profit(entry_price, current_price)
        if tp_signal:
            signals.append(tp_signal)

        if not signals:
            return None

        # Return the most urgent signal
        urgency_order = {
            ExitUrgency.CRITICAL: 0,
            ExitUrgency.HIGH: 1,
            ExitUrgency.MEDIUM: 2,
            ExitUrgency.LOW: 3,
        }
        signals.sort(key=lambda s: urgency_order.get(s.urgency, 99))
        best = signals[0]
        logger.info(
            "Exit signal for %s: %s [%s]", token, best.reason, best.urgency.value,
        )
        return best

    def reset_trailing(self, token: str) -> None:
        """Clear trailing-stop state for a closed position."""
        self._trailing_highs.pop(token, None)
