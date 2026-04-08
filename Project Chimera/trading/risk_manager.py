# trading/risk_manager.py
"""
Risk Management System — dynamic position sizing, adaptive stops,
trailing stops, drawdown protection, and circuit breakers.

This module wraps position-sizing and exit-strategy decisions so that
the paper trader (and eventually a live trader) can delegate all risk
logic to a single, testable component.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import init_db, get_virtual_cash, get_performance_summary

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Centralises all risk-related decisions for the trading bot.
    """

    def __init__(self):
        self.conn = init_db()
        self._peak_equity: Optional[float] = None
        self._circuit_breaker_active = False
        self._consecutive_losses = 0
        logger.info("RiskManager initialised.")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def compute_position_size(
        self,
        signal_score: float,
        token_volatility: Optional[float] = None,
    ) -> float:
        """
        Compute the optimal position size in USD.

        Uses a modified Kelly Criterion with a safety multiplier and
        adjusts for:
        - Signal confidence (signal_score)
        - Token volatility (if available)
        - Current drawdown level
        - Remaining capital

        Parameters
        ----------
        signal_score : float
            Confidence score of the trading signal (0.0–1.0).
        token_volatility : float or None
            Estimated price volatility (e.g. std-dev of returns).
            If None, uses a conservative default.

        Returns
        -------
        float
            Position size in USD.
        """
        cash = get_virtual_cash(self.conn)

        if cash < config.PAPER_MIN_TRADE_USD:
            return 0.0

        if self._circuit_breaker_active:
            logger.warning("Circuit breaker active — position size = 0.")
            return 0.0

        # Base position: percentage of cash
        base_pct = config.PAPER_POSITION_SIZE_PCT / 100.0

        # Kelly-inspired adjustment: scale by confidence
        # kelly_fraction = edge / odds ≈ signal_score for simplified model
        kelly_factor = max(signal_score, 0.1)

        # Safety multiplier (never bet full Kelly — use half-Kelly)
        safety = config.RISK_KELLY_SAFETY_MULTIPLIER

        # Volatility adjustment: reduce size for volatile tokens
        vol_factor = 1.0
        if token_volatility is not None and token_volatility > 0:
            # Higher volatility → smaller position
            vol_factor = max(0.3, 1.0 / (1.0 + token_volatility * 10))

        # Drawdown adjustment: reduce size when in drawdown
        drawdown_factor = self._drawdown_factor()

        # Compute final position
        position_pct = base_pct * kelly_factor * safety * vol_factor * drawdown_factor
        position_usd = cash * position_pct

        # Apply floor and ceiling
        position_usd = max(position_usd, config.PAPER_MIN_TRADE_USD)
        position_usd = min(position_usd, config.PAPER_MAX_POSITION_USD)
        position_usd = min(position_usd, cash)

        logger.debug(
            "Position size: $%.2f (kelly=%.2f, vol=%.2f, dd=%.2f)",
            position_usd, kelly_factor, vol_factor, drawdown_factor,
        )
        return position_usd

    def compute_stop_loss(
        self,
        entry_price: float,
        token_volatility: Optional[float] = None,
    ) -> float:
        """
        Compute an adaptive stop-loss price.

        Uses ATR-like volatility adjustment: wider stops for volatile
        tokens, tighter stops for stable ones.

        Returns the stop-loss price (below entry_price for longs).
        """
        base_sl_pct = config.PAPER_STOP_LOSS_PCT / 100.0

        if token_volatility is not None and token_volatility > 0:
            # Widen stop for volatile tokens (up to 2x base)
            atr_multiplier = min(1.0 + token_volatility * 5, 2.0)
            sl_pct = base_sl_pct * atr_multiplier
        else:
            sl_pct = base_sl_pct

        stop_price = entry_price * (1.0 - sl_pct)
        return max(stop_price, 0.0)

    def compute_take_profit(
        self,
        entry_price: float,
        signal_score: float,
    ) -> float:
        """
        Compute a dynamic take-profit price.

        Higher-confidence signals get wider profit targets.

        Returns the take-profit price (above entry_price for longs).
        """
        base_tp_pct = config.PAPER_TAKE_PROFIT_PCT / 100.0

        # Scale TP target with signal confidence (1x–2x base)
        confidence_multiplier = 1.0 + signal_score
        tp_pct = base_tp_pct * confidence_multiplier

        return entry_price * (1.0 + tp_pct)

    def compute_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
    ) -> float:
        """
        Compute a trailing stop price.

        The trailing stop moves up as the price increases, locking in
        profits.  It never moves down.

        Returns the trailing stop price.
        """
        trail_pct = config.RISK_TRAILING_STOP_PCT / 100.0

        # Trail from the highest price seen
        peak = max(highest_price, current_price, entry_price)
        trailing_stop = peak * (1.0 - trail_pct)

        # Never set trailing stop below the original stop loss
        original_sl = self.compute_stop_loss(entry_price)
        return max(trailing_stop, original_sl)

    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        signal_score: float,
    ) -> tuple[bool, str, str]:
        """
        Determine whether to exit a position.

        Returns (should_exit, outcome, reason).
        """
        if entry_price <= 0:
            return False, "OPEN", "Invalid entry price"

        pnl_pct = (current_price - entry_price) / entry_price * 100.0

        # Take profit
        tp_price = self.compute_take_profit(entry_price, signal_score)
        if current_price >= tp_price:
            return True, "WIN", f"Take profit hit at {pnl_pct:+.1f}%"

        # Trailing stop (only when in profit)
        if current_price > entry_price:
            trail_price = self.compute_trailing_stop(
                entry_price, current_price, highest_price,
            )
            if current_price <= trail_price:
                return True, "WIN", f"Trailing stop hit at {pnl_pct:+.1f}%"

        # Stop loss
        sl_price = self.compute_stop_loss(entry_price)
        if current_price <= sl_price:
            return True, "LOSS", f"Stop loss hit at {pnl_pct:+.1f}%"

        return False, "OPEN", f"Position open at {pnl_pct:+.1f}%"

    def check_circuit_breaker(self) -> bool:
        """
        Check if the circuit breaker should be activated.

        Triggers when:
        - Max drawdown exceeded
        - Too many consecutive losses
        - Available capital too low

        Returns True if the circuit breaker is now active.
        """
        summary = get_performance_summary(self.conn)
        cash = summary.get("virtual_cash_usd", 0)
        start = config.PAPER_TRADING_STARTING_BALANCE

        # Check drawdown
        if self._peak_equity is None:
            self._peak_equity = start
        current_equity = cash + summary.get("total_pnl_usd", 0)
        self._peak_equity = max(self._peak_equity, current_equity)

        drawdown_pct = 0.0
        if self._peak_equity > 0:
            drawdown_pct = (self._peak_equity - current_equity) / self._peak_equity * 100

        if drawdown_pct >= config.RISK_MAX_DRAWDOWN_PCT:
            self._circuit_breaker_active = True
            logger.warning(
                "CIRCUIT BREAKER: Max drawdown %.1f%% exceeded (limit: %.1f%%).",
                drawdown_pct, config.RISK_MAX_DRAWDOWN_PCT,
            )
            return True

        # Check consecutive losses
        losses = summary.get("losses", 0) or 0
        wins = summary.get("wins", 0) or 0
        if losses > 0 and wins == 0:
            self._consecutive_losses = losses
        elif losses > self._consecutive_losses:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= config.RISK_MAX_CONSECUTIVE_LOSSES:
            self._circuit_breaker_active = True
            logger.warning(
                "CIRCUIT BREAKER: %d consecutive losses (limit: %d).",
                self._consecutive_losses, config.RISK_MAX_CONSECUTIVE_LOSSES,
            )
            return True

        # Check minimum capital
        if cash < start * 0.1:
            self._circuit_breaker_active = True
            logger.warning(
                "CIRCUIT BREAKER: Capital critically low ($%.2f).", cash,
            )
            return True

        self._circuit_breaker_active = False
        return False

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker after review."""
        self._circuit_breaker_active = False
        self._consecutive_losses = 0
        logger.info("Circuit breaker reset.")

    def record_trade_outcome(self, outcome: str) -> None:
        """
        Track consecutive wins/losses for circuit breaker logic.

        Parameters
        ----------
        outcome : str
            'WIN' or 'LOSS'.
        """
        if outcome == "LOSS":
            self._consecutive_losses += 1
        elif outcome == "WIN":
            self._consecutive_losses = 0

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _drawdown_factor(self) -> float:
        """
        Return a scaling factor (0.1–1.0) based on current drawdown.

        The deeper the drawdown, the smaller the positions.
        """
        summary = get_performance_summary(self.conn)
        cash = summary.get("virtual_cash_usd", 0)
        start = config.PAPER_TRADING_STARTING_BALANCE

        if start <= 0:
            return 1.0

        equity_ratio = cash / start
        if equity_ratio >= 1.0:
            return 1.0  # No drawdown

        # Linear scale: 100% equity → 1.0, 50% equity → 0.5, etc.
        return max(0.1, equity_ratio)
