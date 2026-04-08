# trading/risk/risk_manager.py
"""
Portfolio Risk Manager — monitors portfolio-level risk metrics and enforces
safety limits to protect capital.

Tracks drawdown, Value-at-Risk, daily PnL limits, position counts, and
provides emergency shutdown recommendations when risk thresholds are breached.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Defaults                                                               #
# --------------------------------------------------------------------- #
DEFAULT_MAX_DRAWDOWN = 0.15          # 15 % max drawdown before alert
DEFAULT_MAX_DAILY_LOSS_PCT = 0.05    # 5 % daily loss limit
DEFAULT_MAX_OPEN_POSITIONS = 10
DEFAULT_VAR_CONFIDENCE_95 = 0.95
DEFAULT_VAR_CONFIDENCE_99 = 0.99
EMERGENCY_DRAWDOWN = 0.25           # 25 % drawdown → emergency
EMERGENCY_DAILY_LOSS = 0.10         # 10 % daily loss → emergency


# --------------------------------------------------------------------- #
# Enums & data classes                                                   #
# --------------------------------------------------------------------- #
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PortfolioState:
    """Snapshot of current portfolio status."""

    cash: float
    positions: List[Dict]
    total_value: float
    daily_pnl: float
    max_drawdown: float
    peak_value: float

    @property
    def open_count(self) -> int:
        return len(self.positions)


@dataclass
class RiskReport:
    """Comprehensive risk assessment for the portfolio."""

    timestamp: float
    portfolio_value: float
    drawdown_pct: float
    var_95: float
    var_99: float
    daily_pnl: float
    open_positions: int
    risk_level: RiskLevel
    alerts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.portfolio_value = round(self.portfolio_value, 2)
        self.drawdown_pct = round(self.drawdown_pct, 6)
        self.var_95 = round(self.var_95, 2)
        self.var_99 = round(self.var_99, 2)
        self.daily_pnl = round(self.daily_pnl, 2)


# --------------------------------------------------------------------- #
# Risk Manager                                                           #
# --------------------------------------------------------------------- #
class RiskManager:
    """
    Monitors portfolio risk in real time and advises on exposure reduction
    or emergency shutdown.
    """

    def __init__(
        self,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT,
        max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS,
        emergency_drawdown: float = EMERGENCY_DRAWDOWN,
        emergency_daily_loss: float = EMERGENCY_DAILY_LOSS,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.emergency_drawdown = emergency_drawdown
        self.emergency_daily_loss = emergency_daily_loss

        # Historical equity curve for drawdown tracking
        self._equity_curve: List[float] = []
        self._peak_value: float = 0.0
        self._reports: List[RiskReport] = []

        logger.info(
            "RiskManager ready  max_dd=%.0f%%  daily_loss=%.0f%%  "
            "max_pos=%d  emerg_dd=%.0f%%",
            self.max_drawdown * 100, self.max_daily_loss_pct * 100,
            self.max_open_positions, self.emergency_drawdown * 100,
        )

    # ------------------------------------------------------------------ #
    # VaR calculations                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_var(
        returns: List[float],
        confidence: float = DEFAULT_VAR_CONFIDENCE_95,
    ) -> float:
        """
        Value-at-Risk using both historical and parametric methods,
        returning the more conservative (larger loss) estimate.

        Parameters
        ----------
        returns : list of float
            Historical period returns (e.g. daily log-returns).
        confidence : float
            Confidence level (0.95 or 0.99).

        Returns
        -------
        float
            VaR as a positive loss amount (percentage of portfolio).
        """
        if len(returns) < 5:
            return 0.0

        sorted_returns = sorted(returns)
        n = len(sorted_returns)

        # --- Historical VaR ---
        index = int((1.0 - confidence) * n)
        index = max(0, min(index, n - 1))
        hist_var = -sorted_returns[index]

        # --- Parametric VaR (assumes normality) ---
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        # z-scores for common confidence levels
        z_scores = {0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(round(confidence, 2), 1.645)
        param_var = -(mean - z * std_dev)

        # Return the more conservative estimate
        var = max(hist_var, param_var, 0.0)
        logger.debug(
            "VaR(%.0f%%)  hist=%.4f  param=%.4f  → %.4f",
            confidence * 100, hist_var, param_var, var,
        )
        return var

    # ------------------------------------------------------------------ #
    # Drawdown                                                             #
    # ------------------------------------------------------------------ #

    def calculate_drawdown(self, equity_curve: List[float]) -> float:
        """
        Current drawdown from peak as a fraction (0–1).

        Also updates the internal peak tracker.
        """
        if not equity_curve:
            return 0.0

        current = equity_curve[-1]
        peak = max(equity_curve)
        self._peak_value = max(self._peak_value, peak)

        if self._peak_value <= 0:
            return 0.0

        drawdown = (self._peak_value - current) / self._peak_value
        return max(0.0, drawdown)

    # ------------------------------------------------------------------ #
    # Limit checks                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_daily_limits(daily_pnl: float, max_daily_loss: float) -> bool:
        """
        Return True if the daily loss limit has been **breached**.

        Parameters
        ----------
        daily_pnl : float
            Today's PnL as a fraction of portfolio value (negative = loss).
        max_daily_loss : float
            Maximum tolerable daily loss as a positive fraction.
        """
        if daily_pnl < 0 and abs(daily_pnl) >= max_daily_loss:
            logger.warning(
                "Daily loss limit breached: PnL=%.2f%%  limit=%.2f%%",
                daily_pnl * 100, max_daily_loss * 100,
            )
            return True
        return False

    @staticmethod
    def check_position_limits(open_count: int, max_positions: int) -> bool:
        """Return True if the position count limit has been reached."""
        if open_count >= max_positions:
            logger.info(
                "Position limit reached: %d / %d", open_count, max_positions,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Exposure advice                                                      #
    # ------------------------------------------------------------------ #

    def should_reduce_exposure(self, risk_report: RiskReport) -> bool:
        """
        Advise whether the portfolio should reduce exposure based on the
        current risk report.

        Returns True when drawdown exceeds the threshold **or** the risk
        level is HIGH / CRITICAL.
        """
        if risk_report.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            logger.warning(
                "Recommending exposure reduction — risk_level=%s  dd=%.1f%%",
                risk_report.risk_level.value, risk_report.drawdown_pct * 100,
            )
            return True
        if risk_report.drawdown_pct >= self.max_drawdown:
            logger.warning(
                "Recommending exposure reduction — drawdown %.1f%% >= %.1f%%",
                risk_report.drawdown_pct * 100, self.max_drawdown * 100,
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Emergency shutdown                                                   #
    # ------------------------------------------------------------------ #

    def emergency_check(self, portfolio: PortfolioState) -> bool:
        """
        Return True if emergency shutdown conditions are met.

        Conditions:
        - Drawdown exceeds *emergency_drawdown*
        - Daily loss exceeds *emergency_daily_loss*
        - Portfolio value drops to zero or negative
        """
        if portfolio.total_value <= 0:
            logger.critical("EMERGENCY: Portfolio value is <= $0.")
            return True

        daily_loss_pct = 0.0
        if portfolio.total_value > 0:
            daily_loss_pct = abs(min(0.0, portfolio.daily_pnl)) / portfolio.total_value

        if daily_loss_pct >= self.emergency_daily_loss:
            logger.critical(
                "EMERGENCY: Daily loss %.1f%% >= %.1f%%.",
                daily_loss_pct * 100, self.emergency_daily_loss * 100,
            )
            return True

        if portfolio.max_drawdown >= self.emergency_drawdown:
            logger.critical(
                "EMERGENCY: Drawdown %.1f%% >= %.1f%%.",
                portfolio.max_drawdown * 100, self.emergency_drawdown * 100,
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    # Risk-level classification                                            #
    # ------------------------------------------------------------------ #

    def _classify_risk(
        self,
        drawdown: float,
        daily_loss_breached: bool,
        position_limit_hit: bool,
        var_95: float,
    ) -> RiskLevel:
        """Determine overall risk level from component signals."""
        if drawdown >= self.emergency_drawdown or daily_loss_breached:
            return RiskLevel.CRITICAL
        if drawdown >= self.max_drawdown:
            return RiskLevel.HIGH
        if position_limit_hit or var_95 > 0.03:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ------------------------------------------------------------------ #
    # Composite risk report                                                #
    # ------------------------------------------------------------------ #

    def assess_portfolio_risk(self, portfolio: PortfolioState) -> RiskReport:
        """
        Run all risk checks and return a comprehensive RiskReport.

        This is the main entry point — callers should invoke this on every
        iteration of the trading loop.
        """
        # Update equity curve
        self._equity_curve.append(portfolio.total_value)
        self._peak_value = max(self._peak_value, portfolio.peak_value)

        return self.get_risk_report(portfolio)

    def get_risk_report(self, portfolio: PortfolioState) -> RiskReport:
        """
        Generate the full risk report from current portfolio state.
        """
        alerts: List[str] = []

        # 1. Drawdown
        drawdown = self.calculate_drawdown(
            self._equity_curve if self._equity_curve else [portfolio.total_value]
        )

        # 2. Daily PnL as fraction
        daily_pnl_frac = 0.0
        if portfolio.total_value > 0:
            daily_pnl_frac = portfolio.daily_pnl / portfolio.total_value

        daily_breached = self.check_daily_limits(
            daily_pnl_frac, self.max_daily_loss_pct
        )
        if daily_breached:
            alerts.append(
                f"Daily loss limit breached: {daily_pnl_frac:.1%} "
                f"(limit {self.max_daily_loss_pct:.1%})"
            )

        # 3. Position limits
        pos_limit = self.check_position_limits(
            portfolio.open_count, self.max_open_positions
        )
        if pos_limit:
            alerts.append(
                f"Position limit reached: {portfolio.open_count} "
                f"/ {self.max_open_positions}"
            )

        # 4. VaR — use equity curve returns if available
        var_95 = 0.0
        var_99 = 0.0
        if len(self._equity_curve) >= 5:
            returns = [
                (self._equity_curve[i] - self._equity_curve[i - 1])
                / self._equity_curve[i - 1]
                for i in range(1, len(self._equity_curve))
                if self._equity_curve[i - 1] > 0
            ]
            if len(returns) >= 5:
                var_95 = self.calculate_var(returns, DEFAULT_VAR_CONFIDENCE_95)
                var_99 = self.calculate_var(returns, DEFAULT_VAR_CONFIDENCE_99)

        # 5. Emergency
        is_emergency = self.emergency_check(portfolio)
        if is_emergency:
            alerts.append("EMERGENCY SHUTDOWN RECOMMENDED")

        # 6. Drawdown alert
        if drawdown >= self.max_drawdown:
            alerts.append(
                f"Drawdown {drawdown:.1%} exceeds threshold {self.max_drawdown:.1%}"
            )

        # Classify
        risk_level = self._classify_risk(
            drawdown, daily_breached, pos_limit, var_95
        )

        report = RiskReport(
            timestamp=time.time(),
            portfolio_value=portfolio.total_value,
            drawdown_pct=drawdown,
            var_95=var_95 * portfolio.total_value if portfolio.total_value > 0 else 0,
            var_99=var_99 * portfolio.total_value if portfolio.total_value > 0 else 0,
            daily_pnl=portfolio.daily_pnl,
            open_positions=portfolio.open_count,
            risk_level=risk_level,
            alerts=alerts,
        )

        self._reports.append(report)
        # Keep report history bounded
        if len(self._reports) > 1_000:
            self._reports = self._reports[-500:]

        logger.info(
            "Risk report  value=$%.2f  dd=%.1f%%  VaR95=$%.2f  "
            "level=%s  alerts=%d",
            report.portfolio_value, report.drawdown_pct * 100,
            report.var_95, report.risk_level.value, len(report.alerts),
        )
        return report

    @property
    def reports(self) -> List[RiskReport]:
        """Access historical risk reports."""
        return list(self._reports)
