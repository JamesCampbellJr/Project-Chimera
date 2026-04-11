"""Performance metrics and reporting for backtesting results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Complete performance report generated from backtest results.

    Attributes:
        sharpe: Annualized Sharpe ratio.
        sortino: Annualized Sortino ratio.
        max_drawdown: Maximum drawdown as a decimal fraction (e.g. 0.15 = 15%).
        max_drawdown_duration_days: Longest drawdown duration in calendar days.
        win_rate: Fraction of profitable trades (0.0 to 1.0).
        profit_factor: Gross profits divided by gross losses.
        expectancy: Expected PnL per trade.
        calmar: Calmar ratio (annualized return / max drawdown).
        total_return: Total return as a decimal fraction.
        annualized_return: Annualized return as a decimal fraction.
        total_trades: Number of trades executed.
        avg_trade_pnl: Average trade PnL in absolute terms.
        best_trade_pnl: Best single-trade PnL.
        worst_trade_pnl: Worst single-trade PnL.
        daily_returns: List of daily return values.
        monthly_returns: Mapping of YYYY-MM strings to monthly return percentages.
        benchmark_correlation: Pearson correlation with benchmark returns.
    """

    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float
    expectancy: float
    calmar: float
    total_return: float
    annualized_return: float
    total_trades: int
    avg_trade_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    daily_returns: List[float] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    benchmark_correlation: Optional[float] = None


class PerformanceAnalyzer:
    """Static utility class for computing backtesting performance metrics.

    All calculations use pure numpy — no external analytics libraries required.
    """

    @staticmethod
    def calculate_sharpe(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            returns: Array of periodic returns.
            risk_free_rate: Annualized risk-free rate (decimal).
            periods_per_year: Number of periods in a year (252 for daily).

        Returns:
            Annualized Sharpe ratio, or 0.0 when insufficient data.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            logger.warning("Insufficient returns for Sharpe calculation")
            return 0.0

        periodic_rf = risk_free_rate / periods_per_year
        excess = returns - periodic_rf
        std = np.std(excess, ddof=1)
        if std == 0.0:
            return 0.0

        return float(np.mean(excess) / std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_sortino(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized Sortino ratio using downside deviation.

        Args:
            returns: Array of periodic returns.
            risk_free_rate: Annualized risk-free rate (decimal).
            periods_per_year: Number of periods in a year (252 for daily).

        Returns:
            Annualized Sortino ratio, or 0.0 when insufficient data.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            logger.warning("Insufficient returns for Sortino calculation")
            return 0.0

        periodic_rf = risk_free_rate / periods_per_year
        excess = returns - periodic_rf
        downside = np.minimum(excess, 0.0)
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std == 0.0:
            return 0.0

        return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_max_drawdown(
        equity_curve: np.ndarray,
    ) -> Tuple[float, int]:
        """Calculate maximum drawdown percentage and its duration in days.

        Args:
            equity_curve: Array of portfolio equity values over time.

        Returns:
            Tuple of (max_drawdown_pct, duration_days).  Drawdown is expressed
            as a positive decimal fraction (e.g. 0.25 for a 25 % drawdown).
            Duration is the number of periods from peak to trough of the
            worst drawdown.
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        if len(equity_curve) < 2:
            return 0.0, 0

        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / np.where(
            running_max == 0.0, 1.0, running_max
        )
        max_dd = float(np.max(drawdowns))

        # Compute duration of the longest drawdown (peak-to-recovery or
        # peak-to-end-of-series).
        longest = 0
        current = 0
        for dd in drawdowns:
            if dd > 0.0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return max_dd, longest

    @staticmethod
    def calculate_win_rate(trades: List[Any]) -> float:
        """Calculate win rate as the fraction of profitable trades.

        Args:
            trades: List of Trade objects with a ``pnl`` attribute.

        Returns:
            Win rate between 0.0 and 1.0, or 0.0 if no trades.
        """
        if not trades:
            return 0.0
        winners = sum(1 for t in trades if t.pnl > 0.0)
        return float(winners / len(trades))

    @staticmethod
    def calculate_profit_factor(trades: List[Any]) -> float:
        """Calculate profit factor (gross profits / gross losses).

        Args:
            trades: List of Trade objects with a ``pnl`` attribute.

        Returns:
            Profit factor.  Returns ``float('inf')`` when there are no
            losses, and 0.0 when there are no trades.
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0.0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0.0))

        if gross_loss == 0.0:
            return float("inf") if gross_profit > 0.0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def calculate_expectancy(trades: List[Any]) -> float:
        """Calculate trade expectancy.

        Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        Args:
            trades: List of Trade objects with a ``pnl`` attribute.

        Returns:
            Expected PnL per trade, or 0.0 when there are no trades.
        """
        if not trades:
            return 0.0

        winners = [t.pnl for t in trades if t.pnl > 0.0]
        losers = [abs(t.pnl) for t in trades if t.pnl <= 0.0]

        total = len(trades)
        win_rate = len(winners) / total
        loss_rate = len(losers) / total

        avg_win = float(np.mean(winners)) if winners else 0.0
        avg_loss = float(np.mean(losers)) if losers else 0.0

        return float(win_rate * avg_win - loss_rate * avg_loss)

    @staticmethod
    def calculate_calmar(
        annualized_return: float,
        max_drawdown: float,
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown).

        Args:
            annualized_return: Annualized return as a decimal fraction.
            max_drawdown: Maximum drawdown as a positive decimal fraction.

        Returns:
            Calmar ratio, or 0.0 when max drawdown is zero.
        """
        if max_drawdown == 0.0:
            return 0.0
        return float(annualized_return / max_drawdown)

    @staticmethod
    def calculate_monthly_returns(
        daily_returns: np.ndarray,
        start_date: datetime,
    ) -> Dict[str, float]:
        """Aggregate daily returns into monthly return percentages.

        Args:
            daily_returns: Array of daily return values.
            start_date: Calendar date of the first daily return.

        Returns:
            Dict mapping ``YYYY-MM`` strings to the compounded monthly
            return expressed as a percentage.
        """
        daily_returns = np.asarray(daily_returns, dtype=np.float64)
        if len(daily_returns) == 0:
            return {}

        monthly: Dict[str, List[float]] = {}
        for i, ret in enumerate(daily_returns):
            date = start_date + timedelta(days=i)
            key = date.strftime("%Y-%m")
            monthly.setdefault(key, []).append(float(ret))

        result: Dict[str, float] = {}
        for key, rets in monthly.items():
            compounded = float(np.prod(1.0 + np.array(rets)) - 1.0)
            result[key] = compounded * 100.0

        return result

    @staticmethod
    def calculate_benchmark_correlation(
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Optional[float]:
        """Calculate Pearson correlation between strategy and benchmark returns.

        Args:
            returns: Array of strategy returns.
            benchmark_returns: Array of benchmark returns.

        Returns:
            Pearson correlation coefficient, or ``None`` when computation
            is not possible (mismatched lengths, insufficient data, or
            zero variance).
        """
        returns = np.asarray(returns, dtype=np.float64)
        benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

        if len(returns) != len(benchmark_returns):
            logger.warning(
                "Returns length (%d) does not match benchmark length (%d)",
                len(returns),
                len(benchmark_returns),
            )
            return None

        if len(returns) < 2:
            return None

        std_r = np.std(returns)
        std_b = np.std(benchmark_returns)
        if std_r == 0.0 or std_b == 0.0:
            return None

        corr_matrix = np.corrcoef(returns, benchmark_returns)
        return float(corr_matrix[0, 1])

    @staticmethod
    def full_report(
        equity_curve: np.ndarray,
        trades: List[Any],
        start_date: datetime,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.0,
    ) -> PerformanceReport:
        """Generate a complete performance report from backtest outputs.

        Args:
            equity_curve: Array of portfolio equity values over time.
            trades: List of Trade objects produced by the backtester.
            start_date: Calendar date corresponding to the first equity value.
            benchmark_returns: Optional array of benchmark daily returns for
                correlation analysis.
            risk_free_rate: Annualized risk-free rate (decimal).

        Returns:
            A fully populated ``PerformanceReport``.
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)

        # --- Daily returns from equity curve ---
        if len(equity_curve) >= 2:
            daily_ret = np.diff(equity_curve) / np.where(
                equity_curve[:-1] == 0.0, 1.0, equity_curve[:-1]
            )
        else:
            daily_ret = np.array([], dtype=np.float64)

        # --- Total / annualized return ---
        if len(equity_curve) >= 2 and equity_curve[0] != 0.0:
            total_return = float(
                (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            )
        else:
            total_return = 0.0

        n_days = max(len(daily_ret), 1)
        annualized_return = float(
            (1.0 + total_return) ** (252.0 / n_days) - 1.0
        )

        # --- Risk metrics ---
        sharpe = PerformanceAnalyzer.calculate_sharpe(
            daily_ret, risk_free_rate
        )
        sortino = PerformanceAnalyzer.calculate_sortino(
            daily_ret, risk_free_rate
        )
        max_dd, max_dd_dur = PerformanceAnalyzer.calculate_max_drawdown(
            equity_curve
        )
        calmar = PerformanceAnalyzer.calculate_calmar(
            annualized_return, max_dd
        )

        # --- Trade metrics ---
        win_rate = PerformanceAnalyzer.calculate_win_rate(trades)
        profit_factor = PerformanceAnalyzer.calculate_profit_factor(trades)
        expectancy = PerformanceAnalyzer.calculate_expectancy(trades)

        total_trades = len(trades)
        if total_trades > 0:
            pnls = [t.pnl for t in trades]
            avg_trade_pnl = float(np.mean(pnls))
            best_trade_pnl = float(np.max(pnls))
            worst_trade_pnl = float(np.min(pnls))
        else:
            avg_trade_pnl = 0.0
            best_trade_pnl = 0.0
            worst_trade_pnl = 0.0

        # --- Monthly returns ---
        monthly = PerformanceAnalyzer.calculate_monthly_returns(
            daily_ret, start_date
        )

        # --- Benchmark correlation ---
        bench_corr: Optional[float] = None
        if benchmark_returns is not None:
            bench_corr = PerformanceAnalyzer.calculate_benchmark_correlation(
                daily_ret, benchmark_returns
            )

        return PerformanceReport(
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_dur,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            calmar=calmar,
            total_return=total_return,
            annualized_return=annualized_return,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            best_trade_pnl=best_trade_pnl,
            worst_trade_pnl=worst_trade_pnl,
            daily_returns=daily_ret.tolist(),
            monthly_returns=monthly,
            benchmark_correlation=bench_corr,
        )
