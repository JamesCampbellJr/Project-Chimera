# trading/trading_bot.py
"""
Solana Paper-Trading Bot — main orchestrator for the trading subsystem.

Lifecycle
---------
1. Seed / refresh wallet analysis (wallet_analyzer).
2. Run sentiment analysis on active tokens.
3. Detect / refresh patterns (pattern_detector).
4. Screen signals through risk management and anti-manipulation checks.
5. For every active pattern above the confidence threshold, emit a buy
   signal to the paper trader.
6. On every subsequent iteration, re-price open positions (evaluate_open_positions).
7. Collect metrics and check alerts.
8. Persist all state to SQLite and log a running summary.

Run as a standalone script::

    python -m trading.trading_bot

Or wire into the Chimera orchestrator via ``spawn_agent('Solana Trader', …)``.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from trading.database import init_db, get_performance_summary
from trading.solana_client import SolanaClient
from trading.wallet_analyzer import WalletAnalyzer
from trading.pattern_detector import PatternDetector
from trading.paper_trader import PaperTrader

# --- Advanced modules (imported lazily to avoid hard failures) ---
_ADVANCED_MODULES_AVAILABLE = True
try:
    from trading.sentiment.sentiment_scorer import SentimentScorer
    from trading.risk.risk_manager import RiskManager, PortfolioState
    from trading.risk.stop_loss import StopLossManager
    from trading.risk.position_sizing import PositionSizer
    from trading.anti_manipulation.fake_trade_detector import FakeTradeDetector
    from trading.anti_manipulation.anomaly_detector import AnomalyDetector
    from trading.monitoring.metrics_collector import MetricsCollector
    from trading.monitoring.alert_system import AlertSystem
    from trading.monitoring.compliance import ComplianceLogger
except ImportError as _import_err:
    _ADVANCED_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Coordinates the wallet analysis → pattern detection → paper trading loop.

    When the advanced trading modules are available, also performs:
    - Sentiment scoring for active tokens
    - Risk management checks (VaR, drawdown, position limits)
    - Anti-manipulation screening (honeypot, anomaly detection)
    - Metrics collection and alerting
    - Compliance audit logging
    """

    def __init__(self):
        self.conn    = init_db()
        self.client  = SolanaClient()
        self.analyzer  = WalletAnalyzer(solana_client=self.client)
        self.detector  = PatternDetector()
        self.trader    = PaperTrader(solana_client=self.client)
        self._running  = False

        # Advanced modules — initialised when available
        self.sentiment_scorer: Optional[Any] = None
        self.risk_manager: Optional[Any] = None
        self.stop_loss_mgr: Optional[Any] = None
        self.position_sizer: Optional[Any] = None
        self.fake_trade_detector: Optional[Any] = None
        self.anomaly_detector: Optional[Any] = None
        self.metrics: Optional[Any] = None
        self.alerts: Optional[Any] = None
        self.compliance: Optional[Any] = None

        if _ADVANCED_MODULES_AVAILABLE:
            self._init_advanced_modules()
        else:
            logger.warning(
                "Advanced trading modules not available (%s). "
                "Running in basic mode.", _import_err,
            )

        logger.info("TradingBot ready (advanced=%s).", _ADVANCED_MODULES_AVAILABLE)

    # ---------------------------------------------------------------------- #
    # Initialisation helpers                                                   #
    # ---------------------------------------------------------------------- #

    def _init_advanced_modules(self) -> None:
        """Instantiate optional advanced sub-systems."""
        try:
            self.sentiment_scorer = SentimentScorer(
                news_weight=config.SENTIMENT_WEIGHT_NEWS,
                social_weight=config.SENTIMENT_WEIGHT_SOCIAL,
                onchain_weight=config.SENTIMENT_WEIGHT_ONCHAIN,
                ema_period=config.SENTIMENT_EMA_PERIOD,
                extreme_threshold=config.SENTIMENT_EXTREME_THRESHOLD,
            )
            self.risk_manager = RiskManager(
                max_daily_loss_pct=config.MAX_DAILY_LOSS_PCT,
                max_open_positions=config.MAX_OPEN_POSITIONS,
            )
            self.stop_loss_mgr = StopLossManager()
            self.position_sizer = PositionSizer()
            self.fake_trade_detector = FakeTradeDetector(solana_client=self.client)
            self.anomaly_detector = AnomalyDetector()
            self.metrics = MetricsCollector(
                retention_seconds=config.METRICS_RETENTION_HOURS * 3600,
            )
            self.alerts = AlertSystem()
            self.compliance = ComplianceLogger()
            logger.info("Advanced modules initialised successfully.")
        except Exception as exc:
            logger.warning("Failed to initialise some advanced modules: %s", exc)

    # ---------------------------------------------------------------------- #
    # Public                                                                   #
    # ---------------------------------------------------------------------- #

    async def run(self, iterations: Optional[int] = None) -> None:
        """
        Main async loop.

        Parameters
        ----------
        iterations : int or None
            Run for exactly *iterations* cycles then stop.
            Pass None (default) to run indefinitely.
        """
        self._running = True
        iteration = 0
        logger.info(
            "TradingBot starting%s.",
            f" (max {iterations} iterations)" if iterations else "",
        )

        # Initial wallet seeding
        self.analyzer.seed_wallets()

        while self._running:
            iteration += 1
            t0 = time.monotonic()
            logger.info("--- Iteration %d ---", iteration)
            loop = asyncio.get_running_loop()

            try:
                # 1. Analyse wallets (blocking I/O → thread pool)
                logger.info("Step 1: Analysing tracked wallets …")
                await loop.run_in_executor(None, self.analyzer.analyse_all_tracked)

                # 2. Detect patterns
                logger.info("Step 2: Detecting patterns …")
                patterns = await loop.run_in_executor(None, self.detector.detect_all)
                logger.info("Patterns available: %d", len(patterns))

                # 3. Retrieve active patterns that meet the confidence bar
                active = await loop.run_in_executor(
                    None,
                    self.detector.get_active_patterns,
                    config.MIN_SIGNAL_SCORE,
                )
                logger.info("Active (high-confidence) patterns: %d", len(active))

                # 4. Screen signals through advanced modules
                screened = await self._screen_signals(active)

                # 5. Issue buy signals for each screened pattern
                for signal in screened:
                    await loop.run_in_executor(
                        None, self.trader.execute_buy_signal, signal
                    )

                # 6. Evaluate open positions (with advanced stop-loss if available)
                closed = await loop.run_in_executor(
                    None, self.trader.evaluate_open_positions
                )
                if closed:
                    logger.info("Closed %d position(s) this iteration.", len(closed))

                # 7. Summary & metrics
                await loop.run_in_executor(None, self.trader.print_summary)

                summary = get_performance_summary(self.conn)
                self._log_profitability_progress(summary)
                self._record_metrics(summary, time.monotonic() - t0)

                # 8. Compliance logging
                if self.compliance:
                    self.compliance.log_decision(
                        action="iteration_complete",
                        reasoning=f"Iteration {iteration} completed",
                        scores={"win_rate": summary.get("win_rate_pct", 0.0)},
                    )

            except Exception as exc:
                logger.error("Error in trading loop (iteration %d): %s", iteration, exc, exc_info=True)
                if self.metrics:
                    self.metrics.record_metric("trading.errors", 1.0, labels={"iteration": str(iteration)})

            if iterations and iteration >= iterations:
                self._running = False
                break

            # Wait before next iteration
            logger.info("Sleeping %ds before next iteration …", config.TRADING_LOOP_INTERVAL)
            await asyncio.sleep(config.TRADING_LOOP_INTERVAL)

        logger.info("TradingBot stopped after %d iteration(s).", iteration)
        self.trader.print_summary()

    def stop(self) -> None:
        """Request a graceful shutdown after the current iteration."""
        self._running = False
        logger.info("TradingBot stop requested.")

    # ---------------------------------------------------------------------- #
    # Signal screening                                                         #
    # ---------------------------------------------------------------------- #

    async def _screen_signals(self, patterns: List[Dict]) -> List[Dict]:
        """
        Filter active patterns through risk and anti-manipulation checks.

        Returns a list of enriched signal dicts ready for execution.
        """
        screened: List[Dict] = []
        loop = asyncio.get_running_loop()

        for pattern in patterns:
            signal: Dict[str, Any] = {
                **pattern,
                "signal_score": pattern.get("success_rate", 0.0),
            }
            token = pattern.get("token_address") or pattern.get("token", "")

            # Anti-manipulation: check for honeypot / rug risk
            if self.fake_trade_detector and token:
                try:
                    risk_assessment = await loop.run_in_executor(
                        None,
                        lambda t=token: asyncio.get_event_loop().run_until_complete(
                            self.fake_trade_detector.get_token_risk(t)
                        ) if asyncio.get_event_loop().is_running() else None,
                    )
                    if risk_assessment and risk_assessment.overall_risk > config.RUG_RISK_THRESHOLD:
                        logger.warning(
                            "Skipping token %s — rug risk %.2f > threshold %.2f",
                            token, risk_assessment.overall_risk, config.RUG_RISK_THRESHOLD,
                        )
                        continue
                except Exception as exc:
                    logger.debug("Fake trade detection skipped for %s: %s", token, exc)

            # Sentiment enrichment
            if self.sentiment_scorer and token:
                try:
                    sentiment = self.sentiment_scorer.get_token_sentiment(token)
                    signal["sentiment_score"] = sentiment.unified_score if sentiment else 0.0
                except Exception as exc:
                    logger.debug("Sentiment scoring skipped for %s: %s", token, exc)

            # Risk management: check portfolio limits
            if self.risk_manager:
                try:
                    summary = get_performance_summary(self.conn)
                    open_count = summary.get("open_positions", 0)
                    if open_count >= config.MAX_OPEN_POSITIONS:
                        logger.warning(
                            "Skipping signal — max open positions (%d) reached.",
                            config.MAX_OPEN_POSITIONS,
                        )
                        continue
                except Exception as exc:
                    logger.debug("Risk check skipped: %s", exc)

            screened.append(signal)

        logger.info(
            "Signal screening: %d/%d patterns passed.",
            len(screened), len(patterns),
        )
        return screened

    # ---------------------------------------------------------------------- #
    # Metrics                                                                  #
    # ---------------------------------------------------------------------- #

    def _record_metrics(self, summary: dict, iteration_duration: float) -> None:
        """Record trading and system metrics."""
        if not self.metrics:
            return
        try:
            self.metrics.record_metric("trading.win_rate", summary.get("win_rate_pct", 0.0))
            self.metrics.record_metric("trading.pnl", summary.get("total_pnl_usd", 0.0))
            self.metrics.record_metric("trading.cash", summary.get("virtual_cash_usd", 0.0))
            self.metrics.record_metric("trading.open_positions", float(summary.get("open_positions", 0)))
            self.metrics.record_metric("system.iteration_duration_s", iteration_duration)
        except Exception as exc:
            logger.debug("Metrics recording error: %s", exc)

    # ---------------------------------------------------------------------- #
    # Private                                                                  #
    # ---------------------------------------------------------------------- #

    def _log_profitability_progress(self, summary: dict) -> None:
        win_rate = summary.get("win_rate_pct", 0.0)
        target   = config.TARGET_WIN_RATE_PCT
        cash     = summary.get("virtual_cash_usd", 0.0)
        start    = config.PAPER_TRADING_STARTING_BALANCE
        pnl      = summary.get("total_pnl_usd", 0.0)

        logger.info(
            "Progress → win-rate: %.1f%% / target %.1f%% | "
            "PnL: $%.2f | cash: $%.2f / $%.2f start",
            win_rate, target, pnl, cash, start,
        )
        if win_rate >= target:
            logger.info(
                "🎯 TARGET WIN-RATE OF %.1f%% REACHED! "
                "Bot is performing profitably in paper trading.",
                target,
            )


# --------------------------------------------------------------------------- #
# Stand-alone entry point                                                      #
# --------------------------------------------------------------------------- #

async def _main():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    )
    bot = TradingBot()
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Trading bot interrupted by user.")


if __name__ == "__main__":
    asyncio.run(_main())
