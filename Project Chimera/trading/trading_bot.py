# trading/trading_bot.py
"""
Solana Paper-Trading Bot — "Project Phoenix" — main orchestrator for the
trading subsystem.

Lifecycle
---------
1. Seed / refresh wallet analysis (wallet_analyzer).
2. Fetch and score sentiment data (sentiment_analyzer).
3. Scan for manipulation / fake trades (fake_trade_detector).
4. Detect / refresh patterns (pattern_detector).
5. For every active pattern above the confidence threshold, run through
   risk management and sentiment checks before issuing a buy signal.
6. On every subsequent iteration, re-price open positions with adaptive
   stop-loss, trailing stop, and take-profit via the risk manager.
7. Check circuit breakers and persist all state to SQLite.

Run as a standalone script::

    python -m trading.trading_bot

Or wire into the Chimera orchestrator via ``spawn_agent('Solana Trader', …)``.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import init_db, get_performance_summary
from trading.solana_client import SolanaClient
from trading.wallet_analyzer import WalletAnalyzer
from trading.pattern_detector import PatternDetector
from trading.paper_trader import PaperTrader
from trading.sentiment_analyzer import SentimentAnalyzer
from trading.fake_trade_detector import FakeTradeDetector
from trading.risk_manager import RiskManager
from trading.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Coordinates the full Project Phoenix trading pipeline:
    wallet analysis → sentiment → manipulation scan → pattern detection →
    risk management → paper trading.
    """

    def __init__(self):
        self.conn       = init_db()
        self.client     = SolanaClient()
        self.analyzer   = WalletAnalyzer(solana_client=self.client)
        self.detector   = PatternDetector()
        self.trader     = PaperTrader(solana_client=self.client)
        self.sentiment  = SentimentAnalyzer()
        self.fake_detector = FakeTradeDetector()
        self.risk       = RiskManager()
        self.indicators = TechnicalIndicators()
        self._running   = False
        logger.info("TradingBot (Project Phoenix) ready.")

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
            logger.info("=== Iteration %d ===", iteration)
            loop = asyncio.get_running_loop()

            try:
                # 0. Check circuit breaker before doing anything
                if self.risk.check_circuit_breaker():
                    logger.warning("Circuit breaker active — skipping trading this iteration.")
                    summary = get_performance_summary(self.conn)
                    self._log_profitability_progress(summary)
                    if iterations and iteration >= iterations:
                        break
                    await asyncio.sleep(config.TRADING_LOOP_INTERVAL)
                    continue

                # 1. Analyse wallets (blocking I/O → thread pool)
                logger.info("Step 1: Analysing tracked wallets …")
                await loop.run_in_executor(None, self.analyzer.analyse_all_tracked)

                # 2. Fetch and score sentiment data
                logger.info("Step 2: Fetching sentiment data …")
                await loop.run_in_executor(None, self.sentiment.fetch_and_score_news)
                await loop.run_in_executor(None, self.sentiment.fetch_trending_sentiment)
                market_sentiment = await loop.run_in_executor(
                    None, self.sentiment.get_market_sentiment
                )
                logger.info("Market sentiment: %.2f", market_sentiment)

                # 3. Scan for manipulation / fake trades
                logger.info("Step 3: Scanning for manipulation …")
                manipulation_flags = await loop.run_in_executor(
                    None, self.fake_detector.scan_all
                )
                if manipulation_flags:
                    logger.warning(
                        "⚠ %d manipulation flag(s) detected!", len(manipulation_flags)
                    )

                # 4. Detect patterns
                logger.info("Step 4: Detecting patterns …")
                patterns = await loop.run_in_executor(None, self.detector.detect_all)
                logger.info("Patterns available: %d", len(patterns))

                # 5. Retrieve active patterns that meet the confidence bar
                active = await loop.run_in_executor(
                    None,
                    self.detector.get_active_patterns,
                    config.MIN_SIGNAL_SCORE,
                )
                logger.info("Active (high-confidence) patterns: %d", len(active))

                # 6. Issue buy signals — with sentiment and manipulation checks
                signals_issued = 0
                signals_blocked = 0
                for pattern in active:
                    token_address = pattern.get("token_address")
                    if not token_address:
                        continue

                    # 6a. Check manipulation flags
                    is_safe, safety_reason = await loop.run_in_executor(
                        None, self.fake_detector.is_safe_to_trade, token_address
                    )
                    if not is_safe:
                        logger.info("BLOCKED (manipulation): %s — %s", token_address, safety_reason)
                        signals_blocked += 1
                        continue

                    # 6b. Check sentiment
                    should_trade, sent_reason = await loop.run_in_executor(
                        None, self.sentiment.should_trade, token_address
                    )
                    if not should_trade:
                        logger.info("BLOCKED (sentiment): %s — %s", token_address, sent_reason)
                        signals_blocked += 1
                        continue

                    # 6c. Compute position size via risk manager
                    signal_score = pattern.get("success_rate", 0.0)
                    position_usd = await loop.run_in_executor(
                        None, self.risk.compute_position_size, signal_score, None
                    )
                    if position_usd <= 0:
                        logger.info("BLOCKED (risk): position size = 0 for %s", token_address)
                        signals_blocked += 1
                        continue

                    signal = {
                        **pattern,
                        "signal_score": signal_score,
                        "position_size_usd": position_usd,
                    }
                    await loop.run_in_executor(
                        None, self.trader.execute_buy_signal, signal
                    )
                    signals_issued += 1

                logger.info(
                    "Signals: %d issued, %d blocked.", signals_issued, signals_blocked
                )

                # 7. Evaluate open positions (with risk manager exit logic)
                closed = await loop.run_in_executor(
                    None, self.trader.evaluate_open_positions
                )
                if closed:
                    logger.info("Closed %d position(s) this iteration.", len(closed))
                    for trade in closed:
                        self.risk.record_trade_outcome(trade.get("outcome", ""))

                # 8. Summary
                await loop.run_in_executor(None, self.trader.print_summary)

                summary = get_performance_summary(self.conn)
                self._log_profitability_progress(summary)

            except Exception as exc:
                logger.error("Error in trading loop (iteration %d): %s", iteration, exc, exc_info=True)

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
