# trading/trading_bot.py
"""
Solana Paper-Trading Bot — main orchestrator for the trading subsystem.

Lifecycle
---------
1. Seed / refresh wallet analysis (wallet_analyzer).
2. Detect / refresh patterns (pattern_detector).
3. For every active pattern above the confidence threshold, emit a buy
   signal to the paper trader.
4. On every subsequent iteration, re-price open positions (evaluate_open_positions).
5. Persist all state to SQLite and log a running summary.

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

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Coordinates the wallet analysis → pattern detection → paper trading loop.
    """

    def __init__(self):
        self.conn    = init_db()
        self.client  = SolanaClient()
        self.analyzer  = WalletAnalyzer(solana_client=self.client)
        self.detector  = PatternDetector()
        self.trader    = PaperTrader(solana_client=self.client)
        self._running  = False
        logger.info("TradingBot ready.")

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

                # 4. Issue buy signals for each active pattern
                for pattern in active:
                    signal = {
                        **pattern,
                        "signal_score": pattern.get("success_rate", 0.0),
                    }
                    await loop.run_in_executor(
                        None, self.trader.execute_buy_signal, signal
                    )

                # 5. Evaluate open positions
                closed = await loop.run_in_executor(
                    None, self.trader.evaluate_open_positions
                )
                if closed:
                    logger.info("Closed %d position(s) this iteration.", len(closed))

                # 6. Summary
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
