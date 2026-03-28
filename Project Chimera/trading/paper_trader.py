# trading/paper_trader.py
"""
Paper Trading Engine — executes simulated buy/sell orders using virtual
cash.  No real funds are ever at risk.

How it works
------------
1. The engine receives a trading *signal* — a pattern dict with a
   ``token_address``, ``signal_score``, etc.
2. It allocates a fraction of the virtual cash balance to each position,
   capped at ``PAPER_MAX_POSITION_USD``.
3. It immediately records the "buy" in the database.
4. On the next iteration, open positions are re-priced.  When a take-profit
   or stop-loss level is hit, the position is "sold" (closed).
5. All history is persisted so the bot can learn which patterns worked best.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import (
    init_db,
    get_virtual_cash,
    set_virtual_cash,
    record_paper_trade,
    get_open_paper_trades,
    close_paper_trade,
    upsert_portfolio_position,
    get_performance_summary,
)
from trading.solana_client import SolanaClient

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Manages a virtual paper-trading portfolio on Solana tokens.
    """

    def __init__(self, solana_client: Optional[SolanaClient] = None):
        self.client = solana_client or SolanaClient()
        self.conn   = init_db()
        logger.info(
            "PaperTrader initialised. Virtual cash: $%.2f",
            get_virtual_cash(self.conn),
        )

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def execute_buy_signal(self, signal: dict) -> Optional[int]:
        """
        Open a paper long position for the token identified in *signal*.

        Returns the new paper_trade row id, or None if the signal was skipped.
        """
        token_address = signal.get("token_address")
        if not token_address:
            logger.warning("Signal missing token_address — skipped.")
            return None

        score = float(signal.get("signal_score") or signal.get("success_rate") or 0)
        if score < config.MIN_SIGNAL_SCORE:
            logger.info(
                "Signal score %.2f below threshold %.2f — skipped.",
                score,
                config.MIN_SIGNAL_SCORE,
            )
            return None

        cash = get_virtual_cash(self.conn)
        if cash < config.PAPER_MIN_TRADE_USD:
            logger.warning(
                "Insufficient virtual cash ($%.2f) — skipping buy.", cash
            )
            return None

        # Position size: % of cash, capped at max
        position_usd = min(
            cash * config.PAPER_POSITION_SIZE_PCT / 100.0,
            config.PAPER_MAX_POSITION_USD,
        )
        position_usd = max(position_usd, config.PAPER_MIN_TRADE_USD)

        price = self.client.get_token_price_usd(token_address)
        token_qty = (position_usd / price) if price and price > 0 else None

        trade = {
            "pattern_id":     signal.get("id"),
            "token_address":  token_address,
            "token_symbol":   signal.get("token_symbol", "UNKNOWN"),
            "action":         "BUY",
            "virtual_amount": position_usd,
            "token_qty":      token_qty,
            "price_usd":      price,
            "pnl_usd":        0.0,
            "pnl_pct":        0.0,
            "outcome":        "OPEN",
            "signal_score":   score,
        }
        trade_id = record_paper_trade(self.conn, trade)

        # Deduct from virtual cash
        set_virtual_cash(self.conn, cash - position_usd)

        # Update portfolio
        upsert_portfolio_position(
            self.conn,
            {
                "token_address": token_address,
                "token_symbol":  trade["token_symbol"],
                "qty":           token_qty or 0.0,
                "avg_cost_usd":  price or 0.0,
                "current_price": price or 0.0,
                "unrealised_pnl": 0.0,
            },
        )

        logger.info(
            "PAPER BUY  | %s | $%.2f | price=%s | score=%.2f | id=%d",
            token_address, position_usd,
            f"${price:.4f}" if price else "unknown",
            score, trade_id,
        )
        return trade_id

    def evaluate_open_positions(self) -> list[dict]:
        """
        Re-price every open position and close any that have hit TP/SL.

        Returns a list of closed-trade dicts for this iteration.
        """
        open_trades = get_open_paper_trades(self.conn)
        if not open_trades:
            return []

        closed: list[dict] = []
        for trade in open_trades:
            token_address = trade["token_address"]
            entry_price   = trade.get("price_usd") or 0.0
            position_usd  = trade.get("virtual_amount") or 0.0
            token_qty     = trade.get("token_qty") or 0.0

            current_price = self.client.get_token_price_usd(token_address)
            if current_price is None or entry_price == 0:
                continue  # can't evaluate without a price

            pnl_pct = (current_price - entry_price) / entry_price * 100.0
            pnl_usd = (current_price - entry_price) * token_qty

            # Update portfolio with latest price
            upsert_portfolio_position(
                self.conn,
                {
                    "token_address": token_address,
                    "qty":           token_qty,
                    "avg_cost_usd":  entry_price,
                    "current_price": current_price,
                    "unrealised_pnl": pnl_usd,
                },
            )

            should_close = False
            outcome      = "OPEN"

            if pnl_pct >= config.PAPER_TAKE_PROFIT_PCT:
                should_close = True
                outcome      = "WIN"
                logger.info(
                    "PAPER SELL (TP) | %s | PnL=+$%.2f (+%.1f%%)",
                    token_address, pnl_usd, pnl_pct,
                )
            elif pnl_pct <= -config.PAPER_STOP_LOSS_PCT:
                should_close = True
                outcome      = "LOSS"
                logger.info(
                    "PAPER SELL (SL) | %s | PnL=-$%.2f (%.1f%%)",
                    token_address, abs(pnl_usd), pnl_pct,
                )

            if should_close:
                close_paper_trade(
                    self.conn,
                    trade["id"],
                    pnl_usd,
                    pnl_pct,
                    outcome,
                )
                # Return virtual cash + proceeds
                proceeds = position_usd + pnl_usd
                cash = get_virtual_cash(self.conn)
                set_virtual_cash(self.conn, cash + proceeds)

                trade["pnl_usd"] = pnl_usd
                trade["pnl_pct"] = pnl_pct
                trade["outcome"] = outcome
                closed.append(trade)

        return closed

    def print_summary(self) -> None:
        """Log a human-readable performance summary."""
        s = get_performance_summary(self.conn)
        logger.info(
            "=== Paper Trading Summary ===\n"
            "  Trades    : %d (wins=%d, losses=%d, open=%d)\n"
            "  Win rate  : %.1f%%\n"
            "  Total PnL : $%.2f\n"
            "  Avg ROI   : %.2f%%\n"
            "  Cash left : $%.2f",
            s.get("total_trades", 0),
            s.get("wins", 0) or 0,
            s.get("losses", 0) or 0,
            s.get("open_trades", 0) or 0,
            s.get("win_rate_pct", 0.0),
            s.get("total_pnl_usd", 0.0),
            s.get("avg_roi_pct", 0.0),
            s.get("virtual_cash_usd", 0.0),
        )

    def get_summary(self) -> dict:
        """Return the performance summary as a dict."""
        return get_performance_summary(self.conn)
