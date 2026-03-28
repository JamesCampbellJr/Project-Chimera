# trading/wallet_analyzer.py
"""
Wallet Analyzer — scans the Solana blockchain for profitable wallets
and computes per-wallet statistics.

Strategy
--------
1. Start from a seed list of known "smart money" wallet addresses.
2. Fetch their recent swap transactions via the RPC client.
3. Compute win-rate, total PnL, and average trade ROI.
4. Persist results to the database.
5. Optionally expand the seed list by following wallets that frequently
   traded alongside highly profitable ones (network-effect discovery).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import (
    init_db,
    upsert_wallet,
    update_wallet_stats,
    insert_transaction,
)
from trading.solana_client import SolanaClient

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Seed wallets — these are *example* well-known DeFi power-users / traders.   #
# Replace or extend this list as you discover better candidates.               #
# --------------------------------------------------------------------------- #
SEED_WALLETS: list[str] = config.SEED_WALLETS


class WalletAnalyzer:
    """
    Analyses on-chain swap activity for a set of tracked wallets and scores
    them by profitability.
    """

    def __init__(self, solana_client: Optional[SolanaClient] = None):
        self.client = solana_client or SolanaClient()
        self.conn = init_db()
        logger.info("WalletAnalyzer initialised.")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def seed_wallets(self) -> None:
        """Register all seed wallets in the database."""
        for address in SEED_WALLETS:
            upsert_wallet(self.conn, address, label="seed")
        logger.info("Seeded %d wallets into the database.", len(SEED_WALLETS))

    def analyse_wallet(self, address: str, tx_limit: int = 100) -> dict:
        """
        Fetch recent swaps for *address*, persist them, and compute stats.

        Returns a dict with keys: total_trades, winning_trades, total_pnl_usd, win_rate
        """
        logger.info("Analysing wallet %s …", address)
        upsert_wallet(self.conn, address)

        signatures = self.client.get_transaction_signatures(address, limit=tx_limit)
        if not signatures:
            logger.info("No transactions found for %s", address)
            return self._empty_stats()

        sol_price = self.client.get_sol_price_usd()
        swaps: list[dict] = []

        for sig_info in signatures:
            signature = sig_info.get("signature")
            if not signature:
                continue
            # Rate-limit to stay within free RPC quota
            time.sleep(config.RPC_RATE_LIMIT_DELAY)

            tx = self.client.get_transaction(signature)
            swap = self.client.parse_swap_from_transaction(tx, address)
            if swap is None:
                continue

            # Estimate USD value of the position
            swap["price_usd"] = self._estimate_usd_value(
                swap, sol_price
            )
            swaps.append(swap)
            insert_transaction(self.conn, swap)

        stats = self._compute_stats(swaps, sol_price)
        update_wallet_stats(
            self.conn,
            address,
            stats["total_trades"],
            stats["winning_trades"],
            stats["total_pnl_usd"],
        )
        logger.info(
            "Wallet %s — trades: %d, win-rate: %.1f%%, PnL: $%.2f",
            address,
            stats["total_trades"],
            stats["win_rate"],
            stats["total_pnl_usd"],
        )
        return stats

    def analyse_all_tracked(self) -> list[dict]:
        """Run analyse_wallet for every tracked wallet in the DB."""
        rows = self.conn.execute(
            "SELECT address FROM wallets WHERE is_tracked = 1 OR label = 'seed'"
        ).fetchall()
        results = []
        for row in rows:
            stats = self.analyse_wallet(row["address"])
            stats["address"] = row["address"]
            results.append(stats)
        return results

    def get_top_wallets(self, min_win_rate: float = 60.0, top_n: int = 20) -> list[dict]:
        """
        Return the top-*N* wallets sorted by PnL that meet the *min_win_rate*
        threshold (percentage, 0–100).
        """
        rows = self.conn.execute(
            """
            SELECT address, label, total_trades, winning_trades,
                   total_pnl_usd, win_rate
            FROM wallets
            WHERE win_rate >= ? AND total_trades >= ?
            ORDER BY total_pnl_usd DESC
            LIMIT ?
            """,
            (min_win_rate, config.MIN_TRADES_FOR_RANKING, top_n),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _compute_stats(self, swaps: list[dict], sol_price: float) -> dict:
        """
        Pair BUY and SELL swaps for the same token and compute aggregate stats.
        """
        # Group swaps by token_out (token received = potential "buy")
        buys: dict[str, list] = {}
        sells: dict[str, list] = {}

        for swap in swaps:
            t_in = swap.get("token_in", "")
            t_out = swap.get("token_out", "")
            if t_out and t_in:
                # Heuristic: if spending SOL/USDC to get a token → BUY
                if t_in in (
                    "So11111111111111111111111111111111111111112",
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                ):
                    buys.setdefault(t_out, []).append(swap)
                else:
                    sells.setdefault(t_in, []).append(swap)

        total_trades = 0
        winning_trades = 0
        total_pnl_usd = 0.0

        all_tokens = set(buys) | set(sells)
        for token in all_tokens:
            token_buys = buys.get(token, [])
            token_sells = sells.get(token, [])

            if not token_buys or not token_sells:
                continue

            cost = sum(s.get("price_usd") or 0.0 for s in token_buys)
            revenue = sum(s.get("price_usd") or 0.0 for s in token_sells)
            pnl = revenue - cost
            total_pnl_usd += pnl
            total_trades += 1
            if pnl > 0:
                winning_trades += 1

        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "total_pnl_usd": total_pnl_usd,
            "win_rate": win_rate,
        }

    def _estimate_usd_value(self, swap: dict, sol_price: float) -> Optional[float]:
        """
        Estimate the USD value of a swap by multiplying SOL amounts by sol_price.
        Returns None if the value cannot be determined.
        """
        WSOL = "So11111111111111111111111111111111111111112"
        if swap.get("token_in") == WSOL and swap.get("amount_in"):
            return swap["amount_in"] * sol_price
        if swap.get("token_out") == WSOL and swap.get("amount_out"):
            return swap["amount_out"] * sol_price
        return None

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl_usd": 0.0,
            "win_rate": 0.0,
        }
