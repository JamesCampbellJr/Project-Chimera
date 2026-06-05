# trading/wallet_analyzer.py
"""
Wallet Analyzer — scans the Solana blockchain for profitable wallets
and computes per-wallet statistics.

Strategy
--------
1. Start from a seed list of known "smart money" wallet addresses.
2. Fetch their recent swap transactions via the RPC client.
3. Compute win-rate, total PnL, and average trade ROI.
4. Flag suspicious wallet behaviour (wash trading, coordinated activity).
5. Compute a "smart money" score combining profitability and legitimacy.
6. Persist results to the database.
7. Optionally expand the seed list by following wallets that frequently
   traded alongside highly profitable ones (network-effect discovery).
"""

import logging
import time
from collections import defaultdict
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
    them by profitability and legitimacy.
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

        Returns a dict with keys: total_trades, winning_trades, total_pnl_usd,
        win_rate, smart_money_score, wash_trade_count
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

        # Compute wash trade indicators
        wash_count = self._detect_wallet_wash_trades(swaps)
        stats["wash_trade_count"] = wash_count

        # Compute smart money score
        stats["smart_money_score"] = self._compute_smart_money_score(stats, wash_count)

        update_wallet_stats(
            self.conn,
            address,
            stats["total_trades"],
            stats["winning_trades"],
            stats["total_pnl_usd"],
        )
        logger.info(
            "Wallet %s — trades: %d, win-rate: %.1f%%, PnL: $%.2f, "
            "smart_money: %.2f, wash_trades: %d",
            address,
            stats["total_trades"],
            stats["win_rate"],
            stats["total_pnl_usd"],
            stats["smart_money_score"],
            wash_count,
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

    def discover_related_wallets(self, address: str, limit: int = 10) -> list[str]:
        """
        Discover wallets that frequently trade the same tokens as *address*
        within similar time windows — potential smart money or coordinated actors.

        Returns a list of discovered wallet addresses (up to *limit*).
        """
        rows = self.conn.execute(
            """
            SELECT DISTINCT wt2.wallet_address
            FROM wallet_transactions wt1
            JOIN wallet_transactions wt2
              ON wt1.token_out = wt2.token_out
             AND wt1.wallet_address != wt2.wallet_address
             AND ABS(wt1.block_time - wt2.block_time) < 600
            WHERE wt1.wallet_address = ?
              AND wt1.block_time IS NOT NULL
              AND wt2.block_time IS NOT NULL
            LIMIT ?
            """,
            (address, limit),
        ).fetchall()

        discovered = [r["wallet_address"] for r in rows]
        for addr in discovered:
            upsert_wallet(self.conn, addr, label="discovered")
        if discovered:
            logger.info(
                "Discovered %d related wallets for %s.", len(discovered), address
            )
        return discovered

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

    def _detect_wallet_wash_trades(self, swaps: list[dict]) -> int:
        """
        Count wash-trade-like patterns in a single wallet's swaps:
        buying and selling the same token within a very short window
        with similar amounts.
        """
        # Group by token
        by_token: dict[str, list] = defaultdict(list)
        for swap in swaps:
            token = swap.get("token_out") or swap.get("token_in")
            if token:
                by_token[token].append(swap)

        wash_count = 0
        for token, token_swaps in by_token.items():
            sorted_swaps = sorted(
                token_swaps,
                key=lambda s: s.get("block_time") or 0,
            )
            for i in range(len(sorted_swaps) - 1):
                t1 = sorted_swaps[i].get("block_time") or 0
                t2 = sorted_swaps[i + 1].get("block_time") or 0
                if 0 < (t2 - t1) < config.WASH_TRADE_WINDOW_SECS:
                    amt1 = sorted_swaps[i].get("amount_out") or 0
                    amt2 = sorted_swaps[i + 1].get("amount_in") or 0
                    if amt1 > 0 and amt2 > 0:
                        ratio = min(amt1, amt2) / max(amt1, amt2)
                        if ratio > 0.90:
                            wash_count += 1

        return wash_count

    def _compute_smart_money_score(self, stats: dict, wash_count: int) -> float:
        """
        Compute a composite "smart money" score (0.0–1.0) based on:
        - Win rate (higher is better)
        - Total PnL (positive is better)
        - Number of trades (more data = more reliable)
        - Wash trade count (penalise suspicious activity)
        """
        win_rate = stats.get("win_rate", 0.0)
        total_trades = stats.get("total_trades", 0)
        pnl = stats.get("total_pnl_usd", 0.0)

        if total_trades == 0:
            return 0.0

        # Win rate component (0–0.4)
        wr_score = min(win_rate / 100.0, 1.0) * 0.4

        # Trade count reliability component (0–0.2)
        # More trades = more reliable signal, caps at 50 trades
        tc_score = min(total_trades / 50.0, 1.0) * 0.2

        # PnL component (0–0.3)
        # Positive PnL is good, scale logarithmically
        if pnl > 0:
            import math
            pnl_score = min(math.log10(pnl + 1) / 5.0, 1.0) * 0.3
        else:
            pnl_score = 0.0

        # Wash trade penalty (0–0.1 deducted)
        wash_penalty = min(wash_count * 0.02, 0.1)

        score = wr_score + tc_score + pnl_score - wash_penalty
        return max(0.0, min(1.0, score))

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
            "smart_money_score": 0.0,
            "wash_trade_count": 0,
        }
