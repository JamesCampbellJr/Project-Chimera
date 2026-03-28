# trading/pattern_detector.py
"""
Pattern Detector — identifies recurring profitable trading patterns from
historical wallet transaction data stored in the database.

Detected pattern types
-----------------------
* MOMENTUM_BUY   — token was bought by >= N profitable wallets in a short
                   window (i.e., early whale accumulation signal).
* QUICK_FLIP     — buy-to-sell round trip completed in < *threshold* seconds
                   with positive PnL (scalping / pump-and-dump riding).
* CONSISTENT_DEX — tokens traded exclusively via a single DEX tend to have
                   higher win-rates for specific wallets.

Each detected pattern is scored 0–1 and persisted to the `patterns` table.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import init_db, insert_pattern

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Thresholds (configurable via config.py)                                      #
# --------------------------------------------------------------------------- #
MOMENTUM_WALLET_THRESHOLD   = config.PATTERN_MOMENTUM_WALLETS   # ≥ N wallets buy same token
MOMENTUM_WINDOW_SECS        = config.PATTERN_MOMENTUM_WINDOW    # within this window (seconds)
QUICK_FLIP_MAX_SECS         = config.PATTERN_QUICK_FLIP_SECS    # max hold time for a flip
CONSISTENCY_MIN_WIN_RATE    = config.PATTERN_CONSISTENCY_WIN_RATE  # % e.g. 70.0


class PatternDetector:
    """
    Reads from the database and writes detected patterns back to it.
    """

    def __init__(self):
        self.conn = init_db()
        logger.info("PatternDetector initialised.")

    # ---------------------------------------------------------------------- #
    # Public                                                                   #
    # ---------------------------------------------------------------------- #

    def detect_all(self) -> list[dict]:
        """Run all detectors and return a combined list of pattern dicts."""
        patterns: list[dict] = []
        patterns.extend(self._detect_momentum_buys())
        patterns.extend(self._detect_quick_flips())
        patterns.extend(self._detect_consistent_dex_patterns())

        for p in patterns:
            insert_pattern(self.conn, p)

        logger.info("Detected and stored %d patterns.", len(patterns))
        return patterns

    def get_active_patterns(self, min_success_rate: float = 0.60) -> list[dict]:
        """Return patterns above the minimum success-rate threshold."""
        rows = self.conn.execute(
            """
            SELECT * FROM patterns
            WHERE success_rate >= ?
            ORDER BY success_rate DESC, sample_size DESC
            """,
            (min_success_rate,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------------- #
    # Detectors                                                                #
    # ---------------------------------------------------------------------- #

    def _detect_momentum_buys(self) -> list[dict]:
        """
        Find tokens that were bought by multiple profitable wallets within a
        short time window — a "smart money accumulation" signal.
        """
        logger.info("Detecting momentum-buy patterns …")

        # Only consider buys (wallet spent SOL/stable to acquire a token)
        QUOTE_TOKENS = {
            "So11111111111111111111111111111111111111112",   # WSOL
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", # USDC
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        }

        rows = self.conn.execute(
            """
            SELECT wt.token_out AS token, wt.block_time, wt.wallet_address,
                   wt.dex_program, w.win_rate
            FROM wallet_transactions wt
            JOIN wallets w ON w.address = wt.wallet_address
            WHERE wt.token_in IN ({})
              AND wt.block_time IS NOT NULL
              AND w.win_rate >= ?
            ORDER BY wt.token_out, wt.block_time
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            (*QUOTE_TOKENS, CONSISTENCY_MIN_WIN_RATE),
        ).fetchall()

        # Group by token
        by_token: dict = defaultdict(list)
        for r in rows:
            by_token[r["token"]].append(r)

        patterns: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()

        for token, buys in by_token.items():
            buys.sort(key=lambda x: x["block_time"])
            # Sliding-window check
            for i, buy in enumerate(buys):
                window = [buy]
                for j in range(i + 1, len(buys)):
                    if buys[j]["block_time"] - buy["block_time"] <= MOMENTUM_WINDOW_SECS:
                        window.append(buys[j])
                    else:
                        break
                unique_wallets = {b["wallet_address"] for b in window}
                if len(unique_wallets) >= MOMENTUM_WALLET_THRESHOLD:
                    dex = window[0]["dex_program"] if window else None
                    avg_win_rate = sum(b["win_rate"] for b in window) / len(window)
                    success_rate = min(avg_win_rate / 100.0, 1.0)
                    patterns.append({
                        "name": "MOMENTUM_BUY",
                        "description": (
                            f"{len(unique_wallets)} profitable wallets bought {token} "
                            f"within {MOMENTUM_WINDOW_SECS}s window"
                        ),
                        "token_address": token,
                        "dex_program": dex,
                        "min_hold_secs": None,
                        "max_hold_secs": None,
                        "avg_entry_usd": None,
                        "avg_roi_pct": None,
                        "sample_size": len(window),
                        "success_rate": success_rate,
                        "last_seen": now,
                    })
                    break  # One pattern per token per pass

        logger.info("Found %d momentum-buy patterns.", len(patterns))
        return patterns

    def _detect_quick_flips(self) -> list[dict]:
        """
        Identify tokens where the same wallet opened and closed a position
        quickly (< QUICK_FLIP_MAX_SECS) with a positive PnL.
        """
        logger.info("Detecting quick-flip patterns …")

        QUOTE_TOKENS = {
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        }

        rows = self.conn.execute(
            """
            SELECT wallet_address, token_out AS token, block_time,
                   amount_in, amount_out, price_usd, dex_program
            FROM wallet_transactions
            WHERE token_in IN ({})
              AND block_time IS NOT NULL
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            tuple(QUOTE_TOKENS),
        ).fetchall()

        # Build buy-events keyed by (wallet, token)
        buys: dict = defaultdict(list)
        for r in rows:
            buys[(r["wallet_address"], r["token"])].append(dict(r))

        sell_rows = self.conn.execute(
            """
            SELECT wallet_address, token_in AS token, block_time,
                   amount_in, amount_out, price_usd, dex_program
            FROM wallet_transactions
            WHERE token_out IN ({})
              AND block_time IS NOT NULL
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            tuple(QUOTE_TOKENS),
        ).fetchall()

        sells: dict = defaultdict(list)
        for r in sell_rows:
            sells[(r["wallet_address"], r["token"])].append(dict(r))

        patterns: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()
        flip_stats: dict = defaultdict(lambda: {"count": 0, "wins": 0, "rois": []})

        for key, buy_list in buys.items():
            wallet, token = key
            sell_list = sells.get(key, [])
            for buy in buy_list:
                for sell in sell_list:
                    if sell["block_time"] <= buy["block_time"]:
                        continue
                    hold = sell["block_time"] - buy["block_time"]
                    if hold > QUICK_FLIP_MAX_SECS:
                        continue
                    buy_usd  = buy.get("price_usd") or 0.0
                    sell_usd = sell.get("price_usd") or 0.0
                    roi = ((sell_usd - buy_usd) / buy_usd * 100.0) if buy_usd else 0.0
                    k = token
                    flip_stats[k]["count"] += 1
                    if roi > 0:
                        flip_stats[k]["wins"] += 1
                    flip_stats[k]["rois"].append(roi)

        for token, stats in flip_stats.items():
            n = stats["count"]
            if n < 3:
                continue
            success_rate = stats["wins"] / n
            avg_roi = sum(stats["rois"]) / len(stats["rois"])
            patterns.append({
                "name": "QUICK_FLIP",
                "description": (
                    f"Token {token} was quick-flipped {n} times "
                    f"(avg ROI {avg_roi:.1f}%)"
                ),
                "token_address": token,
                "dex_program": None,
                "min_hold_secs": 0,
                "max_hold_secs": QUICK_FLIP_MAX_SECS,
                "avg_entry_usd": None,
                "avg_roi_pct": avg_roi,
                "sample_size": n,
                "success_rate": success_rate,
                "last_seen": now,
            })

        logger.info("Found %d quick-flip patterns.", len(patterns))
        return patterns

    def _detect_consistent_dex_patterns(self) -> list[dict]:
        """
        Find (wallet, DEX) combinations where the wallet has a very high
        win-rate when trading on a specific DEX — indicates expertise.
        """
        logger.info("Detecting consistent-DEX patterns …")

        rows = self.conn.execute(
            """
            SELECT w.address, w.win_rate, wt.dex_program,
                   COUNT(*) AS tx_count
            FROM wallets w
            JOIN wallet_transactions wt ON wt.wallet_address = w.address
            WHERE w.win_rate >= ?
              AND wt.dex_program IS NOT NULL
            GROUP BY w.address, wt.dex_program
            HAVING tx_count >= ?
            ORDER BY w.win_rate DESC
            """,
            (CONSISTENCY_MIN_WIN_RATE, config.MIN_TRADES_FOR_RANKING),
        ).fetchall()

        patterns: list[dict] = []
        now = datetime.now(timezone.utc).isoformat()

        for r in rows:
            patterns.append({
                "name": "CONSISTENT_DEX",
                "description": (
                    f"Wallet {r['address']} has {r['win_rate']:.1f}% win-rate "
                    f"on {r['dex_program']} ({r['tx_count']} trades)"
                ),
                "token_address": None,
                "dex_program": r["dex_program"],
                "min_hold_secs": None,
                "max_hold_secs": None,
                "avg_entry_usd": None,
                "avg_roi_pct": None,
                "sample_size": r["tx_count"],
                "success_rate": r["win_rate"] / 100.0,
                "last_seen": now,
            })

        logger.info("Found %d consistent-DEX patterns.", len(patterns))
        return patterns
