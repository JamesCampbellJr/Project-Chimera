# trading/fake_trade_detector.py
"""
Fake Trade & Manipulation Detector — identifies suspicious trading activity
to protect the bot from acting on manipulated market signals.

Detection strategies
--------------------
* WASH_TRADE      — same wallet buying and selling the same token rapidly
                   with no meaningful price change (self-trading for volume).
* PUMP_SCHEME     — coordinated buying by many new/low-activity wallets
                   in a short window, often followed by a single large sell.
* FAKE_VOLUME     — abnormal volume spikes that don't correspond to genuine
                   liquidity changes or news events.
* COORDINATED     — multiple wallets with similar creation times and
                   synchronised trading patterns (likely same entity).
* HONEYPOT        — tokens where buy transactions succeed but sells
                   consistently fail or have extreme slippage.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import init_db, insert_manipulation_flag, is_token_flagged

logger = logging.getLogger(__name__)


class FakeTradeDetector:
    """
    Analyses wallet transaction data to flag manipulative activity.
    """

    def __init__(self):
        self.conn = init_db()
        logger.info("FakeTradeDetector initialised.")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def scan_all(self) -> list[dict]:
        """Run all detection strategies and return combined flags."""
        flags: list[dict] = []
        flags.extend(self._detect_wash_trades())
        flags.extend(self._detect_pump_schemes())
        flags.extend(self._detect_fake_volume())
        flags.extend(self._detect_coordinated_wallets())
        flags.extend(self._detect_honeypots())

        logger.info("Fake trade scan complete: %d flags raised.", len(flags))
        return flags

    def is_safe_to_trade(self, token_address: str) -> tuple[bool, str]:
        """
        Check whether a token is safe to trade based on manipulation flags.

        Returns (is_safe: bool, reason: str).
        """
        if is_token_flagged(self.conn, token_address, min_severity=config.MANIPULATION_BLOCK_SEVERITY):
            return False, f"Token {token_address} has active manipulation flags"
        return True, "No manipulation flags detected"

    # ---------------------------------------------------------------------- #
    # Detection strategies                                                     #
    # ---------------------------------------------------------------------- #

    def _detect_wash_trades(self) -> list[dict]:
        """
        Detect wash trading: same wallet buying then selling the same token
        within a very short window (< 60 seconds) with similar amounts.
        """
        logger.info("Scanning for wash trades …")
        flags: list[dict] = []

        rows = self.conn.execute(
            """
            SELECT wallet_address, token_out AS token, block_time,
                   amount_in, amount_out, signature
            FROM wallet_transactions
            WHERE block_time IS NOT NULL
            ORDER BY wallet_address, token_out, block_time
            """
        ).fetchall()

        # Group by (wallet, token)
        by_key: dict = defaultdict(list)
        for r in rows:
            by_key[(r["wallet_address"], r["token"])].append(dict(r))

        for (wallet, token), txs in by_key.items():
            if not token or len(txs) < 2:
                continue

            for i in range(len(txs) - 1):
                for j in range(i + 1, min(i + 5, len(txs))):
                    time_diff = abs((txs[j].get("block_time") or 0) - (txs[i].get("block_time") or 0))

                    if time_diff > config.WASH_TRADE_WINDOW_SECS:
                        break

                    # Check if amounts are suspiciously similar (within 5%)
                    amt_i = txs[i].get("amount_out") or 0
                    amt_j = txs[j].get("amount_out") or 0
                    if amt_i > 0 and amt_j > 0:
                        ratio = min(amt_i, amt_j) / max(amt_i, amt_j)
                        if ratio > 0.95:
                            severity = 0.8
                            evidence = json.dumps({
                                "type": "WASH_TRADE",
                                "wallet": wallet,
                                "token": token,
                                "time_diff_secs": time_diff,
                                "amount_ratio": round(ratio, 4),
                                "signatures": [txs[i].get("signature"), txs[j].get("signature")],
                            })
                            flag = {
                                "token_address": token,
                                "flag_type": "WASH_TRADE",
                                "severity": severity,
                                "evidence": evidence,
                                "wallet_addresses": wallet,
                            }
                            insert_manipulation_flag(self.conn, flag)
                            flags.append(flag)

        logger.info("Found %d wash trade flags.", len(flags))
        return flags

    def _detect_pump_schemes(self) -> list[dict]:
        """
        Detect coordinated pump schemes: many wallets buying the same
        token in a short window, especially if those wallets have low
        historical activity.
        """
        logger.info("Scanning for pump schemes …")
        flags: list[dict] = []

        QUOTE_TOKENS = {
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        }

        rows = self.conn.execute(
            """
            SELECT wt.wallet_address, wt.token_out AS token, wt.block_time,
                   w.total_trades
            FROM wallet_transactions wt
            JOIN wallets w ON w.address = wt.wallet_address
            WHERE wt.token_in IN ({})
              AND wt.block_time IS NOT NULL
            ORDER BY wt.token_out, wt.block_time
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            tuple(QUOTE_TOKENS),
        ).fetchall()

        # Group buys by token
        by_token: dict = defaultdict(list)
        for r in rows:
            by_token[r["token"]].append(dict(r))

        for token, buys in by_token.items():
            if len(buys) < config.PUMP_MIN_WALLETS:
                continue

            buys.sort(key=lambda x: x.get("block_time") or 0)

            # Sliding window
            for i in range(len(buys)):
                window: list[dict] = [buys[i]]
                for j in range(i + 1, len(buys)):
                    time_diff = (buys[j].get("block_time") or 0) - (buys[i].get("block_time") or 0)
                    if time_diff <= config.PUMP_WINDOW_SECS:
                        window.append(buys[j])
                    else:
                        break

                unique_wallets = {b["wallet_address"] for b in window}
                if len(unique_wallets) < config.PUMP_MIN_WALLETS:
                    continue

                # Check if most wallets have low activity (< 10 trades)
                low_activity = sum(
                    1 for b in window
                    if (b.get("total_trades") or 0) < 10
                )
                low_activity_pct = low_activity / len(window) * 100

                if low_activity_pct >= 60:
                    severity = min(0.5 + low_activity_pct / 200, 1.0)
                    evidence = json.dumps({
                        "type": "PUMP_SCHEME",
                        "token": token,
                        "wallet_count": len(unique_wallets),
                        "low_activity_pct": round(low_activity_pct, 1),
                        "window_secs": config.PUMP_WINDOW_SECS,
                    })
                    wallets_str = ",".join(list(unique_wallets)[:10])
                    flag = {
                        "token_address": token,
                        "flag_type": "PUMP_SCHEME",
                        "severity": severity,
                        "evidence": evidence,
                        "wallet_addresses": wallets_str,
                    }
                    insert_manipulation_flag(self.conn, flag)
                    flags.append(flag)
                    break  # one flag per token

        logger.info("Found %d pump scheme flags.", len(flags))
        return flags

    def _detect_fake_volume(self) -> list[dict]:
        """
        Detect fake volume: tokens with transaction counts far above
        the median for tracked tokens, without corresponding unique
        wallet diversity.
        """
        logger.info("Scanning for fake volume …")
        flags: list[dict] = []

        rows = self.conn.execute(
            """
            SELECT token_out AS token,
                   COUNT(*) AS tx_count,
                   COUNT(DISTINCT wallet_address) AS unique_wallets
            FROM wallet_transactions
            WHERE token_out IS NOT NULL
            GROUP BY token_out
            HAVING tx_count >= 10
            ORDER BY tx_count DESC
            """
        ).fetchall()

        if not rows:
            return flags

        # Compute median tx_count
        tx_counts = sorted(r["tx_count"] for r in rows)
        median_tx = tx_counts[len(tx_counts) // 2] if tx_counts else 1

        for r in rows:
            tx_count = r["tx_count"]
            unique_wallets = r["unique_wallets"]
            token = r["token"]

            # Flag if volume is 5x median but unique wallets are low
            if tx_count > median_tx * 5 and unique_wallets < tx_count * 0.3:
                wallet_ratio = unique_wallets / tx_count if tx_count > 0 else 0
                severity = min(0.4 + (1.0 - wallet_ratio) * 0.5, 1.0)

                evidence = json.dumps({
                    "type": "FAKE_VOLUME",
                    "token": token,
                    "tx_count": tx_count,
                    "unique_wallets": unique_wallets,
                    "median_tx": median_tx,
                    "wallet_ratio": round(wallet_ratio, 4),
                })
                flag = {
                    "token_address": token,
                    "flag_type": "FAKE_VOLUME",
                    "severity": severity,
                    "evidence": evidence,
                    "wallet_addresses": None,
                }
                insert_manipulation_flag(self.conn, flag)
                flags.append(flag)

        logger.info("Found %d fake volume flags.", len(flags))
        return flags

    def _detect_coordinated_wallets(self) -> list[dict]:
        """
        Detect coordinated wallet groups: wallets that consistently
        trade the same tokens within seconds of each other.
        """
        logger.info("Scanning for coordinated wallets …")
        flags: list[dict] = []

        QUOTE_TOKENS = {
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        }

        rows = self.conn.execute(
            """
            SELECT wallet_address, token_out AS token, block_time
            FROM wallet_transactions
            WHERE token_in IN ({})
              AND block_time IS NOT NULL
            ORDER BY token_out, block_time
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            tuple(QUOTE_TOKENS),
        ).fetchall()

        # Group by token
        by_token: dict = defaultdict(list)
        for r in rows:
            by_token[r["token"]].append(dict(r))

        # Track wallet pair co-occurrences
        pair_counts: dict = defaultdict(int)

        for token, buys in by_token.items():
            buys.sort(key=lambda x: x.get("block_time") or 0)
            for i in range(len(buys)):
                for j in range(i + 1, len(buys)):
                    time_diff = abs((buys[j].get("block_time") or 0) - (buys[i].get("block_time") or 0))
                    if time_diff > 30:
                        break
                    w1 = buys[i]["wallet_address"]
                    w2 = buys[j]["wallet_address"]
                    if w1 != w2:
                        pair_key = tuple(sorted([w1, w2]))
                        pair_counts[pair_key] += 1

        # Flag pairs that co-trade frequently
        for (w1, w2), count in pair_counts.items():
            if count >= config.COORDINATED_MIN_COOCCURRENCES:
                severity = min(0.5 + count * 0.1, 1.0)
                evidence = json.dumps({
                    "type": "COORDINATED",
                    "wallet_1": w1,
                    "wallet_2": w2,
                    "co_occurrences": count,
                })
                # Flag all tokens these wallets have co-traded
                flag = {
                    "token_address": None,
                    "flag_type": "COORDINATED",
                    "severity": severity,
                    "evidence": evidence,
                    "wallet_addresses": f"{w1},{w2}",
                }
                insert_manipulation_flag(self.conn, flag)
                flags.append(flag)

        logger.info("Found %d coordinated wallet flags.", len(flags))
        return flags

    def _detect_honeypots(self) -> list[dict]:
        """
        Detect honeypot tokens: tokens that have many buy transactions
        but very few or no sell transactions across all tracked wallets.
        """
        logger.info("Scanning for honeypot tokens …")
        flags: list[dict] = []

        QUOTE_TOKENS = (
            "So11111111111111111111111111111111111111112",
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
        )

        # Count buys per token (spending quote to get token)
        buy_rows = self.conn.execute(
            """
            SELECT token_out AS token, COUNT(*) AS buy_count
            FROM wallet_transactions
            WHERE token_in IN ({})
              AND token_out IS NOT NULL
            GROUP BY token_out
            HAVING buy_count >= 5
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            QUOTE_TOKENS,
        ).fetchall()

        # Count sells per token (spending token to get quote)
        sell_rows = self.conn.execute(
            """
            SELECT token_in AS token, COUNT(*) AS sell_count
            FROM wallet_transactions
            WHERE token_out IN ({})
              AND token_in IS NOT NULL
            GROUP BY token_in
            """.format(",".join("?" * len(QUOTE_TOKENS))),
            QUOTE_TOKENS,
        ).fetchall()

        sell_counts = {r["token"]: r["sell_count"] for r in sell_rows}

        for r in buy_rows:
            token = r["token"]
            buy_count = r["buy_count"]
            sell_count = sell_counts.get(token, 0)

            # Honeypot indicator: many buys, very few sells
            if buy_count >= 5 and sell_count <= buy_count * 0.1:
                sell_ratio = sell_count / buy_count if buy_count > 0 else 0
                severity = min(0.7 + (1.0 - sell_ratio) * 0.3, 1.0)

                evidence = json.dumps({
                    "type": "HONEYPOT",
                    "token": token,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "sell_ratio": round(sell_ratio, 4),
                })
                flag = {
                    "token_address": token,
                    "flag_type": "HONEYPOT",
                    "severity": severity,
                    "evidence": evidence,
                    "wallet_addresses": None,
                }
                insert_manipulation_flag(self.conn, flag)
                flags.append(flag)

        logger.info("Found %d honeypot flags.", len(flags))
        return flags
