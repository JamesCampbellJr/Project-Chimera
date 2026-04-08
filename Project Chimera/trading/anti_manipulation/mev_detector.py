"""
MEV (Maximal Extractable Value) bot detection for Solana.

Identifies sandwich attacks, front-running, and automated trading
bots by analysing on-chain transaction patterns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from trading.solana_client import DEX_PROGRAMS, SolanaClient

logger = logging.getLogger(__name__)


# ====================================================================
# Data classes
# ====================================================================

@dataclass
class MEVActivity:
    """A single detected MEV event."""

    tx_signature: str
    mev_type: str
    profit_estimate: float
    victim_address: str
    block_slot: int


@dataclass
class BotClassification:
    """Result of bot-behaviour classification for a wallet."""

    address: str
    is_bot: bool
    bot_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


# ====================================================================
# MEVDetector
# ====================================================================

class MEVDetector:
    """
    Detects sandwich attacks, front-running, and automated bot
    activity in Solana transactions.
    """

    def __init__(self, solana_client: Optional[SolanaClient] = None) -> None:
        """
        Parameters
        ----------
        solana_client : SolanaClient, optional
            An initialised :class:`SolanaClient`.  If ``None``, a fresh
            instance is created from the default config.
        """
        self.client = solana_client or SolanaClient()

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_swap_info(tx: dict) -> Optional[dict]:
        """
        Extract lightweight swap information from a decoded transaction.

        Returns a dict with keys ``address``, ``token``, ``direction``
        (``"buy"`` or ``"sell"``), ``amount``, ``slot``, ``signature``,
        and ``priority_fee``, or ``None`` if the transaction is not a
        recognisable DEX swap.
        """
        if not tx:
            return None

        meta = tx.get("meta") or {}
        if meta.get("err"):
            return None

        tx_body = tx.get("transaction") or {}
        message = tx_body.get("message") or {}
        account_keys = []
        for acct in message.get("accountKeys", []):
            if isinstance(acct, dict):
                account_keys.append(acct.get("pubkey", ""))
            else:
                account_keys.append(str(acct))

        # Must involve a known DEX
        dex_hit = False
        for prog_id in DEX_PROGRAMS:
            if prog_id in account_keys:
                dex_hit = True
                break
        if not dex_hit:
            return None

        # Fee payer (first signer)
        fee_payer = account_keys[0] if account_keys else ""

        # Determine direction from token balance changes
        pre_balances = {
            b.get("accountIndex"): b for b in (meta.get("preTokenBalances") or [])
        }
        post_balances = {
            b.get("accountIndex"): b for b in (meta.get("postTokenBalances") or [])
        }

        token: Optional[str] = None
        direction: Optional[str] = None
        amount = 0.0

        for idx in set(pre_balances) | set(post_balances):
            pre = pre_balances.get(idx, {})
            post = post_balances.get(idx, {})
            owner = post.get("owner") or pre.get("owner")
            if owner != fee_payer:
                continue

            mint = post.get("mint") or pre.get("mint")
            pre_amt = float(
                (pre.get("uiTokenAmount") or {}).get("uiAmount") or 0
            )
            post_amt = float(
                (post.get("uiTokenAmount") or {}).get("uiAmount") or 0
            )
            delta = post_amt - pre_amt
            if delta > 0:
                token = mint
                direction = "buy"
                amount = delta
            elif delta < 0:
                token = mint
                direction = "sell"
                amount = abs(delta)

        if token is None or direction is None:
            return None

        # Priority fee: difference between total fee and base fee
        fee = meta.get("fee", 0)
        priority_fee = max(fee - 5000, 0)  # base fee is 5000 lamports

        signatures = tx_body.get("signatures", [])
        sig = signatures[0] if signatures else ""

        return {
            "address": fee_payer,
            "token": token,
            "direction": direction,
            "amount": amount,
            "slot": tx.get("slot", 0),
            "signature": sig,
            "priority_fee": priority_fee,
        }

    # ------------------------------------------------------------------ #
    # Sandwich attack detection                                           #
    # ------------------------------------------------------------------ #

    def detect_sandwich_attacks(
        self, block_txs: list[dict]
    ) -> list[MEVActivity]:
        """
        Scan ordered transactions within a block for sandwich-attack
        patterns.

        The classic sandwich pattern is:

        1. Address **A** buys token **X** (front-run).
        2. Address **B** buys token **X** (victim).
        3. Address **A** sells token **X** (back-run / profit-taking).

        All three transactions must appear in order within the same block.

        Parameters
        ----------
        block_txs : list[dict]
            Decoded transactions in block order.

        Returns
        -------
        list[MEVActivity]
            One entry per detected sandwich attack.
        """
        activities: list[MEVActivity] = []

        try:
            swap_infos: list[Optional[dict]] = [
                self._extract_swap_info(tx) for tx in block_txs
            ]

            n = len(swap_infos)
            for i in range(n - 2):
                front = swap_infos[i]
                if front is None or front["direction"] != "buy":
                    continue

                for j in range(i + 1, n - 1):
                    victim = swap_infos[j]
                    if victim is None:
                        continue
                    if victim["token"] != front["token"]:
                        continue
                    if victim["direction"] != "buy":
                        continue
                    if victim["address"] == front["address"]:
                        continue  # same wallet, not a sandwich

                    for k in range(j + 1, n):
                        back = swap_infos[k]
                        if back is None:
                            continue
                        if back["address"] != front["address"]:
                            continue
                        if back["token"] != front["token"]:
                            continue
                        if back["direction"] != "sell":
                            continue

                        profit = back["amount"] - front["amount"]

                        activities.append(
                            MEVActivity(
                                tx_signature=back["signature"],
                                mev_type="sandwich",
                                profit_estimate=profit,
                                victim_address=victim["address"],
                                block_slot=front.get("slot", 0),
                            )
                        )
                        logger.info(
                            "Sandwich detected: attacker=%s victim=%s "
                            "token=%s profit=%.4f",
                            front["address"],
                            victim["address"],
                            front["token"],
                            profit,
                        )
                        break  # only match the first back-run for this pair
                    break  # only match the first victim for this front-run

        except Exception as exc:  # noqa: BLE001
            logger.error("Sandwich detection failed: %s", exc)

        return activities

    # ------------------------------------------------------------------ #
    # Front-running detection                                             #
    # ------------------------------------------------------------------ #

    def detect_frontrunning(
        self,
        pending_txs: list[dict],
        confirmed_txs: list[dict],
    ) -> list[MEVActivity]:
        """
        Compare pending (mempool) ordering with confirmed ordering
        to identify probable front-running.

        A transaction is flagged when:
        * It moved to an *earlier* position relative to the pending order.
        * It trades the same token as a transaction that was pushed back.
        * It pays a higher priority fee.

        Parameters
        ----------
        pending_txs : list[dict]
            Transactions as seen in the mempool / pre-confirmation.
        confirmed_txs : list[dict]
            The same transactions in their confirmed (on-chain) order.

        Returns
        -------
        list[MEVActivity]
            Detected front-running events.
        """
        activities: list[MEVActivity] = []

        try:
            pending_swaps = [self._extract_swap_info(tx) for tx in pending_txs]
            confirmed_swaps = [self._extract_swap_info(tx) for tx in confirmed_txs]

            # Build position maps keyed by signature
            pending_pos: dict[str, int] = {}
            for idx, s in enumerate(pending_swaps):
                if s is not None:
                    pending_pos[s["signature"]] = idx

            confirmed_pos: dict[str, int] = {}
            for idx, s in enumerate(confirmed_swaps):
                if s is not None:
                    confirmed_pos[s["signature"]] = idx

            # Map signature → swap info for quick lookup
            swap_by_sig: dict[str, dict] = {}
            for s in pending_swaps:
                if s is not None:
                    swap_by_sig[s["signature"]] = s
            for s in confirmed_swaps:
                if s is not None:
                    swap_by_sig.setdefault(s["signature"], s)

            for sig, conf_idx in confirmed_pos.items():
                pend_idx = pending_pos.get(sig)
                if pend_idx is None:
                    continue
                if conf_idx >= pend_idx:
                    continue  # did not move forward

                suspect = swap_by_sig.get(sig)
                if suspect is None:
                    continue

                # Look for a victim that was pushed back
                for other_sig, other_pend_idx in pending_pos.items():
                    if other_sig == sig:
                        continue
                    other_conf_idx = confirmed_pos.get(other_sig)
                    if other_conf_idx is None:
                        continue
                    if other_conf_idx <= other_pend_idx:
                        continue  # not pushed back

                    other_swap = swap_by_sig.get(other_sig)
                    if other_swap is None:
                        continue

                    # Must involve the same token
                    if other_swap["token"] != suspect["token"]:
                        continue

                    # Suspect should have a higher priority fee
                    if suspect["priority_fee"] <= other_swap["priority_fee"]:
                        continue

                    activities.append(
                        MEVActivity(
                            tx_signature=sig,
                            mev_type="frontrun",
                            profit_estimate=0.0,
                            victim_address=other_swap["address"],
                            block_slot=suspect.get("slot", 0),
                        )
                    )
                    logger.info(
                        "Front-run detected: suspect=%s (fee=%d) "
                        "pushed victim=%s (fee=%d) on token=%s",
                        suspect["address"],
                        suspect["priority_fee"],
                        other_swap["address"],
                        other_swap["priority_fee"],
                        suspect["token"],
                    )
                    break  # one victim match is sufficient

        except Exception as exc:  # noqa: BLE001
            logger.error("Front-running detection failed: %s", exc)

        return activities

    # ------------------------------------------------------------------ #
    # Bot classification                                                  #
    # ------------------------------------------------------------------ #

    def classify_bot(
        self, address: str, transactions: list[dict]
    ) -> BotClassification:
        """
        Analyse a wallet's transaction history to determine whether it
        is likely an automated trading bot.

        Signals checked:

        1. **Transaction frequency** — >100 txs/hour is suspicious.
        2. **Priority-fee patterns** — consistently elevated fees.
        3. **Multi-DEX arbitrage** — using 3+ DEX programs in a short
           window.
        4. **Round-trip time** — buy→sell of the same token in <2 blocks.

        Parameters
        ----------
        address : str
            Wallet address to classify.
        transactions : list[dict]
            Recent decoded transactions for this address.

        Returns
        -------
        BotClassification
        """
        evidence: list[str] = []
        confidence = 0.0

        try:
            # ---------------------------------------------------------- #
            # Signal 1: transaction frequency                             #
            # ---------------------------------------------------------- #
            timestamps = sorted(
                tx.get("blockTime", 0)
                for tx in transactions
                if tx.get("blockTime")
            )

            if len(timestamps) >= 2:
                span_hours = max(
                    (timestamps[-1] - timestamps[0]) / 3600.0, 1 / 3600.0
                )
                tx_per_hour = len(timestamps) / span_hours
                if tx_per_hour > 100:
                    confidence += 0.30
                    evidence.append(
                        f"high_tx_frequency: {tx_per_hour:.0f} tx/h"
                    )

            # ---------------------------------------------------------- #
            # Signal 2: priority fee patterns                             #
            # ---------------------------------------------------------- #
            fees: list[int] = []
            for tx in transactions:
                meta = tx.get("meta") or {}
                fee = meta.get("fee", 0)
                priority = max(fee - 5000, 0)
                fees.append(priority)

            if fees:
                avg_fee = sum(fees) / len(fees)
                high_fee_count = sum(1 for f in fees if f > 10_000)
                high_fee_ratio = high_fee_count / len(fees)
                if high_fee_ratio > 0.5:
                    confidence += 0.20
                    evidence.append(
                        f"consistent_high_fees: avg={avg_fee:.0f} lamports, "
                        f"{high_fee_ratio:.0%} above threshold"
                    )

            # ---------------------------------------------------------- #
            # Signal 3: multi-DEX arbitrage                               #
            # ---------------------------------------------------------- #
            dex_set: set[str] = set()
            for tx in transactions:
                tx_body = tx.get("transaction") or {}
                message = tx_body.get("message") or {}
                for acct in message.get("accountKeys", []):
                    key = acct.get("pubkey", "") if isinstance(acct, dict) else str(acct)
                    if key in DEX_PROGRAMS:
                        dex_set.add(DEX_PROGRAMS[key])

            if len(dex_set) >= 3:
                confidence += 0.25
                evidence.append(
                    f"multi_dex_usage: {sorted(dex_set)}"
                )

            # ---------------------------------------------------------- #
            # Signal 4: round-trip time (buy→sell < 2 blocks)             #
            # ---------------------------------------------------------- #
            swap_events: list[dict] = []
            for tx in transactions:
                info = self._extract_swap_info(tx)
                if info is not None:
                    swap_events.append(info)

            # Group swaps by token
            token_swaps: dict[str, list[dict]] = {}
            for sw in swap_events:
                token_swaps.setdefault(sw["token"], []).append(sw)

            round_trips = 0
            for token, swaps in token_swaps.items():
                buys = sorted(
                    (s for s in swaps if s["direction"] == "buy"),
                    key=lambda s: s["slot"],
                )
                sells = sorted(
                    (s for s in swaps if s["direction"] == "sell"),
                    key=lambda s: s["slot"],
                )
                sell_idx = 0
                for buy in buys:
                    while sell_idx < len(sells) and sells[sell_idx]["slot"] < buy["slot"]:
                        sell_idx += 1
                    if sell_idx < len(sells):
                        slot_diff = sells[sell_idx]["slot"] - buy["slot"]
                        if 0 <= slot_diff <= 2:
                            round_trips += 1
                            sell_idx += 1

            if round_trips >= 2:
                confidence += 0.25
                evidence.append(
                    f"fast_round_trips: {round_trips} buy-sell pairs within 2 blocks"
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("Bot classification failed for %s: %s", address, exc)

        confidence = min(confidence, 1.0)
        is_bot = confidence >= 0.50

        # Determine bot type
        if not is_bot:
            bot_type = "none"
        elif "multi_dex_usage" in " ".join(evidence):
            bot_type = "arbitrage"
        elif "fast_round_trips" in " ".join(evidence):
            bot_type = "scalper"
        elif "high_tx_frequency" in " ".join(evidence):
            bot_type = "high_frequency"
        else:
            bot_type = "generic"

        return BotClassification(
            address=address,
            is_bot=is_bot,
            bot_type=bot_type,
            confidence=round(confidence, 4),
            evidence=evidence,
        )

    # ------------------------------------------------------------------ #
    # Token-level MEV risk score                                          #
    # ------------------------------------------------------------------ #

    def get_mev_risk_score(self, token: str) -> dict:
        """
        Compute an aggregate MEV-risk score for *token*.

        Fetches recent transactions and checks for sandwich and
        front-running patterns.

        Parameters
        ----------
        token : str
            SPL token mint address.

        Returns
        -------
        dict
            Keys: ``risk_score`` (0–1), ``sandwich_risk``,
            ``frontrun_risk``, ``bot_activity_level``, ``details``.
        """
        result: dict[str, Any] = {
            "risk_score": 0.0,
            "sandwich_risk": 0.0,
            "frontrun_risk": 0.0,
            "bot_activity_level": "low",
            "details": {},
        }

        try:
            signatures = self.client.get_transaction_signatures(token, limit=100)
            if not signatures:
                return result

            txs: list[dict] = []
            for sig_info in signatures:
                sig = sig_info.get("signature")
                if not sig:
                    continue
                tx = self.client.get_transaction(sig)
                if tx is not None:
                    txs.append(tx)

            if not txs:
                return result

            # Sandwich detection across the collected transactions
            sandwiches = self.detect_sandwich_attacks(txs)
            sandwich_risk = min(len(sandwiches) / 5.0, 1.0)

            # Unique addresses for bot classification
            addresses: set[str] = set()
            for tx in txs:
                keys = (
                    tx.get("transaction", {})
                    .get("message", {})
                    .get("accountKeys", [])
                )
                for acct in keys:
                    key = acct.get("pubkey", "") if isinstance(acct, dict) else str(acct)
                    if key:
                        addresses.add(key)

            bot_count = 0
            total_checked = 0
            for addr in list(addresses)[:20]:  # cap to avoid excessive RPC
                addr_sigs = self.client.get_transaction_signatures(addr, limit=50)
                addr_txs: list[dict] = []
                for s in addr_sigs[:20]:
                    t = self.client.get_transaction(s.get("signature", ""))
                    if t is not None:
                        addr_txs.append(t)
                if addr_txs:
                    classification = self.classify_bot(addr, addr_txs)
                    if classification.is_bot:
                        bot_count += 1
                    total_checked += 1

            bot_ratio = bot_count / max(total_checked, 1)

            if bot_ratio > 0.5:
                bot_level = "critical"
            elif bot_ratio > 0.3:
                bot_level = "high"
            elif bot_ratio > 0.1:
                bot_level = "moderate"
            else:
                bot_level = "low"

            frontrun_risk = min(bot_ratio, 1.0)

            risk_score = round(
                0.45 * sandwich_risk + 0.35 * frontrun_risk + 0.20 * bot_ratio,
                4,
            )

            result["risk_score"] = risk_score
            result["sandwich_risk"] = round(sandwich_risk, 4)
            result["frontrun_risk"] = round(frontrun_risk, 4)
            result["bot_activity_level"] = bot_level
            result["details"] = {
                "sandwiches_found": len(sandwiches),
                "bots_detected": bot_count,
                "addresses_checked": total_checked,
                "transactions_analysed": len(txs),
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("MEV risk scoring failed for %s: %s", token, exc)

        return result
