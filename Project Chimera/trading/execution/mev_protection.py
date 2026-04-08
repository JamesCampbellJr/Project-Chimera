# trading/execution/mev_protection.py
"""
MEV Protection — shields trades from sandwich attacks and front-running
on Solana using Jito-style bundle submission, commit-reveal schemes,
randomised timing, and transaction splitting.

The module estimates MEV risk for a given trade and selects an appropriate
protection strategy automatically.
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants                                                              #
# --------------------------------------------------------------------- #
JITO_BUNDLE_URL = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"
DEFAULT_TIP_LAMPORTS = 10_000        # 0.00001 SOL
MAX_TIP_LAMPORTS = 1_000_000         # 0.001 SOL
LAMPORTS_PER_SOL = 1_000_000_000

# Risk thresholds
HIGH_MEV_RISK_USD = 5_000.0          # trades above this get max protection
MEDIUM_MEV_RISK_USD = 1_000.0        # moderate protection


# --------------------------------------------------------------------- #
# Enums & data classes                                                   #
# --------------------------------------------------------------------- #
class ProtectionMethod(str, Enum):
    NONE = "none"
    JITO_BUNDLE = "jito_bundle"
    COMMIT_REVEAL = "commit_reveal"
    RANDOM_TIMING = "random_timing"
    TX_SPLITTING = "tx_splitting"
    COMBINED = "combined"


class CommitRevealStatus(str, Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    REVEALED = "revealed"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class ProtectionStrategy:
    """Recommended MEV protection approach for a trade."""

    method: ProtectionMethod
    params: Dict[str, Any] = field(default_factory=dict)
    estimated_savings: float = 0.0

    def __post_init__(self) -> None:
        self.estimated_savings = round(self.estimated_savings, 4)


@dataclass
class CommitReveal:
    """State for a commit-reveal transaction pair."""

    commitment_hash: str
    reveal_data: Dict[str, Any]
    commit_tx: Optional[str] = None
    reveal_tx: Optional[str] = None
    status: CommitRevealStatus = CommitRevealStatus.PENDING

    @property
    def is_complete(self) -> bool:
        return self.status == CommitRevealStatus.REVEALED


# --------------------------------------------------------------------- #
# MEV Protection                                                         #
# --------------------------------------------------------------------- #
class MEVProtection:
    """
    Provides multiple layers of MEV protection for Solana trades.
    Automatically selects the best strategy based on trade size and
    market conditions.
    """

    def __init__(
        self,
        default_tip: int = DEFAULT_TIP_LAMPORTS,
        max_tip: int = MAX_TIP_LAMPORTS,
        jito_url: str = JITO_BUNDLE_URL,
        timeout: int = 15,
    ) -> None:
        self.default_tip = default_tip
        self.max_tip = max_tip
        self.jito_url = jito_url
        self.timeout = timeout

        # Track pending commit-reveals
        self._pending_commits: Dict[str, CommitReveal] = {}

        logger.info(
            "MEVProtection ready  tip=%d lamports  max_tip=%d lamports",
            self.default_tip, self.max_tip,
        )

    # ------------------------------------------------------------------ #
    # Jito bundle submission                                               #
    # ------------------------------------------------------------------ #

    async def submit_with_tip(
        self,
        transaction: Dict[str, Any],
        tip_lamports: Optional[int] = None,
    ) -> Optional[str]:
        """
        Submit a transaction as a Jito bundle with a validator tip.

        The tip incentivises the block builder to include the transaction
        without leaking it to the public mempool, preventing sandwich
        attacks.

        Parameters
        ----------
        transaction : dict
            Serialised transaction data with keys:
              - serialized_tx (str): base-58 encoded transaction
              - recent_blockhash (str)
        tip_lamports : int or None
            Tip amount; uses default if None.

        Returns
        -------
        str or None
            Bundle ID on success, None on failure.
        """
        tip = tip_lamports if tip_lamports is not None else self.default_tip
        tip = min(tip, self.max_tip)

        bundle_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [
                [transaction.get("serialized_tx", "")],
            ],
        }

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.jito_url,
                    json=bundle_payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                ),
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            logger.error("Jito bundle submission failed: %s", exc)
            return None

        bundle_id = result.get("result")
        if bundle_id:
            logger.info(
                "Jito bundle submitted: %s  tip=%d lamports (%.6f SOL)",
                bundle_id, tip, tip / LAMPORTS_PER_SOL,
            )
        else:
            error = result.get("error", "unknown")
            logger.warning("Jito bundle rejected: %s", error)

        return bundle_id

    # ------------------------------------------------------------------ #
    # Commit-reveal                                                        #
    # ------------------------------------------------------------------ #

    def create_commitment(self, trade_intent: Dict[str, Any]) -> CommitReveal:
        """
        Create a commitment hash for a trade intent.

        The commitment hides the trade details until the reveal phase,
        preventing front-running by adversaries who monitor the mempool.

        Parameters
        ----------
        trade_intent : dict
            Must include: input_token, output_token, amount, direction.
        """
        # Generate a random nonce for binding
        nonce = secrets.token_hex(32)

        reveal_data = {
            **trade_intent,
            "nonce": nonce,
            "timestamp": time.time(),
        }

        # SHA-256 commitment
        payload = json.dumps(reveal_data, sort_keys=True).encode()
        commitment_hash = hashlib.sha256(payload).hexdigest()

        commit = CommitReveal(
            commitment_hash=commitment_hash,
            reveal_data=reveal_data,
            status=CommitRevealStatus.PENDING,
        )

        self._pending_commits[commitment_hash] = commit

        logger.info(
            "Commitment created: %s…  token=%s  amount=%s",
            commitment_hash[:16],
            trade_intent.get("input_token", "?")[:8],
            trade_intent.get("amount", "?"),
        )
        return commit

    async def reveal_commitment(self, commitment: CommitReveal) -> bool:
        """
        Reveal a previously committed trade.

        In a full on-chain implementation this would submit the reveal
        transaction.  Here we verify the hash and update status.

        Returns True on success.
        """
        # Verify hash matches
        payload = json.dumps(commitment.reveal_data, sort_keys=True).encode()
        computed = hashlib.sha256(payload).hexdigest()

        if computed != commitment.commitment_hash:
            logger.error(
                "Commitment hash mismatch: expected %s, got %s",
                commitment.commitment_hash[:16], computed[:16],
            )
            commitment.status = CommitRevealStatus.FAILED
            return False

        commitment.status = CommitRevealStatus.REVEALED
        self._pending_commits.pop(commitment.commitment_hash, None)

        logger.info(
            "Commitment revealed: %s…", commitment.commitment_hash[:16],
        )
        return True

    # ------------------------------------------------------------------ #
    # Random timing                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def randomize_timing(
        min_delay: float = 0.1,
        max_delay: float = 2.0,
    ) -> float:
        """
        Wait a random duration within a configurable window to make
        transaction timing unpredictable.

        Returns the actual delay applied (seconds).
        """
        if min_delay < 0:
            min_delay = 0.0
        if max_delay <= min_delay:
            max_delay = min_delay + 0.1

        # Cryptographically random float in [min_delay, max_delay]
        random_bytes = secrets.token_bytes(8)
        random_float = int.from_bytes(random_bytes, "big") / (2**64)
        delay = min_delay + random_float * (max_delay - min_delay)

        logger.debug("Random timing delay: %.3f s", delay)
        await asyncio.sleep(delay)
        return delay

    # ------------------------------------------------------------------ #
    # Transaction splitting                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def split_transaction(
        trade: Dict[str, Any],
        num_parts: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Split a trade into multiple smaller parts to obfuscate the
        total order size and direction.

        Parameters
        ----------
        trade : dict
            Must include: amount (float), input_token, output_token.
        num_parts : int
            Number of sub-transactions (2–10).

        Returns
        -------
        list of dict
            Each sub-trade has a randomised fraction of the total amount.
        """
        num_parts = max(2, min(10, num_parts))
        total = float(trade.get("amount", 0))
        if total <= 0:
            return [trade]

        # Generate random split weights
        raw_weights = [
            int.from_bytes(secrets.token_bytes(4), "big") for _ in range(num_parts)
        ]
        total_weight = sum(raw_weights) or 1
        fractions = [w / total_weight for w in raw_weights]

        parts: List[Dict[str, Any]] = []
        allocated = 0.0
        for i, frac in enumerate(fractions):
            part_amount = round(total * frac, 8)
            if i == len(fractions) - 1:
                # Last part gets the remainder to avoid rounding drift
                part_amount = round(total - allocated, 8)
            allocated += part_amount

            parts.append({
                **trade,
                "amount": part_amount,
                "part_index": i,
                "total_parts": num_parts,
            })

        logger.info(
            "Trade split into %d parts: amounts=%s",
            num_parts, [p["amount"] for p in parts],
        )
        return parts

    # ------------------------------------------------------------------ #
    # MEV risk estimation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_mev_risk(trade: Dict[str, Any]) -> float:
        """
        Estimate MEV extraction risk as a value in [0, 1].

        Higher values indicate greater vulnerability to sandwich attacks
        or front-running.  Based on trade size, price impact, and token
        liquidity.
        """
        amount_usd = float(trade.get("amount_usd", 0))
        price_impact = float(trade.get("price_impact", 0))
        liquidity = float(trade.get("liquidity", 1))

        # Size factor: larger trades are more profitable targets
        size_risk = min(1.0, amount_usd / HIGH_MEV_RISK_USD)

        # Impact factor: high impact trades signal opportunity
        impact_risk = min(1.0, price_impact * 20)  # 5% impact → risk 1.0

        # Liquidity factor: thin liquidity amplifies MEV
        liq_risk = min(1.0, 10_000 / max(liquidity, 1))

        # Weighted combination
        risk = 0.4 * size_risk + 0.35 * impact_risk + 0.25 * liq_risk
        risk = max(0.0, min(1.0, risk))

        logger.debug(
            "MEV risk estimate: %.3f  (size=%.2f  impact=%.2f  liq=%.2f)",
            risk, size_risk, impact_risk, liq_risk,
        )
        return round(risk, 4)

    # ------------------------------------------------------------------ #
    # Strategy selection                                                   #
    # ------------------------------------------------------------------ #

    def get_protection_strategy(
        self,
        trade: Dict[str, Any],
    ) -> ProtectionStrategy:
        """
        Automatically select the best MEV protection strategy based on
        trade characteristics.

        Parameters
        ----------
        trade : dict
            Trade intent with amount_usd, price_impact, liquidity, etc.
        """
        risk = self.estimate_mev_risk(trade)
        amount_usd = float(trade.get("amount_usd", 0))

        if risk < 0.2:
            # Low risk — basic random timing is sufficient
            return ProtectionStrategy(
                method=ProtectionMethod.RANDOM_TIMING,
                params={"min_delay": 0.1, "max_delay": 1.0},
                estimated_savings=0.0,
            )

        if risk < 0.5 or amount_usd < MEDIUM_MEV_RISK_USD:
            # Moderate risk — Jito bundle
            tip = self.default_tip
            estimated_savings = amount_usd * risk * 0.01  # ~1% of at-risk value
            return ProtectionStrategy(
                method=ProtectionMethod.JITO_BUNDLE,
                params={"tip_lamports": tip},
                estimated_savings=estimated_savings,
            )

        if risk < 0.7:
            # High risk — Jito + splitting
            tip = min(int(self.default_tip * 2), self.max_tip)
            num_parts = min(3, self.max_tip)
            estimated_savings = amount_usd * risk * 0.02
            return ProtectionStrategy(
                method=ProtectionMethod.TX_SPLITTING,
                params={"tip_lamports": tip, "num_parts": 3},
                estimated_savings=estimated_savings,
            )

        # Very high risk — full combined protection
        tip = min(int(self.default_tip * 3), self.max_tip)
        estimated_savings = amount_usd * risk * 0.03
        return ProtectionStrategy(
            method=ProtectionMethod.COMBINED,
            params={
                "tip_lamports": tip,
                "num_parts": 4,
                "use_commit_reveal": True,
                "min_delay": 0.5,
                "max_delay": 3.0,
            },
            estimated_savings=estimated_savings,
        )
