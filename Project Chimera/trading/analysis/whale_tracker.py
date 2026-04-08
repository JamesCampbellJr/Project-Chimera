# trading/analysis/whale_tracker.py
"""
Whale tracking system for Solana tokens.

Identifies the largest holders of a given SPL token, scores them as
"smart money" based on historical win-rate and ROI, and monitors
recent large movements that may signal upcoming price action.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from trading.solana_client import SolanaClient
import config

logger = logging.getLogger(__name__)

# Average Solana slot duration in seconds (used for age estimation).
_SLOT_DURATION_SECS = 0.4


# --------------------------------------------------------------------- #
#  Data classes                                                          #
# --------------------------------------------------------------------- #

@dataclass
class WhaleInfo:
    """Profile of a single large token holder."""

    address: str
    balance_usd: float
    win_rate: float
    avg_roi: float
    wallet_age_days: int
    is_smart_money: bool


@dataclass
class WhaleMovement:
    """A significant token transfer made by a whale wallet."""

    timestamp: float
    whale: str
    token: str
    amount_usd: float
    direction: str  # "buy" | "sell" | "transfer_in" | "transfer_out"
    significance_score: float


# --------------------------------------------------------------------- #
#  WhaleTracker                                                          #
# --------------------------------------------------------------------- #

class WhaleTracker:
    """Track and analyse whale wallets for a given SPL token.

    The tracker fetches the largest token accounts from on-chain data,
    resolves the owning wallets, and scores them based on historical
    trading performance.
    """

    def __init__(self, solana_client: Optional[SolanaClient] = None) -> None:
        self._client = solana_client or SolanaClient()

        # In-memory caches keyed by wallet address.
        self._whale_cache: dict[str, WhaleInfo] = {}
        self._movement_cache: dict[str, list[WhaleMovement]] = {}
        # Cache for wallet trade stats: {address: {win_rate, avg_roi, age_days}}
        self._stats_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    #  Core public API                                                     #
    # ------------------------------------------------------------------ #

    async def track_whales(
        self, token: str, top_n: int = 20
    ) -> list[WhaleInfo]:
        """Identify the *top_n* largest holders of *token* and return
        enriched :class:`WhaleInfo` objects.

        Uses ``getTokenLargestAccounts`` to discover accounts, resolves
        owner addresses, converts balances to USD, and computes win-rate /
        ROI from recent transaction history.

        Args:
            token: SPL token mint address.
            top_n: Maximum number of whale wallets to return.

        Returns:
            Sorted list of :class:`WhaleInfo` (descending by balance_usd).
        """
        loop = asyncio.get_event_loop()

        try:
            # 1. Fetch the largest token accounts on-chain.
            largest = await loop.run_in_executor(
                None,
                lambda: self._client._rpc(
                    "getTokenLargestAccounts",
                    [token, {"commitment": "finalized"}],
                ),
            )

            if not largest or "value" not in largest:
                logger.warning("No largest-account data for token %s", token)
                return []

            accounts: list[dict] = largest["value"][:top_n]

            # 2. Resolve owner addresses for each token account.
            owners: list[tuple[str, float]] = []
            for acct in accounts:
                acct_address = acct.get("address", "")
                ui_amount = float(
                    acct.get("uiAmount")
                    or (acct.get("amount", "0") if isinstance(acct.get("amount"), (int, float)) else 0)
                )
                owner = await loop.run_in_executor(
                    None, lambda addr=acct_address: self._resolve_owner(addr)
                )
                if owner:
                    owners.append((owner, ui_amount))

            # 3. Get SOL price for USD conversion.
            sol_price = await loop.run_in_executor(
                None, self._client.get_sol_price_usd
            )
            token_price = await loop.run_in_executor(
                None, lambda: self._client.get_token_price_usd(token)
            )
            price_usd = token_price if token_price else 0.0

            # 4. Build WhaleInfo objects.
            whales: list[WhaleInfo] = []
            for owner_addr, balance in owners:
                stats = await loop.run_in_executor(
                    None,
                    lambda addr=owner_addr: self._compute_wallet_stats(addr),
                )
                balance_usd = balance * price_usd

                whale = WhaleInfo(
                    address=owner_addr,
                    balance_usd=balance_usd,
                    win_rate=stats["win_rate"],
                    avg_roi=stats["avg_roi"],
                    wallet_age_days=stats["age_days"],
                    is_smart_money=(
                        stats["win_rate"] >= 60.0 and stats["avg_roi"] > 0
                    ),
                )
                whales.append(whale)
                self._whale_cache[owner_addr] = whale

            whales.sort(key=lambda w: w.balance_usd, reverse=True)
            return whales

        except Exception:
            logger.exception("Error tracking whales for token %s", token)
            return []

    def identify_smart_money(
        self,
        wallets: list[WhaleInfo],
        min_win_rate: float = 60.0,
    ) -> list[WhaleInfo]:
        """Filter *wallets* to those that qualify as smart money.

        A wallet is considered smart money when its historical win-rate
        meets or exceeds *min_win_rate* **and** it has a positive average
        ROI.

        Args:
            wallets: List of whale profiles to filter.
            min_win_rate: Minimum win-rate percentage threshold.

        Returns:
            Subset of *wallets* that satisfy the smart-money criteria.
        """
        return [
            w
            for w in wallets
            if w.win_rate >= min_win_rate and w.avg_roi > 0
        ]

    async def get_whale_movements(
        self, token: str, period_hours: int = 24
    ) -> list[WhaleMovement]:
        """Scan top whale wallets for large token movements within
        the last *period_hours* hours.

        The significance score for each movement is calculated relative
        to the average transfer amount across all observed movements.

        Args:
            token: SPL token mint address.
            period_hours: Look-back window in hours.

        Returns:
            List of :class:`WhaleMovement` sorted by significance
            (descending).
        """
        loop = asyncio.get_event_loop()

        try:
            # Ensure whale cache is populated.
            whales = list(self._whale_cache.values())
            if not whales:
                whales = await self.track_whales(token, top_n=20)

            cutoff = time.time() - (period_hours * 3600)
            raw_movements: list[dict] = []

            for whale in whales[:20]:
                sigs = await loop.run_in_executor(
                    None,
                    lambda addr=whale.address: (
                        self._client.get_transaction_signatures(addr, limit=50)
                    ),
                )
                for sig_info in sigs:
                    block_time = sig_info.get("blockTime", 0)
                    if block_time and block_time < cutoff:
                        continue
                    if sig_info.get("err"):
                        continue

                    tx = await loop.run_in_executor(
                        None,
                        lambda s=sig_info["signature"]: (
                            self._client.get_transaction(s)
                        ),
                    )
                    if not tx:
                        continue

                    swap = self._client.parse_swap_from_transaction(
                        tx, whale.address
                    )
                    if swap and (
                        swap.get("token_in") == token
                        or swap.get("token_out") == token
                    ):
                        direction = (
                            "sell" if swap["token_in"] == token else "buy"
                        )
                        amount_usd = swap.get("amount_out", 0) or swap.get(
                            "amount_in", 0
                        )
                        raw_movements.append(
                            {
                                "timestamp": swap.get("block_time", 0) or 0,
                                "whale": whale.address,
                                "token": token,
                                "amount_usd": amount_usd,
                                "direction": direction,
                            }
                        )

            # Compute significance scores relative to mean volume.
            amounts = [m["amount_usd"] for m in raw_movements]
            avg_amount = sum(amounts) / len(amounts) if amounts else 1.0

            movements: list[WhaleMovement] = []
            for m in raw_movements:
                sig_score = min(m["amount_usd"] / avg_amount, 10.0)
                movements.append(
                    WhaleMovement(
                        timestamp=m["timestamp"],
                        whale=m["whale"],
                        token=m["token"],
                        amount_usd=m["amount_usd"],
                        direction=m["direction"],
                        significance_score=round(sig_score, 4),
                    )
                )

            movements.sort(key=lambda m: m.significance_score, reverse=True)
            self._movement_cache[token] = movements
            return movements

        except Exception:
            logger.exception(
                "Error fetching whale movements for token %s", token
            )
            return []

    def analyze_wallet_age(self, address: str, created_slot: int) -> int:
        """Estimate a wallet's age in days from its creation slot.

        Uses the approximate Solana slot duration of ~0.4 s to convert the
        delta between the current (estimated) slot and *created_slot* into
        calendar days.

        Args:
            address: Wallet address (for logging / cache key).
            created_slot: Slot number at which the wallet's first
                transaction was recorded.

        Returns:
            Estimated age in whole days (floored).
        """
        try:
            slot_info = self._client._rpc("getSlot", [{"commitment": "finalized"}])
            current_slot = slot_info if isinstance(slot_info, int) else 0
            if current_slot <= created_slot:
                return 0
            elapsed_secs = (current_slot - created_slot) * _SLOT_DURATION_SECS
            age_days = int(elapsed_secs / 86400)
            return age_days
        except Exception:
            logger.exception(
                "Error estimating wallet age for %s at slot %d",
                address,
                created_slot,
            )
            return 0

    def get_whale_score(self, token: str) -> dict:
        """Return a synchronous summary score for *token*.

        The result dict contains:

        * **concentration** – fraction of total tracked USD held by the
          top-5 wallets.
        * **smart_money_ratio** – proportion of tracked whales that are
          classified as smart money.
        * **recent_movement_score** – average significance score of
          cached movements (0 if none).
        * **overall_score** – weighted composite (0–1).

        Args:
            token: SPL token mint address.

        Returns:
            Dict with the four score components described above.
        """
        whales = list(self._whale_cache.values())
        movements = self._movement_cache.get(token, [])

        # Concentration: share of total USD in top-5 wallets.
        total_usd = sum(w.balance_usd for w in whales) or 1.0
        top5_usd = sum(
            w.balance_usd for w in sorted(
                whales, key=lambda w: w.balance_usd, reverse=True
            )[:5]
        )
        concentration = top5_usd / total_usd

        # Smart-money ratio.
        smart_count = sum(1 for w in whales if w.is_smart_money)
        smart_ratio = smart_count / len(whales) if whales else 0.0

        # Recent movement score.
        avg_sig = (
            sum(m.significance_score for m in movements) / len(movements)
            if movements
            else 0.0
        )
        movement_score = min(avg_sig / 5.0, 1.0)

        # Weighted composite.
        overall = (
            0.35 * concentration
            + 0.35 * smart_ratio
            + 0.30 * movement_score
        )

        return {
            "concentration": round(concentration, 4),
            "smart_money_ratio": round(smart_ratio, 4),
            "recent_movement_score": round(movement_score, 4),
            "overall_score": round(overall, 4),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_owner(self, token_account: str) -> Optional[str]:
        """Resolve the owner wallet address of an SPL token account."""
        try:
            info = self._client._rpc(
                "getAccountInfo",
                [token_account, {"encoding": "jsonParsed"}],
            )
            if not info or "value" not in info:
                return None
            data = info["value"].get("data", {})
            if isinstance(data, dict):
                parsed = data.get("parsed", {})
                owner = parsed.get("info", {}).get("owner")
                return owner
            return None
        except Exception:
            logger.debug(
                "Could not resolve owner for token account %s", token_account
            )
            return None

    def _compute_wallet_stats(self, address: str) -> dict:
        """Compute win-rate, average ROI, and age for *address*.

        Returns a dict with keys ``win_rate``, ``avg_roi``, and
        ``age_days``.  Results are cached for the lifetime of this
        tracker instance.
        """
        if address in self._stats_cache:
            return self._stats_cache[address]

        stats: dict = {"win_rate": 0.0, "avg_roi": 0.0, "age_days": 0}

        try:
            sigs = self._client.get_transaction_signatures(address, limit=100)
            if not sigs:
                self._stats_cache[address] = stats
                return stats

            # Estimate wallet age from the earliest signature's slot.
            earliest_slot = min(s.get("slot", 0) for s in sigs)
            stats["age_days"] = self.analyze_wallet_age(address, earliest_slot)

            # Parse trades to derive win-rate / ROI.
            trades: list[dict] = []
            for sig_info in sigs[:50]:  # limit to keep RPC calls bounded
                if sig_info.get("err"):
                    continue
                tx = self._client.get_transaction(sig_info["signature"])
                if not tx:
                    continue
                swap = self._client.parse_swap_from_transaction(tx, address)
                if swap:
                    trades.append(swap)
                time.sleep(config.RPC_RATE_LIMIT_DELAY)

            if trades:
                # Pair buys/sells by token to estimate per-trade PnL.
                buys: dict[str, list[float]] = {}
                sells: dict[str, list[float]] = {}
                for t in trades:
                    tok_out = t.get("token_out", "")
                    tok_in = t.get("token_in", "")
                    if tok_out:
                        buys.setdefault(tok_out, []).append(
                            t.get("amount_out", 0)
                        )
                    if tok_in:
                        sells.setdefault(tok_in, []).append(
                            t.get("amount_in", 0)
                        )

                wins = 0
                rois: list[float] = []
                for tok in set(buys) | set(sells):
                    total_buy = sum(buys.get(tok, []))
                    total_sell = sum(sells.get(tok, []))
                    if total_buy > 0:
                        roi = (total_sell - total_buy) / total_buy
                        rois.append(roi)
                        if roi > 0:
                            wins += 1

                token_count = len(set(buys) | set(sells))
                stats["win_rate"] = (
                    (wins / token_count * 100) if token_count else 0.0
                )
                stats["avg_roi"] = (
                    sum(rois) / len(rois) * 100 if rois else 0.0
                )

        except Exception:
            logger.exception(
                "Error computing wallet stats for %s", address
            )

        self._stats_cache[address] = stats
        return stats
