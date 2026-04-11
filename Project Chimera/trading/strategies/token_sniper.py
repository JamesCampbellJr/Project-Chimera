"""New Token Launch Sniping Strategy.

Monitors Raydium and Orca for new pool creation events via Solana RPC,
analyses tokenomics and liquidity, scores rug-pull risk, and evaluates
sniping opportunities for newly launched tokens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

# Well-known Solana program IDs for pool creation
RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
ORCA_WHIRLPOOL_PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"

DEFAULT_MIN_LIQUIDITY_USD = 5_000.0
DEFAULT_MIN_SAFETY_SCORE = 0.4


@dataclass
class NewToken:
    """Metadata about a newly launched token."""

    address: str
    name: str
    symbol: str
    launch_time: float
    initial_liquidity: float
    holder_count: int
    mint_authority_revoked: bool
    freeze_authority_revoked: bool
    lp_locked: bool
    rug_risk_score: float = 0.0  # 0 (safe) → 1 (very risky)

    @property
    def age_seconds(self) -> float:
        """Seconds since launch."""
        return time.time() - self.launch_time


@dataclass
class SnipeOpportunity:
    """Evaluated sniping opportunity for a new token."""

    token: NewToken
    entry_price: float
    target_exit_pct: float  # e.g. 200.0 for a 2× target
    risk_score: float
    liquidity: float
    timing_window_secs: float
    safety_score: float

    @property
    def is_actionable(self) -> bool:
        """Quick check: liquidity and safety above minimums."""
        return (
            self.liquidity >= DEFAULT_MIN_LIQUIDITY_USD
            and self.safety_score >= DEFAULT_MIN_SAFETY_SCORE
        )


class TokenSniper:
    """Monitor new token launches and evaluate sniping opportunities.

    Args:
        rpc_url: Solana JSON-RPC endpoint.
        min_liquidity: Minimum initial liquidity (USD) to consider.
        min_safety_score: Minimum safety score (0–1) to consider.
        session: Optional shared ``aiohttp.ClientSession``.
    """

    def __init__(
        self,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        min_liquidity: float = DEFAULT_MIN_LIQUIDITY_USD,
        min_safety_score: float = DEFAULT_MIN_SAFETY_SCORE,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.rpc_url = rpc_url
        self.min_liquidity = min_liquidity
        self.min_safety_score = min_safety_score
        self._session = session
        self._owns_session = session is None
        self._monitoring = False
        self._detected_tokens: List[NewToken] = []

    # ── Session helpers ───────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        self._monitoring = False
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Solana RPC helpers ────────────────────────────────────────────

    async def _rpc_call(self, method: str, params: List[Any]) -> Any:
        """Make a JSON-RPC call to the Solana cluster."""
        session = await self._get_session()
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        try:
            async with session.post(
                self.rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    logger.warning("RPC error: %s", data["error"])
                    return None
                return data.get("result")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("RPC call %s failed: %s", method, exc)
            return None

    async def _get_recent_signatures(self, program_id: str, limit: int = 20) -> List[str]:
        """Fetch recent transaction signatures for a program."""
        result = await self._rpc_call(
            "getSignaturesForAddress",
            [program_id, {"limit": limit}],
        )
        if result is None:
            return []
        return [entry["signature"] for entry in result if "signature" in entry]

    # ── Launch monitoring ─────────────────────────────────────────────

    async def monitor_launches(
        self,
        callback: Callable[[NewToken], Coroutine[Any, Any, None]],
        poll_interval: float = 2.0,
    ) -> None:
        """Continuously poll for new pool creation events.

        For each new pool detected, *callback* is invoked with a
        ``NewToken`` instance.  Call :meth:`close` or set
        ``self._monitoring = False`` to stop.

        Args:
            callback: Async callable invoked for every new token found.
            poll_interval: Seconds between polls.
        """
        self._monitoring = True
        seen_signatures: set[str] = set()
        logger.info("Starting launch monitor (interval=%.1fs)", poll_interval)

        while self._monitoring:
            for program_id in (RAYDIUM_AMM_PROGRAM, ORCA_WHIRLPOOL_PROGRAM):
                sigs = await self._get_recent_signatures(program_id)
                for sig in sigs:
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    token = await self._parse_pool_creation(sig, program_id)
                    if token is not None:
                        token.rug_risk_score = self.calculate_rug_risk(token)
                        self._detected_tokens.append(token)
                        logger.info(
                            "New token detected: %s (%s) risk=%.2f",
                            token.symbol, token.address[:12], token.rug_risk_score,
                        )
                        try:
                            await callback(token)
                        except Exception as exc:
                            logger.error("Callback error for %s: %s", token.symbol, exc)

            await asyncio.sleep(poll_interval)

    async def _parse_pool_creation(
        self, signature: str, program_id: str
    ) -> Optional[NewToken]:
        """Parse a transaction to extract new-pool / new-token data."""
        tx = await self._rpc_call(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        )
        if tx is None:
            return None

        try:
            meta = tx.get("meta", {})
            msg = tx.get("transaction", {}).get("message", {})
            accounts = msg.get("accountKeys", [])

            if not accounts:
                return None

            token_address = accounts[0] if isinstance(accounts[0], str) else accounts[0].get("pubkey", "")
            post_balances = meta.get("postTokenBalances", [])

            initial_liquidity = 0.0
            for bal in post_balances:
                amount = bal.get("uiTokenAmount", {}).get("uiAmount")
                if amount and amount > initial_liquidity:
                    initial_liquidity = float(amount)

            placeholder_name = token_address[:16] if len(token_address) >= 16 else "UNKNOWN"
            placeholder_symbol = token_address[:6].upper() if len(token_address) >= 6 else "UNK"

            return NewToken(
                address=token_address,
                name=placeholder_name,
                symbol=placeholder_symbol,
                launch_time=time.time(),
                initial_liquidity=initial_liquidity,
                holder_count=1,
                mint_authority_revoked=False,
                freeze_authority_revoked=False,
                lp_locked=False,
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.debug("Failed to parse tx %s: %s", signature[:12], exc)
            return None

    # ── Tokenomics analysis ───────────────────────────────────────────

    @staticmethod
    def analyze_tokenomics(token_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse supply distribution and vesting characteristics.

        Args:
            token_info: Dict with keys ``total_supply``, ``top_holders``
                (list of ``{address, balance}``), and optional ``vesting_schedules``.

        Returns:
            Analysis dict with concentration metrics and flags.
        """
        total_supply = float(token_info.get("total_supply", 0))
        top_holders: List[Dict[str, Any]] = token_info.get("top_holders", [])
        vesting = token_info.get("vesting_schedules", [])

        if total_supply <= 0:
            return {"error": "invalid total_supply", "risk": "extreme"}

        holder_pcts = [
            float(h.get("balance", 0)) / total_supply * 100.0 for h in top_holders
        ]
        top_holder_pct = holder_pcts[0] if holder_pcts else 0.0
        top_10_pct = float(np.sum(holder_pcts[:10]))
        hhi = float(np.sum(np.array(holder_pcts) ** 2)) if holder_pcts else 0.0

        has_vesting = len(vesting) > 0
        vested_pct = sum(float(v.get("percentage", 0)) for v in vesting)

        concentration = "low"
        if top_10_pct > 80:
            concentration = "extreme"
        elif top_10_pct > 50:
            concentration = "high"
        elif top_10_pct > 30:
            concentration = "moderate"

        return {
            "total_supply": total_supply,
            "num_holders": len(top_holders),
            "top_holder_pct": float(np.round(top_holder_pct, 2)),
            "top_10_pct": float(np.round(top_10_pct, 2)),
            "hhi": float(np.round(hhi, 2)),
            "concentration": concentration,
            "has_vesting": has_vesting,
            "vested_pct": float(np.round(vested_pct, 2)),
        }

    # ── Liquidity assessment ──────────────────────────────────────────

    def assess_initial_liquidity(
        self, pool_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess whether initial pool liquidity is adequate.

        Args:
            pool_info: Dict with keys ``liquidity_usd``, ``token_reserve``,
                ``quote_reserve``, ``pool_age_seconds``.

        Returns:
            Assessment dict with adequacy verdict and metrics.
        """
        liquidity = float(pool_info.get("liquidity_usd", 0))
        token_reserve = float(pool_info.get("token_reserve", 0))
        quote_reserve = float(pool_info.get("quote_reserve", 0))
        pool_age = float(pool_info.get("pool_age_seconds", 0))

        reserve_ratio = (
            min(token_reserve, quote_reserve) / max(token_reserve, quote_reserve)
            if max(token_reserve, quote_reserve) > 0
            else 0.0
        )

        adequate = liquidity >= self.min_liquidity
        balanced = reserve_ratio >= 0.3

        return {
            "liquidity_usd": liquidity,
            "adequate": adequate,
            "reserve_ratio": float(np.round(reserve_ratio, 4)),
            "balanced": balanced,
            "pool_age_seconds": pool_age,
            "verdict": "pass" if (adequate and balanced) else "fail",
        }

    # ── Rug-pull risk scoring ─────────────────────────────────────────

    @staticmethod
    def calculate_rug_risk(token: NewToken) -> float:
        """Compute a composite rug-pull probability score.

        Weighted factors (total weight = 1.0):
            * ``mint_authority_revoked``   — weight 0.25 (revoked → lower risk)
            * ``freeze_authority_revoked`` — weight 0.20 (revoked → lower risk)
            * ``lp_locked``                — weight 0.25 (locked → lower risk)
            * ``holder_concentration``     — weight 0.15 (fewer → higher risk)
            * ``liquidity_ratio``          — weight 0.15 (lower → higher risk)

        Returns:
            Score in ``[0, 1]`` where 1 = maximum rug-pull risk.
        """
        weights = {
            "mint_authority": 0.25,
            "freeze_authority": 0.20,
            "lp_lock": 0.25,
            "holder_concentration": 0.15,
            "liquidity_ratio": 0.15,
        }

        scores: Dict[str, float] = {}

        # Mint authority: not revoked → risky
        scores["mint_authority"] = 0.0 if token.mint_authority_revoked else 1.0

        # Freeze authority: not revoked → risky
        scores["freeze_authority"] = 0.0 if token.freeze_authority_revoked else 1.0

        # LP locked: not locked → risky
        scores["lp_lock"] = 0.0 if token.lp_locked else 1.0

        # Holder concentration: fewer holders → riskier
        if token.holder_count >= 1000:
            scores["holder_concentration"] = 0.0
        elif token.holder_count >= 100:
            scores["holder_concentration"] = 0.3
        elif token.holder_count >= 10:
            scores["holder_concentration"] = 0.6
        else:
            scores["holder_concentration"] = 1.0

        # Liquidity: low liquidity → riskier
        if token.initial_liquidity >= 100_000:
            scores["liquidity_ratio"] = 0.0
        elif token.initial_liquidity >= 50_000:
            scores["liquidity_ratio"] = 0.2
        elif token.initial_liquidity >= 10_000:
            scores["liquidity_ratio"] = 0.5
        else:
            scores["liquidity_ratio"] = 1.0

        composite = sum(weights[k] * scores[k] for k in weights)
        return float(np.clip(np.round(composite, 4), 0.0, 1.0))

    # ── Safety score ──────────────────────────────────────────────────

    @staticmethod
    def calculate_safety_score(token: NewToken) -> float:
        """Safety score is the inverse of rug risk.

        Returns:
            Score in ``[0, 1]`` where 1 = maximum safety.
        """
        risk = TokenSniper.calculate_rug_risk(token)
        return float(np.round(1.0 - risk, 4))

    # ── Opportunity evaluation ────────────────────────────────────────

    async def evaluate_snipe_opportunity(
        self, token: NewToken
    ) -> Optional[SnipeOpportunity]:
        """Run full analysis on a newly detected token.

        Args:
            token: The ``NewToken`` to evaluate.

        Returns:
            A ``SnipeOpportunity`` if the token passes filters, else ``None``.
        """
        rug_risk = self.calculate_rug_risk(token)
        safety = self.calculate_safety_score(token)

        if token.initial_liquidity < self.min_liquidity:
            logger.info(
                "Skipping %s: liquidity $%.0f < min $%.0f",
                token.symbol, token.initial_liquidity, self.min_liquidity,
            )
            return None

        if safety < self.min_safety_score:
            logger.info(
                "Skipping %s: safety %.2f < min %.2f",
                token.symbol, safety, self.min_safety_score,
            )
            return None

        # Estimate entry price from liquidity (simplified model)
        entry_price = (
            token.initial_liquidity / 1_000_000
            if token.initial_liquidity > 0
            else 0.0
        )

        # Target exit scales inversely with risk
        target_exit_pct = float(np.round(100.0 + (1.0 - rug_risk) * 300.0, 2))

        # Timing window shrinks with age
        age = token.age_seconds
        timing_window = max(30.0 - age, 0.0)

        opp = SnipeOpportunity(
            token=token,
            entry_price=entry_price,
            target_exit_pct=target_exit_pct,
            risk_score=rug_risk,
            liquidity=token.initial_liquidity,
            timing_window_secs=timing_window,
            safety_score=safety,
        )

        logger.info(
            "Snipe opportunity: %s entry=$%.6f target=%.0f%% safety=%.2f window=%.0fs",
            token.symbol, entry_price, target_exit_pct, safety, timing_window,
        )
        return opp
