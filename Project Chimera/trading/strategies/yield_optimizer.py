"""Yield Optimization Strategy.

Tracks and compares yield opportunities across Solana DeFi protocols
including staking (Marinade, Lido, Jito), lending (Solend, Marginfi, Drift),
and LP farming.  Calculates impermanent loss and net yields to surface the
best risk-adjusted returns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

# ── Protocol registries ──────────────────────────────────────────────
STAKING_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "marinade": {
        "name": "Marinade Finance",
        "token": "mSOL",
        "url": "https://api.marinade.finance/msol/apy",
        "min_amount": 0.01,
    },
    "lido": {
        "name": "Lido on Solana",
        "token": "stSOL",
        "url": "https://solana.lido.fi/api/stats",
        "min_amount": 0.01,
    },
    "jito": {
        "name": "Jito Stake Pool",
        "token": "JitoSOL",
        "url": "https://api.jito.network/apy",
        "min_amount": 0.1,
    },
}

LENDING_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "solend": {
        "name": "Solend",
        "url": "https://api.solend.fi/v1/markets",
        "risk": 0.3,
    },
    "marginfi": {
        "name": "Marginfi",
        "url": "https://api.marginfi.com/v1/markets",
        "risk": 0.35,
    },
    "drift": {
        "name": "Drift Protocol",
        "url": "https://api.drift.trade/v1/markets",
        "risk": 0.4,
    },
}


# ── Dataclasses ───────────────────────────────────────────────────────

@dataclass
class YieldOpportunity:
    """A single yield-bearing opportunity."""

    protocol: str
    type: str  # "staking", "lending", "lp", "farming"
    apy: float
    tvl: float
    risk_score: float  # 0 (safest) → 1 (riskiest)
    token: str
    min_amount: float

    @property
    def daily_yield(self) -> float:
        """Annualized → daily yield."""
        return self.apy / 365.0

    @property
    def monthly_yield(self) -> float:
        """Annualized → monthly yield (simple)."""
        return self.apy / 12.0


@dataclass
class ImpermanentLoss:
    """Impermanent-loss snapshot for a liquidity pool position."""

    pool: str
    token_a: str
    token_b: str
    initial_ratio: float
    current_ratio: float
    il_pct: float  # expressed as a *negative* percentage (loss)

    @property
    def severity(self) -> str:
        """Human-friendly IL severity label."""
        abs_il = abs(self.il_pct)
        if abs_il < 1.0:
            return "negligible"
        if abs_il < 5.0:
            return "low"
        if abs_il < 15.0:
            return "moderate"
        return "high"


# ── Optimizer ─────────────────────────────────────────────────────────

class YieldOptimizer:
    """Find and compare the best yield opportunities on Solana.

    Args:
        session: Optional shared ``aiohttp.ClientSession``.
        default_risk_tolerance: Default maximum acceptable risk score.
    """

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        default_risk_tolerance: float = 0.5,
    ) -> None:
        self._session = session
        self._owns_session = session is None
        self.default_risk_tolerance = default_risk_tolerance
        self._yield_cache: Dict[str, List[YieldOpportunity]] = {}
        self._airdrop_scores: Dict[str, Dict[str, float]] = {}

    # ── Session management ────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Staking ───────────────────────────────────────────────────────

    async def _fetch_staking_apy(self, protocol: str) -> Optional[float]:
        """Fetch the current staking APY from *protocol*'s API."""
        info = STAKING_PROTOCOLS.get(protocol)
        if info is None:
            return None

        session = await self._get_session()
        try:
            async with session.get(
                info["url"], timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    logger.warning("Non-200 from %s: %d", protocol, resp.status)
                    return None
                data = await resp.json()
                # Each API has its own schema
                if protocol == "marinade":
                    return float(data.get("apy", data.get("value", 0)))
                if protocol == "lido":
                    return float(data.get("apy", 0))
                if protocol == "jito":
                    return float(data.get("apy", 0))
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.debug("Error fetching %s APY: %s", protocol, exc)
            return None

    async def find_staking_opportunities(
        self, amount: float
    ) -> List[YieldOpportunity]:
        """List SOL staking options available for *amount*.

        Args:
            amount: SOL amount the user wants to stake.

        Returns:
            Sorted list of ``YieldOpportunity`` (highest APY first).
        """
        opportunities: List[YieldOpportunity] = []

        tasks = {
            proto: self._fetch_staking_apy(proto) for proto in STAKING_PROTOCOLS
        }
        results = dict(
            zip(tasks.keys(), await asyncio.gather(*tasks.values()))
        )

        for proto, apy in results.items():
            info = STAKING_PROTOCOLS[proto]
            if amount < info["min_amount"]:
                continue
            effective_apy = apy if apy is not None else 0.0
            opp = YieldOpportunity(
                protocol=proto,
                type="staking",
                apy=effective_apy,
                tvl=0.0,
                risk_score=0.1,  # staking is low-risk
                token=info["token"],
                min_amount=info["min_amount"],
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda o: o.apy, reverse=True)
        self._yield_cache["staking"] = opportunities
        logger.info("Found %d staking opportunities for %.2f SOL", len(opportunities), amount)
        return opportunities

    # ── Impermanent Loss ──────────────────────────────────────────────

    @staticmethod
    def calculate_impermanent_loss(price_ratio_change: float) -> float:
        """Compute impermanent loss for a constant-product AMM.

        Formula::

            IL = 2 * sqrt(r) / (1 + r) - 1

        where *r* is the ratio of the new price to the original price.

        Args:
            price_ratio_change: ``new_price / old_price``.

        Returns:
            IL as a decimal (negative means loss). For example -0.05 ≈ 5 % loss.
        """
        if price_ratio_change <= 0:
            raise ValueError("price_ratio_change must be positive")

        r = float(price_ratio_change)
        il = 2.0 * np.sqrt(r) / (1.0 + r) - 1.0
        return float(il)

    def calculate_impermanent_loss_detail(
        self,
        pool: str,
        token_a: str,
        token_b: str,
        initial_ratio: float,
        current_ratio: float,
    ) -> ImpermanentLoss:
        """Return an ``ImpermanentLoss`` dataclass with full context.

        Args:
            pool: Pool identifier / address.
            token_a: Symbol of token A.
            token_b: Symbol of token B.
            initial_ratio: Price ratio at entry (price_a / price_b).
            current_ratio: Current price ratio.

        Returns:
            ``ImpermanentLoss`` instance.
        """
        if initial_ratio <= 0:
            raise ValueError("initial_ratio must be positive")
        r = current_ratio / initial_ratio
        il = self.calculate_impermanent_loss(r)
        return ImpermanentLoss(
            pool=pool,
            token_a=token_a,
            token_b=token_b,
            initial_ratio=initial_ratio,
            current_ratio=current_ratio,
            il_pct=float(np.round(il * 100.0, 4)),
        )

    # ── Lending ───────────────────────────────────────────────────────

    async def _fetch_lending_rates(
        self, protocol: str, token: str
    ) -> Optional[Dict[str, float]]:
        """Fetch supply/borrow rates for *token* on *protocol*."""
        info = LENDING_PROTOCOLS.get(protocol)
        if info is None:
            return None

        session = await self._get_session()
        try:
            async with session.get(
                info["url"],
                params={"token": token},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return {
                    "supply_apy": float(data.get("supply_apy", 0)),
                    "borrow_apy": float(data.get("borrow_apy", 0)),
                    "tvl": float(data.get("tvl", 0)),
                    "utilization": float(data.get("utilization", 0)),
                }
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            logger.debug("Lending rate fetch failed (%s/%s): %s", protocol, token, exc)
            return None

    async def analyze_lending_yields(
        self, token: str
    ) -> List[YieldOpportunity]:
        """Compare lending rates for *token* across protocols.

        Args:
            token: Token symbol (e.g. ``"SOL"``, ``"USDC"``).

        Returns:
            Sorted list of ``YieldOpportunity`` (highest supply APY first).
        """
        opportunities: List[YieldOpportunity] = []
        tasks = {
            proto: self._fetch_lending_rates(proto, token)
            for proto in LENDING_PROTOCOLS
        }
        results = dict(
            zip(tasks.keys(), await asyncio.gather(*tasks.values()))
        )

        for proto, rates in results.items():
            info = LENDING_PROTOCOLS[proto]
            supply_apy = rates["supply_apy"] if rates else 0.0
            tvl = rates["tvl"] if rates else 0.0

            opp = YieldOpportunity(
                protocol=proto,
                type="lending",
                apy=supply_apy,
                tvl=tvl,
                risk_score=info["risk"],
                token=token,
                min_amount=0.0,
            )
            opportunities.append(opp)

        opportunities.sort(key=lambda o: o.apy, reverse=True)
        self._yield_cache.setdefault("lending", []).extend(opportunities)
        logger.info("Analyzed lending yields for %s across %d protocols", token, len(opportunities))
        return opportunities

    # ── Airdrop Eligibility ───────────────────────────────────────────

    def track_airdrop_eligibility(
        self,
        wallet: str,
        protocols: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Score a wallet's airdrop eligibility per protocol.

        Scoring criteria (each on 0-1 scale, averaged):
            * transaction_count — normalised against a 500-tx ceiling.
            * volume_usd — normalised against a $100 000 ceiling.
            * unique_days — normalised against a 180-day ceiling.
            * tvl_contributed — normalised against a $50 000 ceiling.

        Args:
            wallet: Wallet public key.
            protocols: ``{protocol_name: {criterion: value, …}, …}``

        Returns:
            ``{protocol_name: eligibility_score}`` with scores in ``[0, 1]``.
        """
        ceilings = {
            "transaction_count": 500.0,
            "volume_usd": 100_000.0,
            "unique_days": 180.0,
            "tvl_contributed": 50_000.0,
        }
        scores: Dict[str, float] = {}

        for proto, criteria in protocols.items():
            criterion_scores: List[float] = []
            for key, ceiling in ceilings.items():
                value = float(criteria.get(key, 0))
                criterion_scores.append(min(value / ceiling, 1.0))
            score = float(np.mean(criterion_scores)) if criterion_scores else 0.0
            scores[proto] = float(np.round(score, 4))

        self._airdrop_scores[wallet] = scores
        logger.info("Airdrop eligibility for %s: %s", wallet[:8], scores)
        return scores

    # ── Best Yield Selection ──────────────────────────────────────────

    def _net_yield(
        self,
        opp: YieldOpportunity,
        il_pct: float = 0.0,
        management_fee_pct: float = 0.0,
    ) -> float:
        """Compute net yield: APY - IL - fees."""
        return opp.apy - abs(il_pct) - management_fee_pct

    async def get_best_yield(
        self,
        token: str,
        amount: float,
        risk_tolerance: float = 0.5,
    ) -> Optional[YieldOpportunity]:
        """Find the single best yield opportunity for *token* given constraints.

        Args:
            token: Token symbol.
            amount: Amount to deploy.
            risk_tolerance: Maximum risk score (0–1).

        Returns:
            The best ``YieldOpportunity`` or ``None``.
        """
        candidates: List[YieldOpportunity] = []

        # Gather staking (only relevant for SOL-like tokens)
        if token.upper() in ("SOL",):
            staking = await self.find_staking_opportunities(amount)
            candidates.extend(staking)

        # Gather lending
        lending = await self.analyze_lending_yields(token)
        candidates.extend(lending)

        # Filter by risk
        filtered = [c for c in candidates if c.risk_score <= risk_tolerance]
        if not filtered:
            logger.info("No yield opportunities within risk tolerance %.2f", risk_tolerance)
            return None

        best = max(filtered, key=lambda o: self._net_yield(o))
        logger.info(
            "Best yield for %s: %s @ %.2f%% APY (risk %.2f)",
            token, best.protocol, best.apy, best.risk_score,
        )
        return best

    # ── Portfolio Optimization ────────────────────────────────────────

    async def optimize_idle_capital(
        self,
        portfolio: Dict[str, float],
        risk_tolerance: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Auto-allocate idle capital across yield strategies.

        Args:
            portfolio: ``{token_symbol: idle_amount, …}``
            risk_tolerance: Maximum acceptable risk score.

        Returns:
            ``{token: {opportunity: …, allocated: …, expected_annual_yield: …}}``
        """
        tolerance = risk_tolerance or self.default_risk_tolerance
        allocations: Dict[str, Dict[str, Any]] = {}

        for token, amount in portfolio.items():
            if amount <= 0:
                continue
            best = await self.get_best_yield(token, amount, tolerance)
            if best is None:
                allocations[token] = {
                    "opportunity": None,
                    "allocated": 0.0,
                    "expected_annual_yield": 0.0,
                }
            else:
                expected_yield = amount * best.apy / 100.0
                allocations[token] = {
                    "opportunity": best,
                    "allocated": amount,
                    "expected_annual_yield": float(np.round(expected_yield, 4)),
                }

        total_yield = sum(a["expected_annual_yield"] for a in allocations.values())
        logger.info(
            "Optimized %d tokens, expected total annual yield: $%.2f",
            len(allocations), total_yield,
        )
        return allocations
