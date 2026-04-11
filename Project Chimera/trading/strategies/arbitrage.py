"""Cross-DEX Arbitrage Engine.

Detects and executes arbitrage opportunities across Solana DEXs
(Raydium, Orca, Jupiter) including cross-DEX and triangular arbitrage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_DEXS = ["raydium", "orca", "jupiter"]
DEFAULT_FEE_PCT = 0.3
DEFAULT_SLIPPAGE_BPS = 50
DEFAULT_OPPORTUNITY_TTL = 10.0


@dataclass
class DEXPrice:
    """Price quote from a specific DEX."""

    dex: str
    token: str
    price: float
    liquidity: float
    timestamp: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        """Return how many seconds old this price quote is."""
        return time.time() - self.timestamp

    def is_stale(self, max_age: float = 5.0) -> bool:
        """Check if the price quote is too old to be reliable."""
        return self.age_seconds > max_age


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""

    type: str  # "cross_dex" or "triangular"
    path: List[str]
    expected_profit_pct: float
    expected_profit_usd: float
    fees: float
    net_profit: float
    confidence: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """Check whether this opportunity has expired."""
        return time.time() > self.expires_at

    @property
    def time_remaining(self) -> float:
        """Seconds until expiry."""
        return max(0.0, self.expires_at - time.time())


class ArbitrageEngine:
    """Async engine for detecting and executing arbitrage across Solana DEXs.

    Supports cross-DEX arbitrage (buying on one DEX, selling on another) and
    triangular arbitrage (A → B → C → A circular paths).

    Args:
        fee_pct: Fee percentage per swap (default 0.3%).
        slippage_bps: Slippage tolerance in basis points (default 50).
        opportunity_ttl: Seconds before an opportunity expires (default 10).
        min_profit_pct: Minimum net profit percentage to consider (default 0.1%).
        session: Optional shared ``aiohttp.ClientSession``.
    """

    def __init__(
        self,
        fee_pct: float = DEFAULT_FEE_PCT,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        opportunity_ttl: float = DEFAULT_OPPORTUNITY_TTL,
        min_profit_pct: float = 0.1,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.fee_pct = fee_pct
        self.slippage_bps = slippage_bps
        self.opportunity_ttl = opportunity_ttl
        self.min_profit_pct = min_profit_pct
        self._session = session
        self._owns_session = session is None
        self._price_cache: Dict[str, List[DEXPrice]] = {}
        self._opportunities: List[ArbitrageOpportunity] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (and lazily create) an ``aiohttp`` session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    async def _fetch_dex_price(
        self, dex: str, token: str
    ) -> Optional[DEXPrice]:
        """Fetch the current price of *token* on *dex*.

        In production this would hit the DEX's API; here we build the
        request but handle connection errors gracefully so the engine
        can be tested in isolation.
        """
        endpoints = {
            "raydium": f"https://api.raydium.io/v2/main/price?tokens={token}",
            "orca": f"https://api.orca.so/v1/token/price?token={token}",
            "jupiter": f"https://price.jup.ag/v4/price?ids={token}",
        }
        url = endpoints.get(dex)
        if url is None:
            logger.warning("Unknown DEX: %s", dex)
            return None

        session = await self._get_session()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.warning("Non-200 from %s for %s: %d", dex, token, resp.status)
                    return None
                data = await resp.json()
                price, liquidity = self._parse_price_response(dex, token, data)
                return DEXPrice(
                    dex=dex,
                    token=token,
                    price=price,
                    liquidity=liquidity,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.debug("Failed to fetch price from %s for %s: %s", dex, token, exc)
            return None

    @staticmethod
    def _parse_price_response(
        dex: str, token: str, data: Any
    ) -> Tuple[float, float]:
        """Extract price and liquidity from a DEX API response."""
        try:
            if dex == "jupiter":
                entry = data.get("data", {}).get(token, {})
                return float(entry.get("price", 0)), float(entry.get("liquidity", 0))
            if dex == "raydium":
                return float(data.get(token, 0)), 0.0
            if dex == "orca":
                return float(data.get("price", 0)), float(data.get("liquidity", 0))
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("Parse error for %s/%s: %s", dex, token, exc)
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Cross-DEX arbitrage
    # ------------------------------------------------------------------

    async def scan_cross_dex(
        self, token_pairs: List[str]
    ) -> List[ArbitrageOpportunity]:
        """Compare prices of *token_pairs* across DEXs and return opportunities.

        Args:
            token_pairs: List of token mint addresses / symbols to scan.

        Returns:
            Sorted list of profitable ``ArbitrageOpportunity`` objects.
        """
        opportunities: List[ArbitrageOpportunity] = []

        for token in token_pairs:
            tasks = [self._fetch_dex_price(dex, token) for dex in SUPPORTED_DEXS]
            results = await asyncio.gather(*tasks)
            prices: List[DEXPrice] = [p for p in results if p is not None and p.price > 0]

            if len(prices) < 2:
                continue

            self._price_cache[token] = prices

            for buy in prices:
                for sell in prices:
                    if buy.dex == sell.dex:
                        continue
                    profit = self.calculate_arbitrage_profit(
                        buy_price=buy.price,
                        sell_price=sell.price,
                        amount=1.0,
                        fees_pct=self.fee_pct,
                    )
                    if profit["net_profit_pct"] < self.min_profit_pct:
                        continue

                    confidence = self._estimate_confidence(buy, sell)
                    opp = ArbitrageOpportunity(
                        type="cross_dex",
                        path=[f"{buy.dex}:{token}", f"{sell.dex}:{token}"],
                        expected_profit_pct=profit["gross_profit_pct"],
                        expected_profit_usd=profit["net_profit_usd"],
                        fees=profit["total_fees"],
                        net_profit=profit["net_profit_pct"],
                        confidence=confidence,
                        expires_at=time.time() + self.opportunity_ttl,
                    )
                    opportunities.append(opp)

        self._opportunities = self.rank_opportunities(opportunities)
        logger.info("Cross-DEX scan found %d opportunities", len(self._opportunities))
        return self._opportunities

    def _estimate_confidence(self, buy: DEXPrice, sell: DEXPrice) -> float:
        """Heuristic confidence score in ``[0, 1]``."""
        age_penalty = 1.0 - min((buy.age_seconds + sell.age_seconds) / 20.0, 1.0)
        liq = min(buy.liquidity, sell.liquidity)
        liq_score = min(liq / 100_000, 1.0) if liq > 0 else 0.5
        return float(np.clip(age_penalty * 0.6 + liq_score * 0.4, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Triangular arbitrage
    # ------------------------------------------------------------------

    def find_triangular_paths(
        self, price_matrix: Dict[str, Dict[str, float]]
    ) -> List[ArbitrageOpportunity]:
        """Find circular arbitrage paths A → B → C → A.

        Args:
            price_matrix: Nested dict ``{tokenA: {tokenB: price_A_per_B, …}, …}``
                representing exchange rates between every pair of tokens.

        Returns:
            List of profitable ``ArbitrageOpportunity`` objects.
        """
        tokens = list(price_matrix.keys())
        if len(tokens) < 3:
            return []

        opportunities: List[ArbitrageOpportunity] = []
        fee_mult = 1.0 - self.fee_pct / 100.0
        slippage_mult = 1.0 - self.slippage_bps / 10_000.0

        for path in permutations(tokens, 3):
            a, b, c = path
            try:
                rate_ab = price_matrix[a][b]
                rate_bc = price_matrix[b][c]
                rate_ca = price_matrix[c][a]
            except KeyError:
                continue

            if rate_ab <= 0 or rate_bc <= 0 or rate_ca <= 0:
                continue

            # After three swaps (each incurring fee + slippage) starting with 1 unit of A
            end_amount = (
                rate_ab * fee_mult * slippage_mult
                * rate_bc * fee_mult * slippage_mult
                * rate_ca * fee_mult * slippage_mult
            )
            gross_pct = (end_amount - 1.0) * 100.0
            total_fee_pct = (1.0 - fee_mult ** 3) * 100.0
            net_pct = gross_pct  # fees already embedded in end_amount

            if net_pct < self.min_profit_pct:
                continue

            opp = ArbitrageOpportunity(
                type="triangular",
                path=[a, b, c, a],
                expected_profit_pct=gross_pct,
                expected_profit_usd=0.0,  # requires notional sizing
                fees=total_fee_pct,
                net_profit=net_pct,
                confidence=0.7,
                expires_at=time.time() + self.opportunity_ttl,
            )
            opportunities.append(opp)

        result = self.rank_opportunities(opportunities)
        logger.info("Triangular scan found %d opportunities", len(result))
        return result

    # ------------------------------------------------------------------
    # Profit calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_arbitrage_profit(
        buy_price: float,
        sell_price: float,
        amount: float,
        fees_pct: float = DEFAULT_FEE_PCT,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    ) -> Dict[str, float]:
        """Compute net profit for a simple buy-low / sell-high arbitrage.

        Args:
            buy_price: Price on the cheaper DEX.
            sell_price: Price on the more expensive DEX.
            amount: Notional amount in tokens.
            fees_pct: Fee per swap as a percentage.
            slippage_bps: Expected slippage in basis points.

        Returns:
            Dict with gross/net profit (pct and USD) and total fees.
        """
        if buy_price <= 0:
            return {
                "gross_profit_pct": 0.0,
                "gross_profit_usd": 0.0,
                "total_fees": 0.0,
                "net_profit_pct": 0.0,
                "net_profit_usd": 0.0,
            }

        fee_mult = 1.0 - fees_pct / 100.0
        slippage_mult = 1.0 - slippage_bps / 10_000.0

        # Buy side: actual cost is higher (divide by discount factors)
        # Sell side: actual proceeds are lower (multiply by discount factors)
        effective_buy = buy_price / (fee_mult * slippage_mult)
        effective_sell = sell_price * fee_mult * slippage_mult

        gross_pct = ((sell_price - buy_price) / buy_price) * 100.0
        net_pct = ((effective_sell - effective_buy) / effective_buy) * 100.0

        cost = amount * effective_buy
        revenue = amount * effective_sell
        gross_usd = amount * (sell_price - buy_price)
        net_usd = revenue - cost
        total_fees = gross_usd - net_usd

        return {
            "gross_profit_pct": float(np.round(gross_pct, 6)),
            "gross_profit_usd": float(np.round(gross_usd, 6)),
            "total_fees": float(np.round(total_fees, 6)),
            "net_profit_pct": float(np.round(net_pct, 6)),
            "net_profit_usd": float(np.round(net_usd, 6)),
        }

    # ------------------------------------------------------------------
    # Ranking & execution
    # ------------------------------------------------------------------

    @staticmethod
    def rank_opportunities(
        opportunities: List[ArbitrageOpportunity],
    ) -> List[ArbitrageOpportunity]:
        """Sort opportunities by net profit descending, pruning expired ones."""
        now = time.time()
        valid = [o for o in opportunities if o.expires_at > now]
        return sorted(valid, key=lambda o: o.net_profit, reverse=True)

    async def execute_arbitrage(
        self, opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute an arbitrage opportunity.

        In a live system this would build and submit Solana transactions;
        here it validates the opportunity and returns an execution report.

        Args:
            opportunity: The opportunity to execute.

        Returns:
            Execution report dict.
        """
        if opportunity.is_expired:
            logger.warning("Opportunity expired %.1fs ago", -opportunity.time_remaining)
            return {"status": "expired", "opportunity": opportunity}

        if opportunity.confidence < 0.3:
            logger.warning("Low confidence (%.2f), skipping", opportunity.confidence)
            return {"status": "skipped_low_confidence", "opportunity": opportunity}

        logger.info(
            "Executing %s arb: path=%s net=%.4f%%",
            opportunity.type,
            opportunity.path,
            opportunity.net_profit,
        )

        session = await self._get_session()
        try:
            # In production: build + sign + send Solana transaction(s)
            await asyncio.sleep(0)  # placeholder for async tx submission
            return {
                "status": "submitted",
                "opportunity": opportunity,
                "path": opportunity.path,
                "expected_net_profit_pct": opportunity.net_profit,
            }
        except Exception as exc:
            logger.error("Execution failed: %s", exc)
            return {"status": "error", "error": str(exc), "opportunity": opportunity}
