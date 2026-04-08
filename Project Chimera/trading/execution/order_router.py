# trading/execution/order_router.py
"""
Smart Order Router — compares prices across Solana DEXs (Jupiter, Raydium,
Orca) and determines the optimal execution route for each trade.

Integrates with the Jupiter aggregator API for best-price routing and
supports order splitting for large trades to minimise price impact.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# API endpoints                                                          #
# --------------------------------------------------------------------- #
JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
RAYDIUM_API_URL = "https://api.raydium.io/v2"

# Fee constants (basis points)
JUPITER_FEE_BPS = 0          # Jupiter charges 0 platform fee
RAYDIUM_FEE_BPS = 25         # 0.25 %
ORCA_FEE_BPS = 30            # 0.30 %

# Solana transaction costs
BASE_TX_FEE_LAMPORTS = 5_000
PRIORITY_FEE_LAMPORTS = 10_000
LAMPORTS_PER_SOL = 1_000_000_000

# Order-splitting thresholds
LARGE_ORDER_USD = 10_000.0
MAX_SPLIT_PARTS = 5
DEFAULT_SLIPPAGE_BPS = 50    # 0.5 %


# --------------------------------------------------------------------- #
# Data classes                                                           #
# --------------------------------------------------------------------- #
@dataclass
class RouteQuote:
    """A price quote from a single DEX or route."""

    dex: str
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    price_impact: float
    fee: float
    route_path: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_amount = round(self.output_amount, 8)
        self.price_impact = round(self.price_impact, 6)
        self.fee = round(self.fee, 8)

    @property
    def effective_rate(self) -> float:
        """Output per unit of input after fees."""
        if self.input_amount <= 0:
            return 0.0
        return self.output_amount / self.input_amount


@dataclass
class OrderPlan:
    """Full execution plan: best route, optional splits, totals."""

    quotes: List[RouteQuote]
    best_route: RouteQuote
    split_orders: List[Dict]
    estimated_total_output: float
    total_fees: float

    def __post_init__(self) -> None:
        self.estimated_total_output = round(self.estimated_total_output, 8)
        self.total_fees = round(self.total_fees, 8)


# --------------------------------------------------------------------- #
# Order Router                                                           #
# --------------------------------------------------------------------- #
class OrderRouter:
    """
    Queries multiple Solana DEXs, compares prices, estimates slippage,
    and optionally splits large orders for better execution.
    """

    def __init__(
        self,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        large_order_threshold: float = LARGE_ORDER_USD,
        max_split_parts: int = MAX_SPLIT_PARTS,
        timeout: int = 15,
    ) -> None:
        self.slippage_bps = slippage_bps
        self.large_order_threshold = large_order_threshold
        self.max_split_parts = max_split_parts
        self.timeout = timeout
        logger.info(
            "OrderRouter ready  slippage=%d bps  split_threshold=$%.0f",
            self.slippage_bps, self.large_order_threshold,
        )

    # ------------------------------------------------------------------ #
    # Jupiter aggregator                                                   #
    # ------------------------------------------------------------------ #

    async def get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
    ) -> Optional[RouteQuote]:
        """
        Fetch a swap quote from the Jupiter v6 aggregator.

        Parameters
        ----------
        input_mint : str
            SPL token mint address of the input token.
        output_mint : str
            SPL token mint address of the output token.
        amount : int
            Input amount in the token's smallest unit (lamports / base units).
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": self.slippage_bps,
        }

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    JUPITER_QUOTE_URL, params=params, timeout=self.timeout
                ),
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("Jupiter quote failed: %s", exc)
            return None

        try:
            out_amount = int(data.get("outAmount", 0))
            price_impact = float(data.get("priceImpactPct", 0))
            route_plan = data.get("routePlan", [])
            route_labels = [
                step.get("swapInfo", {}).get("label", "unknown")
                for step in route_plan
            ]
            total_fee = sum(
                int(step.get("swapInfo", {}).get("feeAmount", 0))
                for step in route_plan
            )
        except (ValueError, TypeError) as exc:
            logger.error("Failed to parse Jupiter response: %s", exc)
            return None

        return RouteQuote(
            dex="Jupiter",
            input_token=input_mint,
            output_token=output_mint,
            input_amount=float(amount),
            output_amount=float(out_amount),
            price_impact=price_impact,
            fee=float(total_fee),
            route_path=route_labels,
        )

    # ------------------------------------------------------------------ #
    # Raydium                                                              #
    # ------------------------------------------------------------------ #

    async def get_raydium_quote(
        self,
        pool: str,
        amount: float,
    ) -> Optional[RouteQuote]:
        """
        Estimate output from a Raydium AMM pool using the constant-product
        formula with the standard Raydium fee.

        Parameters
        ----------
        pool : str
            Pool identifier (JSON dict with reserve_a, reserve_b, fee_bps,
            input_token, output_token) or a pool address string.
        amount : float
            Input amount in base units.
        """
        if isinstance(pool, str):
            # Fetch pool info from Raydium API
            loop = asyncio.get_running_loop()
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{RAYDIUM_API_URL}/ammV3/ammPools",
                        timeout=self.timeout,
                    ),
                )
                resp.raise_for_status()
                pools_data = resp.json()
            except requests.RequestException as exc:
                logger.error("Raydium pool fetch failed: %s", exc)
                return None

            # Find the matching pool
            pool_info = None
            for p in pools_data.get("data", []):
                if p.get("id") == pool:
                    pool_info = p
                    break
            if pool_info is None:
                logger.warning("Raydium pool %s not found.", pool)
                return None

            reserve_a = float(pool_info.get("tvl", 0)) / 2
            reserve_b = reserve_a
            input_token = pool_info.get("mintA", "")
            output_token = pool_info.get("mintB", "")
            fee_bps = RAYDIUM_FEE_BPS
        else:
            # pool passed as a dict
            pool_dict = pool  # type: ignore[assignment]
            reserve_a = float(pool_dict.get("reserve_a", 0))
            reserve_b = float(pool_dict.get("reserve_b", 0))
            input_token = pool_dict.get("input_token", "")
            output_token = pool_dict.get("output_token", "")
            fee_bps = int(pool_dict.get("fee_bps", RAYDIUM_FEE_BPS))

        if reserve_a <= 0 or reserve_b <= 0:
            logger.warning("Invalid Raydium pool reserves.")
            return None

        # Constant-product AMM formula
        fee_fraction = fee_bps / 10_000
        amount_after_fee = amount * (1.0 - fee_fraction)
        output = (reserve_b * amount_after_fee) / (reserve_a + amount_after_fee)
        price_impact = amount_after_fee / (reserve_a + amount_after_fee)
        fee_amount = amount * fee_fraction

        return RouteQuote(
            dex="Raydium",
            input_token=input_token,
            output_token=output_token,
            input_amount=amount,
            output_amount=output,
            price_impact=price_impact,
            fee=fee_amount,
            route_path=["Raydium AMM"],
        )

    # ------------------------------------------------------------------ #
    # Cross-DEX comparison                                                 #
    # ------------------------------------------------------------------ #

    async def compare_dex_prices(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
    ) -> List[RouteQuote]:
        """
        Query all supported DEXs in parallel and return sorted quotes
        (best output first).
        """
        tasks = [
            self.get_jupiter_quote(input_mint, output_mint, amount),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: List[RouteQuote] = []
        for result in results:
            if isinstance(result, RouteQuote):
                quotes.append(result)
            elif isinstance(result, Exception):
                logger.warning("DEX query error: %s", result)

        quotes.sort(key=lambda q: q.output_amount, reverse=True)
        logger.info(
            "Compared %d DEX quotes for %s → %s (amount=%s)",
            len(quotes), input_mint[:8], output_mint[:8], amount,
        )
        return quotes

    # ------------------------------------------------------------------ #
    # Slippage estimation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_slippage(amount: float, liquidity: float) -> float:
        """
        Estimate slippage as a fraction of trade amount relative to
        available liquidity using a constant-product model.

        Returns slippage as a positive fraction (0.01 = 1 %).
        """
        if liquidity <= 0:
            return 1.0
        # x·y = k model: slippage ≈ amount / (liquidity + amount)
        return amount / (liquidity + amount)

    # ------------------------------------------------------------------ #
    # Order splitting                                                      #
    # ------------------------------------------------------------------ #

    def split_order(
        self,
        amount: float,
        liquidity_map: Dict[str, float],
    ) -> List[Dict]:
        """
        Split a large order across DEXs proportional to their liquidity
        to minimise aggregate price impact.

        Parameters
        ----------
        amount : float
            Total order amount (base units).
        liquidity_map : dict
            DEX name → available liquidity.

        Returns
        -------
        list of dict
            Each element: {"dex": str, "amount": float, "pct": float}
        """
        total_liquidity = sum(liquidity_map.values())
        if total_liquidity <= 0:
            # Fall back to single order on first DEX
            first_dex = next(iter(liquidity_map), "Jupiter")
            return [{"dex": first_dex, "amount": amount, "pct": 1.0}]

        splits: List[Dict] = []
        remaining = amount
        for dex, liq in sorted(
            liquidity_map.items(), key=lambda kv: kv[1], reverse=True
        ):
            share = liq / total_liquidity
            part = round(amount * share, 0)
            part = min(part, remaining)
            if part <= 0:
                continue
            splits.append({
                "dex": dex,
                "amount": part,
                "pct": part / amount if amount > 0 else 0,
            })
            remaining -= part
            if remaining <= 0 or len(splits) >= self.max_split_parts:
                break

        # Allocate any remainder to the top-liquidity DEX
        if remaining > 0 and splits:
            splits[0]["amount"] += remaining
            splits[0]["pct"] = splits[0]["amount"] / amount if amount > 0 else 0

        logger.info(
            "Order split into %d parts: %s",
            len(splits),
            [(s["dex"], s["amount"]) for s in splits],
        )
        return splits

    # ------------------------------------------------------------------ #
    # Gas estimation                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_gas_cost(num_instructions: int = 1, sol_price: float = 150.0) -> float:
        """
        Estimate transaction gas cost in USD.

        Parameters
        ----------
        num_instructions : int
            Number of instructions in the transaction.
        sol_price : float
            Current SOL price in USD.

        Returns
        -------
        float
            Estimated cost in USD.
        """
        total_lamports = BASE_TX_FEE_LAMPORTS + PRIORITY_FEE_LAMPORTS * num_instructions
        sol_cost = total_lamports / LAMPORTS_PER_SOL
        return round(sol_cost * sol_price, 6)

    # ------------------------------------------------------------------ #
    # Best route                                                           #
    # ------------------------------------------------------------------ #

    async def get_best_route(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        liquidity_map: Optional[Dict[str, float]] = None,
    ) -> OrderPlan:
        """
        Determine the optimal execution plan: compare DEXs, optionally
        split, and return a complete OrderPlan.
        """
        quotes = await self.compare_dex_prices(input_mint, output_mint, amount)

        if not quotes:
            logger.warning("No valid quotes received; returning empty plan.")
            empty_quote = RouteQuote(
                dex="none",
                input_token=input_mint,
                output_token=output_mint,
                input_amount=float(amount),
                output_amount=0.0,
                price_impact=1.0,
                fee=0.0,
            )
            return OrderPlan(
                quotes=[],
                best_route=empty_quote,
                split_orders=[],
                estimated_total_output=0.0,
                total_fees=0.0,
            )

        best = quotes[0]

        # Decide whether to split
        split_orders: List[Dict] = []
        if liquidity_map and float(amount) > self.large_order_threshold:
            split_orders = self.split_order(float(amount), liquidity_map)

        total_fees = sum(q.fee for q in quotes[:1])  # best route fees
        gas_usd = self.estimate_gas_cost()

        plan = OrderPlan(
            quotes=quotes,
            best_route=best,
            split_orders=split_orders,
            estimated_total_output=best.output_amount,
            total_fees=total_fees + gas_usd,
        )

        logger.info(
            "Best route: %s  output=%s  impact=%.4f%%  fees=%.6f",
            best.dex, best.output_amount,
            best.price_impact * 100, plan.total_fees,
        )
        return plan
