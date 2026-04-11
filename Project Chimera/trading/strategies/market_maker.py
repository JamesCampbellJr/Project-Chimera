"""Market Making Strategy.

Implements an Avellaneda-Stoikov inspired market-making engine with
inventory-aware spread calculation, exponential inventory penalty,
bid/ask skew, fee tracking, and full PnL decomposition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """A two-sided market quote."""

    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    timestamp: float = field(default_factory=time.time)

    @property
    def mid_price(self) -> float:
        """Midpoint of the quote."""
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_pct(self) -> float:
        """Spread as a percentage of mid price."""
        mid = self.mid_price
        return ((self.ask_price - self.bid_price) / mid * 100.0) if mid > 0 else 0.0


@dataclass
class MMState:
    """Full state snapshot of the market maker."""

    inventory: float = 0.0
    inventory_value: float = 0.0
    mid_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees: float = 0.0
    quotes_placed: int = 0
    quotes_filled: int = 0
    fill_rate: float = 0.0

    @property
    def total_pnl(self) -> float:
        """Realised + unrealised + fees."""
        return self.realized_pnl + self.unrealized_pnl + self.total_fees


class MarketMaker:
    """Inventory-aware market maker using the Avellaneda-Stoikov model.

    The optimal spread is computed as::

        spread = γ σ² T  +  (2/γ) ln(1 + γ/k)

    where:
        * γ (gamma)  — risk-aversion parameter
        * σ (sigma)  — asset volatility
        * T          — remaining time horizon
        * k          — order-book shape parameter

    Inventory skew shifts the mid-price proportionally to current
    inventory, and an exponential penalty widens the spread as
    inventory approaches the maximum.

    Args:
        max_inventory: Hard inventory cap (absolute value, both sides).
        gamma: Risk-aversion parameter (default 0.1).
        k: Order-book shape parameter (default 1.5).
        time_horizon: Trading session length in seconds (default 3600).
        fee_rate: Maker fee rate (default 0.02 %).
    """

    def __init__(
        self,
        max_inventory: float = 100.0,
        gamma: float = 0.1,
        k: float = 1.5,
        time_horizon: float = 3600.0,
        fee_rate: float = 0.0002,
    ) -> None:
        self.max_inventory = max_inventory
        self.gamma = gamma
        self.k = k
        self.time_horizon = time_horizon
        self.fee_rate = fee_rate

        # Mutable state
        self._inventory: float = 0.0
        self._avg_entry_price: float = 0.0
        self._realized_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._mid_price: float = 0.0
        self._quotes_placed: int = 0
        self._quotes_filled: int = 0
        self._trade_log: List[Dict] = []
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Spread calculation (Avellaneda-Stoikov)
    # ------------------------------------------------------------------

    def calculate_spread(
        self,
        volatility: float,
        liquidity: float,
        inventory_risk: float,
        gamma: Optional[float] = None,
    ) -> float:
        """Compute the optimal bid-ask spread.

        Args:
            volatility: Annualised volatility (σ). Must be positive.
            liquidity: A proxy for available order-book depth (used as *k*
                when > 0; otherwise falls back to ``self.k``).
            inventory_risk: Current inventory risk factor (0–1).
            gamma: Override for the risk-aversion parameter.

        Returns:
            Spread in *price units* (not bps).
        """
        g = gamma if gamma is not None else self.gamma
        if g <= 0:
            raise ValueError("gamma must be positive")
        if volatility < 0:
            raise ValueError("volatility must be non-negative")

        sigma = volatility
        remaining = max(self.time_horizon - (time.time() - self._start_time), 1.0)
        t_frac = remaining / self.time_horizon

        effective_k = liquidity if liquidity > 0 else self.k

        # Avellaneda-Stoikov spread
        variance_component = g * (sigma ** 2) * t_frac
        depth_component = (2.0 / g) * np.log(1.0 + g / effective_k)

        base_spread = float(variance_component + depth_component)

        # Exponential inventory penalty
        inv_ratio = abs(inventory_risk)
        penalty = float(np.exp(2.0 * inv_ratio) - 1.0)
        spread = base_spread * (1.0 + penalty)

        logger.debug(
            "Spread: base=%.6f penalty=%.4f final=%.6f (σ=%.4f γ=%.4f inv_risk=%.4f)",
            base_spread, penalty, spread, sigma, g, inventory_risk,
        )
        return spread

    # ------------------------------------------------------------------
    # Quote generation
    # ------------------------------------------------------------------

    def generate_quotes(
        self,
        mid_price: float,
        spread: float,
        size: float,
        inventory: float,
        max_inventory: Optional[float] = None,
    ) -> Quote:
        """Create a bid/ask quote with inventory-based skew.

        The mid-price is shifted toward the side that would *reduce*
        inventory: when long, the bid is lowered (discouraging buys)
        and the ask is lowered (encouraging sells), and vice-versa.

        Args:
            mid_price: Current fair-value mid price.
            spread: Total spread in price units.
            size: Base order size.
            inventory: Current net inventory.
            max_inventory: Inventory cap (uses ``self.max_inventory`` if ``None``).

        Returns:
            A ``Quote`` with skew-adjusted bid/ask prices and sizes.
        """
        if mid_price <= 0:
            raise ValueError("mid_price must be positive")
        if spread < 0:
            raise ValueError("spread must be non-negative")
        if size <= 0:
            raise ValueError("size must be positive")

        max_inv = max_inventory or self.max_inventory
        half = spread / 2.0

        # Skew: shift mid proportionally to inventory / max
        skew = 0.0
        if max_inv > 0:
            skew = -(inventory / max_inv) * half

        adjusted_mid = mid_price + skew
        bid = adjusted_mid - half
        ask = adjusted_mid + half

        # Size adjustment: reduce size on the over-exposed side
        inv_ratio = inventory / max_inv if max_inv > 0 else 0.0
        bid_size = size * max(1.0 - inv_ratio, 0.1)
        ask_size = size * max(1.0 + inv_ratio, 0.1)

        spread_bps = (ask - bid) / mid_price * 10_000.0

        self._mid_price = mid_price
        self._quotes_placed += 1

        quote = Quote(
            bid_price=float(np.round(bid, 8)),
            ask_price=float(np.round(ask, 8)),
            bid_size=float(np.round(bid_size, 6)),
            ask_size=float(np.round(ask_size, 6)),
            spread_bps=float(np.round(spread_bps, 2)),
        )
        logger.debug(
            "Quote: bid=%.6f ask=%.6f skew=%.6f spread=%.2f bps",
            quote.bid_price, quote.ask_price, skew, quote.spread_bps,
        )
        return quote

    # ------------------------------------------------------------------
    # Inventory management
    # ------------------------------------------------------------------

    def update_inventory(self, side: str, amount: float, price: float) -> None:
        """Record a fill and update inventory and PnL.

        Args:
            side: ``"buy"`` or ``"sell"``.
            amount: Number of tokens filled.
            price: Execution price.

        Raises:
            ValueError: If *side* is invalid or *amount*/*price* non-positive.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if amount <= 0:
            raise ValueError("amount must be positive")
        if price <= 0:
            raise ValueError("price must be positive")

        fee = amount * price * self.fee_rate
        self._total_fees += fee

        if side == "buy":
            # Update weighted average entry price
            total_cost = self._avg_entry_price * abs(self._inventory) if self._inventory > 0 else 0.0
            new_cost = price * amount
            self._inventory += amount
            if self._inventory > 0:
                self._avg_entry_price = (total_cost + new_cost) / self._inventory
        else:
            # Realise PnL on sells
            if self._inventory > 0 and self._avg_entry_price > 0:
                sell_amount = min(amount, self._inventory)
                self._realized_pnl += sell_amount * (price - self._avg_entry_price)
            self._inventory -= amount

        self._quotes_filled += 1
        self._trade_log.append({
            "side": side,
            "amount": amount,
            "price": price,
            "fee": fee,
            "inventory_after": self._inventory,
            "timestamp": time.time(),
        })

        logger.info(
            "Fill: %s %.4f @ %.6f | inv=%.4f rpnl=%.4f fees=%.6f",
            side, amount, price, self._inventory, self._realized_pnl, fee,
        )

    def manage_inventory_risk(
        self, current_inventory: float, max_inventory: Optional[float] = None
    ) -> Dict[str, float]:
        """Assess current inventory risk.

        Args:
            current_inventory: Net inventory position.
            max_inventory: Inventory cap (uses ``self.max_inventory`` if ``None``).

        Returns:
            Dict with risk metrics and recommended actions.
        """
        max_inv = max_inventory or self.max_inventory
        if max_inv <= 0:
            return {"risk_ratio": 0.0, "risk_level": 0.0, "action": "none"}

        ratio = abs(current_inventory) / max_inv
        risk_level = float(np.clip(ratio, 0.0, 1.0))

        # Exponential penalty for inventory risk
        penalty = float(np.exp(2.0 * risk_level) - 1.0)

        if risk_level >= 0.9:
            action = "emergency_flatten"
        elif risk_level >= 0.7:
            action = "aggressive_reduce"
        elif risk_level >= 0.5:
            action = "gentle_reduce"
        else:
            action = "normal"

        direction = "long" if current_inventory > 0 else "short" if current_inventory < 0 else "flat"

        return {
            "risk_ratio": float(np.round(ratio, 4)),
            "risk_level": float(np.round(risk_level, 4)),
            "penalty": float(np.round(penalty, 4)),
            "direction": direction,
            "action": action,
        }

    # ------------------------------------------------------------------
    # Performance reporting
    # ------------------------------------------------------------------

    def get_mm_performance(self) -> MMState:
        """Return a full PnL and performance report.

        The unrealised PnL is computed using the last known mid price.

        Returns:
            ``MMState`` snapshot.
        """
        unrealised = 0.0
        if self._inventory != 0 and self._mid_price > 0 and self._avg_entry_price > 0:
            unrealised = self._inventory * (self._mid_price - self._avg_entry_price)

        fill_rate = (
            self._quotes_filled / self._quotes_placed
            if self._quotes_placed > 0
            else 0.0
        )

        state = MMState(
            inventory=float(np.round(self._inventory, 6)),
            inventory_value=float(np.round(self._inventory * self._mid_price, 4)),
            mid_price=self._mid_price,
            realized_pnl=float(np.round(self._realized_pnl, 6)),
            unrealized_pnl=float(np.round(unrealised, 6)),
            total_fees=float(np.round(self._total_fees, 6)),
            quotes_placed=self._quotes_placed,
            quotes_filled=self._quotes_filled,
            fill_rate=float(np.round(fill_rate, 4)),
        )

        logger.info(
            "MM Performance: inv=%.4f rpnl=%.4f upnl=%.4f fees=%.4f total=%.4f fill=%.1f%%",
            state.inventory, state.realized_pnl, state.unrealized_pnl,
            state.total_fees, state.total_pnl, state.fill_rate * 100,
        )
        return state

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state back to initial values."""
        self._inventory = 0.0
        self._avg_entry_price = 0.0
        self._realized_pnl = 0.0
        self._total_fees = 0.0
        self._mid_price = 0.0
        self._quotes_placed = 0
        self._quotes_filled = 0
        self._trade_log.clear()
        self._start_time = time.time()
        logger.info("Market maker state reset")
