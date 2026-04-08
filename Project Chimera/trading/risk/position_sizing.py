# trading/risk/position_sizing.py
"""
Position Sizing Engine — determines optimal trade sizes using Kelly Criterion
and dynamic adjustments for volatility, correlation, and drawdown.

The engine enforces portfolio concentration limits and maximum position sizes
to prevent catastrophic losses from any single trade.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Configuration defaults                                                 #
# --------------------------------------------------------------------- #
DEFAULT_KELLY_FRACTION = 0.25        # quarter-Kelly for safety
MAX_KELLY_FRACTION = 0.50            # never exceed half-Kelly
MIN_POSITION_USD = 10.0              # ignore dust orders
MAX_SINGLE_POSITION_PCT = 0.10       # 10 % of portfolio per position
MAX_PORTFOLIO_CONCENTRATION = 0.30   # 30 % max in correlated cluster
MAX_POSITION_USD = 5_000.0           # hard cap per position


# --------------------------------------------------------------------- #
# Data classes                                                           #
# --------------------------------------------------------------------- #
@dataclass
class PositionSize:
    """Computed position sizing recommendation."""

    token: str
    size_usd: float
    size_pct: float
    kelly_fraction: float
    confidence_adjustment: float
    risk_score: float

    def __post_init__(self) -> None:
        self.size_usd = round(self.size_usd, 2)
        self.size_pct = round(self.size_pct, 6)
        self.kelly_fraction = round(self.kelly_fraction, 6)
        self.confidence_adjustment = round(self.confidence_adjustment, 4)
        self.risk_score = round(self.risk_score, 4)


@dataclass
class PortfolioHolding:
    """Snapshot of one existing position used for concentration checks."""

    token: str
    value_usd: float
    weight: float
    correlation_group: str = "default"


# --------------------------------------------------------------------- #
# Position Sizer                                                         #
# --------------------------------------------------------------------- #
class PositionSizer:
    """
    Computes optimal trade sizes using Kelly Criterion with fractional
    scaling, then adjusts for real-time conditions (volatility, drawdown,
    correlation, model confidence).
    """

    def __init__(
        self,
        kelly_fraction: float = DEFAULT_KELLY_FRACTION,
        max_position_pct: float = MAX_SINGLE_POSITION_PCT,
        max_position_usd: float = MAX_POSITION_USD,
        min_position_usd: float = MIN_POSITION_USD,
        max_concentration: float = MAX_PORTFOLIO_CONCENTRATION,
    ) -> None:
        if not 0 < kelly_fraction <= MAX_KELLY_FRACTION:
            raise ValueError(
                f"kelly_fraction must be in (0, {MAX_KELLY_FRACTION}], "
                f"got {kelly_fraction}"
            )
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_position_usd = max_position_usd
        self.min_position_usd = min_position_usd
        self.max_concentration = max_concentration
        logger.info(
            "PositionSizer ready  kelly=%.2f  max_pct=%.1f%%  max_usd=$%.0f",
            self.kelly_fraction,
            self.max_position_pct * 100,
            self.max_position_usd,
        )

    # ------------------------------------------------------------------ #
    # Core Kelly calculations                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly fraction: f* = (b·p − q) / b

        Parameters
        ----------
        win_rate : float
            Probability of a winning trade (0–1).
        avg_win : float
            Average profit on a winning trade (absolute, positive).
        avg_loss : float
            Average loss on a losing trade (absolute, positive).

        Returns
        -------
        float
            Optimal fraction of bankroll to risk.  Can be negative
            (meaning the edge is negative — do not trade).
        """
        if avg_loss <= 0:
            logger.warning("avg_loss must be > 0; returning 0.")
            return 0.0
        if not 0 <= win_rate <= 1:
            logger.warning("win_rate %.4f out of [0, 1]; clamping.", win_rate)
            win_rate = max(0.0, min(1.0, win_rate))

        b = avg_win / avg_loss          # odds ratio
        p = win_rate
        q = 1.0 - p
        kelly = (b * p - q) / b
        logger.debug(
            "Kelly  b=%.3f  p=%.3f  q=%.3f  f*=%.4f", b, p, q, kelly
        )
        return kelly

    @staticmethod
    def fractional_kelly(full_kelly: float, fraction: float) -> float:
        """
        Scale full Kelly by a safety fraction (0.25 = quarter-Kelly).

        Returns 0 when the raw Kelly is non-positive (no edge).
        """
        if full_kelly <= 0:
            return 0.0
        return full_kelly * max(0.0, min(1.0, fraction))

    # ------------------------------------------------------------------ #
    # Dynamic adjustments                                                  #
    # ------------------------------------------------------------------ #

    def dynamic_position_size(
        self,
        signal_confidence: float,
        volatility: float,
        drawdown_pct: float,
        portfolio_correlation: float,
    ) -> float:
        """
        Compute a multiplicative adjustment factor ∈ (0, 1] applied to
        the fractional Kelly size.

        Parameters
        ----------
        signal_confidence : float
            Model confidence score 0–1 (higher → larger position).
        volatility : float
            Annualised realised volatility of the token (0–∞).
            Higher vol → smaller position.
        drawdown_pct : float
            Current portfolio drawdown from peak (0–1, e.g. 0.05 = 5 %).
        portfolio_correlation : float
            Average correlation of this token with existing holdings
            (0–1).  Higher corr → smaller position.

        Returns
        -------
        float
            Adjustment multiplier in (0, 1].
        """
        # --- confidence factor (linear) ---
        conf_factor = max(0.1, min(1.0, signal_confidence))

        # --- volatility factor (inverse sigmoid-like) ---
        # Vol of 0.50 (50 % annual) is baseline = 1.0 factor
        vol_baseline = 0.50
        if volatility <= 0:
            vol_factor = 1.0
        else:
            vol_factor = min(1.0, vol_baseline / volatility)

        # --- drawdown factor ---
        # Scale linearly: at 20 % drawdown, reduce to 0.  Mild drawdown
        # (< 2 %) has no effect.
        dd_threshold = 0.02
        dd_max = 0.20
        if drawdown_pct <= dd_threshold:
            dd_factor = 1.0
        elif drawdown_pct >= dd_max:
            dd_factor = 0.0
        else:
            dd_factor = 1.0 - (drawdown_pct - dd_threshold) / (dd_max - dd_threshold)

        # --- correlation factor ---
        corr_factor = max(0.3, 1.0 - portfolio_correlation * 0.7)

        adjustment = conf_factor * vol_factor * dd_factor * corr_factor
        adjustment = max(0.0, min(1.0, adjustment))

        logger.debug(
            "Dynamic adj  conf=%.2f  vol=%.2f  dd=%.2f  corr=%.2f → %.4f",
            conf_factor, vol_factor, dd_factor, corr_factor, adjustment,
        )
        return adjustment

    # ------------------------------------------------------------------ #
    # Concentration checks                                                 #
    # ------------------------------------------------------------------ #

    def check_concentration(
        self,
        token: str,
        current_portfolio: List[PortfolioHolding],
    ) -> float:
        """
        Return the maximum additional allocation (as fraction of portfolio)
        allowed for *token* given current holdings and concentration limits.

        If the token is already at or beyond the single-position cap, returns 0.
        """
        token_weight = 0.0
        group_weight = 0.0
        token_group: Optional[str] = None

        for h in current_portfolio:
            if h.token == token:
                token_weight += h.weight
                token_group = h.correlation_group
            elif token_group and h.correlation_group == token_group:
                group_weight += h.weight

        # how much room under single-position cap?
        single_room = max(0.0, self.max_position_pct - token_weight)

        # how much room under the cluster concentration cap?
        cluster_total = token_weight + group_weight
        cluster_room = max(0.0, self.max_concentration - cluster_total)

        allowed = min(single_room, cluster_room)
        logger.debug(
            "Concentration check  token=%s  single_room=%.4f  "
            "cluster_room=%.4f  allowed=%.4f",
            token, single_room, cluster_room, allowed,
        )
        return allowed

    # ------------------------------------------------------------------ #
    # Top-level entry point                                                #
    # ------------------------------------------------------------------ #

    def get_optimal_size(
        self,
        signal: Dict,
        portfolio_state: Dict,
    ) -> PositionSize:
        """
        Determine the optimal position size for a new trade.

        Parameters
        ----------
        signal : dict
            Must contain at minimum:
              - token (str): token mint address
              - win_rate (float): estimated probability of win (0–1)
              - avg_win (float): average win amount
              - avg_loss (float): average loss amount
              - confidence (float): model confidence 0–1
              - volatility (float): annualised vol
        portfolio_state : dict
            Must contain:
              - total_value (float): portfolio total value in USD
              - drawdown_pct (float): current drawdown from peak (0–1)
              - holdings (List[PortfolioHolding]): existing positions
              - correlation (float): avg correlation with new token
        """
        token: str = signal["token"]
        total_value: float = portfolio_state["total_value"]

        if total_value <= 0:
            logger.warning("Portfolio value is zero or negative; cannot size.")
            return PositionSize(token, 0.0, 0.0, 0.0, 0.0, 1.0)

        # 1. Raw Kelly
        full_kelly = self.kelly_criterion(
            signal["win_rate"], signal["avg_win"], signal["avg_loss"]
        )
        frac_kelly = self.fractional_kelly(full_kelly, self.kelly_fraction)

        # 2. Dynamic adjustment
        adjustment = self.dynamic_position_size(
            signal_confidence=signal.get("confidence", 0.5),
            volatility=signal.get("volatility", 0.5),
            drawdown_pct=portfolio_state.get("drawdown_pct", 0.0),
            portfolio_correlation=portfolio_state.get("correlation", 0.0),
        )

        adjusted_pct = frac_kelly * adjustment

        # 3. Concentration limit
        holdings = portfolio_state.get("holdings", [])
        max_allowed = self.check_concentration(token, holdings)
        adjusted_pct = min(adjusted_pct, max_allowed)

        # 4. Hard caps
        adjusted_pct = min(adjusted_pct, self.max_position_pct)
        size_usd = adjusted_pct * total_value
        size_usd = min(size_usd, self.max_position_usd)

        # 5. Minimum threshold
        if size_usd < self.min_position_usd:
            logger.info(
                "Computed size $%.2f below min $%.2f for %s; skipping.",
                size_usd, self.min_position_usd, token,
            )
            size_usd = 0.0
            adjusted_pct = 0.0

        # Re-derive pct after USD cap
        if total_value > 0 and size_usd > 0:
            adjusted_pct = size_usd / total_value

        risk_score = 1.0 - adjustment  # higher score = riskier

        result = PositionSize(
            token=token,
            size_usd=size_usd,
            size_pct=adjusted_pct,
            kelly_fraction=frac_kelly,
            confidence_adjustment=adjustment,
            risk_score=risk_score,
        )
        logger.info(
            "Optimal size  token=%s  $%.2f (%.2f%%)  kelly=%.4f  adj=%.4f  risk=%.4f",
            token, result.size_usd, result.size_pct * 100,
            result.kelly_fraction, result.confidence_adjustment, result.risk_score,
        )
        return result
