# trading/analysis/behavioral_patterns.py
"""
Behavioral pattern detection for Solana token markets.

Detects accumulation / distribution phases, wash-trading rings, insider
activity around announcements, and unusual DEX-router usage patterns.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from trading.solana_client import SolanaClient
import config

logger = logging.getLogger(__name__)

# Well-known Solana DEX router program IDs.
_KNOWN_ROUTERS: dict[str, str] = {
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter v6",
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcPX73": "Jupiter v4",
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium AMM",
    "27haf8L6oxUeXrHrgEgsexjSY5hbVUWEmvv9Nyxg8vQv": "Raydium v4",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca Whirlpool",
}


# --------------------------------------------------------------------- #
#  Data classes                                                          #
# --------------------------------------------------------------------- #

@dataclass
class PhaseDetection:
    """Detection of an accumulation or distribution phase."""

    token: str
    phase: str  # "accumulation" | "distribution"
    confidence: float
    start_time: float
    end_time: float
    volume_profile: list[float]


@dataclass
class WashTradeAlert:
    """Alert indicating suspected wash-trading activity."""

    token: str
    wallets: list[str]
    volume: float
    confidence: float
    pattern_type: str  # "self_trade" | "round_trip" | "volume_no_impact"


# --------------------------------------------------------------------- #
#  BehavioralAnalyzer                                                    #
# --------------------------------------------------------------------- #

class BehavioralAnalyzer:
    """Detect behavioural patterns in Solana token trading data.

    The analyser operates on pre-fetched trade / transaction lists rather
    than calling the RPC directly, making it easy to test and compose
    with the rest of the analysis pipeline.
    """

    def __init__(
        self,
        solana_client: Optional[SolanaClient] = None,
        lookback_periods: int = 30,
        z_threshold: float = 2.0,
    ) -> None:
        self._client = solana_client or SolanaClient()
        self.lookback_periods = lookback_periods
        self.z_threshold = z_threshold

    # ------------------------------------------------------------------ #
    #  Accumulation detection                                              #
    # ------------------------------------------------------------------ #

    def detect_accumulation(
        self, token_data: list[dict]
    ) -> Optional[PhaseDetection]:
        """Detect an accumulation phase in *token_data*.

        An accumulation phase is characterised by steadily **increasing
        buy volume** while the price remains flat or slightly declining.
        The method uses a rolling-window approach to compute volume
        trends and price slope over the lookback window.

        Each element of *token_data* should contain at minimum::

            {"timestamp": float, "price": float, "volume": float,
             "direction": "buy"|"sell"}

        Args:
            token_data: Chronologically sorted trade records.

        Returns:
            A :class:`PhaseDetection` with ``phase="accumulation"`` if
            the pattern is detected, otherwise ``None``.
        """
        if len(token_data) < self.lookback_periods:
            return None

        try:
            window = token_data[-self.lookback_periods:]

            # Separate buy volumes per period.
            buy_volumes: list[float] = [
                d.get("volume", 0.0)
                for d in window
                if d.get("direction") == "buy"
            ]
            prices: list[float] = [d.get("price", 0.0) for d in window]

            if len(buy_volumes) < 3 or len(prices) < 3:
                return None

            # Volume trend: linear-regression slope on buy volumes.
            vol_slope = self._linear_slope(buy_volumes)

            # Price trend: slope on prices.
            price_slope = self._linear_slope(prices)
            price_mean = np.mean(prices)
            norm_price_slope = (
                price_slope / price_mean if price_mean else 0.0
            )

            # Accumulation heuristic: rising buy volume + flat/falling price.
            is_rising_volume = vol_slope > 0
            is_flat_or_falling_price = norm_price_slope <= 0.02

            if not (is_rising_volume and is_flat_or_falling_price):
                return None

            # Confidence: combine volume-trend strength and price flatness.
            vol_strength = min(abs(vol_slope) / (np.mean(buy_volumes) + 1e-9), 1.0)
            price_flatness = max(1.0 - abs(norm_price_slope) * 20, 0.0)
            confidence = 0.6 * vol_strength + 0.4 * price_flatness
            confidence = round(min(max(confidence, 0.0), 1.0), 4)

            volume_profile = [d.get("volume", 0.0) for d in window]

            return PhaseDetection(
                token=window[0].get("token", ""),
                phase="accumulation",
                confidence=confidence,
                start_time=window[0].get("timestamp", 0.0),
                end_time=window[-1].get("timestamp", 0.0),
                volume_profile=volume_profile,
            )

        except Exception:
            logger.exception("Error detecting accumulation phase")
            return None

    # ------------------------------------------------------------------ #
    #  Distribution detection                                              #
    # ------------------------------------------------------------------ #

    def detect_distribution(
        self, token_data: list[dict]
    ) -> Optional[PhaseDetection]:
        """Detect a distribution phase in *token_data*.

        Distribution is signalled by **large sell volumes** coinciding
        with a rising (or topping) price, or sudden volume spikes at
        local price highs.

        Args:
            token_data: Chronologically sorted trade records (same
                schema as :meth:`detect_accumulation`).

        Returns:
            A :class:`PhaseDetection` with ``phase="distribution"`` if
            the pattern is detected, otherwise ``None``.
        """
        if len(token_data) < self.lookback_periods:
            return None

        try:
            window = token_data[-self.lookback_periods:]

            sell_volumes: list[float] = [
                d.get("volume", 0.0)
                for d in window
                if d.get("direction") == "sell"
            ]
            prices: list[float] = [d.get("price", 0.0) for d in window]

            if len(sell_volumes) < 3 or len(prices) < 3:
                return None

            vol_slope = self._linear_slope(sell_volumes)
            price_slope = self._linear_slope(prices)
            price_mean = np.mean(prices)
            norm_price_slope = (
                price_slope / price_mean if price_mean else 0.0
            )

            # Distribution heuristic: rising sell volume + rising or
            # topping price.
            is_rising_sell = vol_slope > 0
            is_price_rising_or_flat = norm_price_slope >= -0.02

            # Also flag volume spikes at price highs.
            all_volumes = np.array(
                [d.get("volume", 0.0) for d in window], dtype=np.float64
            )
            vol_mean = np.mean(all_volumes)
            vol_std = np.std(all_volumes)
            has_spike = bool(
                vol_std > 0
                and np.any(all_volumes > vol_mean + self.z_threshold * vol_std)
            )

            if not (is_rising_sell and is_price_rising_or_flat) and not has_spike:
                return None

            vol_strength = min(
                abs(vol_slope) / (np.mean(sell_volumes) + 1e-9), 1.0
            )
            spike_signal = 0.0
            if has_spike and vol_std > 0:
                spike_signal = min(
                    float(np.max(all_volumes) - vol_mean) / (vol_std * 3),
                    1.0,
                )
            confidence = 0.5 * vol_strength + 0.3 * spike_signal + 0.2
            confidence = round(min(max(confidence, 0.0), 1.0), 4)

            volume_profile = [d.get("volume", 0.0) for d in window]

            return PhaseDetection(
                token=window[0].get("token", ""),
                phase="distribution",
                confidence=confidence,
                start_time=window[0].get("timestamp", 0.0),
                end_time=window[-1].get("timestamp", 0.0),
                volume_profile=volume_profile,
            )

        except Exception:
            logger.exception("Error detecting distribution phase")
            return None

    # ------------------------------------------------------------------ #
    #  Wash-trading detection                                              #
    # ------------------------------------------------------------------ #

    def identify_wash_trading(
        self, trades: list[dict]
    ) -> list[WashTradeAlert]:
        """Identify wash-trading patterns in *trades*.

        Three pattern types are checked:

        1. **Self-trades** – the same wallet appears on both sides of a
           swap (``wallet_in == wallet_out``).
        2. **Round-trip patterns** – A sends to B and B sends back to A
           within a short time window (10 minutes).
        3. **Volume without price impact** – high trading volume that
           produces negligible price movement, suggesting artificial
           volume inflation.

        Each element of *trades* should include::

            {"timestamp": float, "token": str, "wallet": str,
             "direction": "buy"|"sell", "amount": float,
             "price": float}

        Optionally ``wallet_in`` and ``wallet_out`` for counterparty
        resolution.

        Args:
            trades: Chronologically sorted trade records.

        Returns:
            List of :class:`WashTradeAlert` objects.
        """
        alerts: list[WashTradeAlert] = []

        try:
            # --- Pattern 1: self-trades ---
            self_trade_wallets: dict[str, float] = defaultdict(float)
            for t in trades:
                w_in = t.get("wallet_in", "")
                w_out = t.get("wallet_out", "")
                if w_in and w_out and w_in == w_out:
                    self_trade_wallets[w_in] += t.get("amount", 0.0)

            for wallet, volume in self_trade_wallets.items():
                alerts.append(
                    WashTradeAlert(
                        token=trades[0].get("token", "") if trades else "",
                        wallets=[wallet],
                        volume=volume,
                        confidence=0.95,
                        pattern_type="self_trade",
                    )
                )

            # --- Pattern 2: round-trip (A→B→A within 600 s) ---
            round_trip_window = 600.0  # seconds
            transfer_pairs: dict[
                tuple[str, str], list[dict]
            ] = defaultdict(list)

            for t in trades:
                sender = t.get("wallet", t.get("wallet_in", ""))
                receiver = t.get("wallet_out", "")
                if sender and receiver and sender != receiver:
                    transfer_pairs[(sender, receiver)].append(t)

            seen_round_trips: set[tuple[str, str]] = set()
            for (a, b), fwd_list in transfer_pairs.items():
                rev_list = transfer_pairs.get((b, a), [])
                if not rev_list:
                    continue

                for fwd in fwd_list:
                    for rev in rev_list:
                        dt = abs(
                            rev.get("timestamp", 0) - fwd.get("timestamp", 0)
                        )
                        if dt <= round_trip_window:
                            pair_key = tuple(sorted((a, b)))
                            if pair_key not in seen_round_trips:
                                seen_round_trips.add(pair_key)
                                combined_vol = fwd.get(
                                    "amount", 0
                                ) + rev.get("amount", 0)
                                # Confidence scales with how quickly the
                                # round-trip completes.
                                conf = max(
                                    0.5,
                                    1.0 - dt / round_trip_window,
                                )
                                alerts.append(
                                    WashTradeAlert(
                                        token=fwd.get("token", ""),
                                        wallets=sorted([a, b]),
                                        volume=combined_vol,
                                        confidence=round(conf, 4),
                                        pattern_type="round_trip",
                                    )
                                )
                            break  # one alert per pair suffices

            # --- Pattern 3: volume without price impact ---
            if len(trades) >= 5:
                token_groups: dict[str, list[dict]] = defaultdict(list)
                for t in trades:
                    token_groups[t.get("token", "unknown")].append(t)

                for token, group in token_groups.items():
                    prices = [g.get("price", 0.0) for g in group if g.get("price")]
                    volumes = [g.get("amount", 0.0) for g in group]

                    if len(prices) < 2:
                        continue

                    price_arr = np.array(prices, dtype=np.float64)
                    price_change = abs(price_arr[-1] - price_arr[0])
                    price_range = np.ptp(price_arr)
                    avg_price = np.mean(price_arr)

                    total_volume = sum(volumes)
                    norm_price_change = (
                        price_change / avg_price if avg_price else 0.0
                    )

                    # High volume but negligible price movement.
                    if total_volume > 0 and norm_price_change < 0.005:
                        wallets_involved = list(
                            {g.get("wallet", "") for g in group if g.get("wallet")}
                        )
                        conf = min(
                            0.3
                            + 0.7 * (1.0 - norm_price_change / 0.005),
                            1.0,
                        )
                        alerts.append(
                            WashTradeAlert(
                                token=token,
                                wallets=sorted(wallets_involved),
                                volume=total_volume,
                                confidence=round(conf, 4),
                                pattern_type="volume_no_impact",
                            )
                        )

        except Exception:
            logger.exception("Error in wash-trading detection")

        return alerts

    # ------------------------------------------------------------------ #
    #  Insider-activity detection                                          #
    # ------------------------------------------------------------------ #

    def detect_insider_activity(
        self,
        trades: list[dict],
        announcements: list[dict],
    ) -> list[dict]:
        """Find abnormal trading activity preceding announcements.

        For each announcement, the method inspects a pre-event window
        (default: 1 hour) and flags trades whose volume or unique-wallet
        count exceeds the z-score threshold.

        *trades* elements::

            {"timestamp": float, "wallet": str, "volume": float, …}

        *announcements* elements::

            {"timestamp": float, "description": str, …}

        Args:
            trades: Chronologically sorted trade records.
            announcements: Known announcement / event timestamps.

        Returns:
            List of dicts, each containing:

            - **announcement** – the matched announcement dict.
            - **suspicious_wallets** – wallets active in the pre-event
              window.
            - **pre_event_volume** – total volume in the pre-event
              window.
            - **z_score_volume** – z-score of the pre-event volume
              relative to normal baseline periods.
            - **z_score_wallets** – z-score of unique wallet count.
            - **confidence** – composite confidence score (0–1).
        """
        results: list[dict] = []

        if not trades or not announcements:
            return results

        try:
            pre_event_secs = 3600.0  # 1-hour look-back before event
            baseline_secs = 86400.0  # 24-hour baseline window

            timestamps = np.array(
                [t.get("timestamp", 0.0) for t in trades], dtype=np.float64
            )

            for ann in announcements:
                ann_ts = ann.get("timestamp", 0.0)
                if ann_ts == 0.0:
                    continue

                # Pre-event window.
                pre_mask = (timestamps >= ann_ts - pre_event_secs) & (
                    timestamps < ann_ts
                )
                pre_trades = [
                    t for t, m in zip(trades, pre_mask) if m
                ]

                # Baseline window (24 h before the pre-event window).
                base_start = ann_ts - pre_event_secs - baseline_secs
                base_end = ann_ts - pre_event_secs
                base_mask = (timestamps >= base_start) & (
                    timestamps < base_end
                )
                base_trades = [
                    t for t, m in zip(trades, base_mask) if m
                ]

                if not base_trades:
                    continue

                # Compute hourly volume buckets for baseline.
                n_hours = max(int(baseline_secs / 3600), 1)
                hourly_volumes: list[float] = [0.0] * n_hours
                hourly_wallets: list[int] = [0] * n_hours

                for bt in base_trades:
                    hour_idx = min(
                        int(
                            (bt.get("timestamp", 0.0) - base_start) / 3600
                        ),
                        n_hours - 1,
                    )
                    hourly_volumes[hour_idx] += bt.get("volume", 0.0)

                # Unique wallets per hour in baseline.
                hourly_wallet_sets: list[set[str]] = [
                    set() for _ in range(n_hours)
                ]
                for bt in base_trades:
                    hour_idx = min(
                        int(
                            (bt.get("timestamp", 0.0) - base_start) / 3600
                        ),
                        n_hours - 1,
                    )
                    hourly_wallet_sets[hour_idx].add(
                        bt.get("wallet", "")
                    )
                hourly_wallets = [len(s) for s in hourly_wallet_sets]

                vol_arr = np.array(hourly_volumes, dtype=np.float64)
                wal_arr = np.array(hourly_wallets, dtype=np.float64)

                vol_mean, vol_std = float(np.mean(vol_arr)), float(
                    np.std(vol_arr)
                )
                wal_mean, wal_std = float(np.mean(wal_arr)), float(
                    np.std(wal_arr)
                )

                pre_volume = sum(t.get("volume", 0.0) for t in pre_trades)
                pre_wallets = {
                    t.get("wallet", "") for t in pre_trades
                }

                z_vol = (
                    (pre_volume - vol_mean) / vol_std if vol_std > 0 else 0.0
                )
                z_wal = (
                    (len(pre_wallets) - wal_mean) / wal_std
                    if wal_std > 0
                    else 0.0
                )

                if z_vol >= self.z_threshold or z_wal >= self.z_threshold:
                    max_z = max(z_vol, z_wal)
                    confidence = round(
                        min(max_z / (self.z_threshold * 2), 1.0), 4
                    )

                    results.append(
                        {
                            "announcement": ann,
                            "suspicious_wallets": sorted(pre_wallets),
                            "pre_event_volume": round(pre_volume, 4),
                            "z_score_volume": round(z_vol, 4),
                            "z_score_wallets": round(z_wal, 4),
                            "confidence": confidence,
                        }
                    )

        except Exception:
            logger.exception("Error detecting insider activity")

        return results

    # ------------------------------------------------------------------ #
    #  DEX-router pattern analysis                                         #
    # ------------------------------------------------------------------ #

    def analyze_router_patterns(
        self, transactions: list[dict]
    ) -> dict:
        """Profile DEX router usage across *transactions* and flag
        unusual patterns.

        The analysis includes:

        * Per-router transaction counts and volume share.
        * Detection of *router-hopping* (wallets switching routers
          frequently, possibly to evade detection).
        * Detection of *exclusive-router* wallets (wallets that use
          only a single obscure router, possibly a custom bot).

        Args:
            transactions: Decoded transaction dicts.

        Returns:
            Dict with keys:

            - **router_distribution** – ``{router_name: count}``.
            - **router_volume** – ``{router_name: total_volume}``.
            - **router_hopping_wallets** – list of wallets using 3+
              distinct routers.
            - **exclusive_router_wallets** – ``{wallet: router_name}``
              for wallets using only one non-major router.
            - **anomaly_score** – 0–1 composite anomaly indicator.
        """
        router_counts: dict[str, int] = defaultdict(int)
        router_volume: dict[str, float] = defaultdict(float)
        wallet_routers: dict[str, set[str]] = defaultdict(set)

        try:
            for tx in transactions:
                msg = tx.get("transaction", {}).get("message", {})
                keys_raw = msg.get("accountKeys", [])
                keys: list[str] = [
                    (k.get("pubkey", k) if isinstance(k, dict) else str(k))
                    for k in keys_raw
                ]

                # Identify which router was used in this tx.
                matched_router: Optional[str] = None
                for prog_id, name in _KNOWN_ROUTERS.items():
                    if prog_id in keys:
                        matched_router = name
                        break

                if not matched_router:
                    continue

                router_counts[matched_router] += 1

                # Extract wallet (fee payer = first signer).
                wallet = keys[0] if keys else ""

                # Estimate volume from balance changes.
                meta = tx.get("meta", {})
                pre_bal = meta.get("preBalances", [])
                post_bal = meta.get("postBalances", [])
                vol = 0.0
                if pre_bal and post_bal:
                    deltas = [
                        abs(post_bal[i] - pre_bal[i])
                        for i in range(min(len(pre_bal), len(post_bal)))
                    ]
                    vol = max(deltas) / 1e9 if deltas else 0.0

                router_volume[matched_router] += vol
                if wallet:
                    wallet_routers[wallet].add(matched_router)

            # Router-hopping: wallets using 3+ distinct routers.
            hoppers = [
                w for w, routers in wallet_routers.items() if len(routers) >= 3
            ]

            # Exclusive-router: wallets using only one non-major router.
            major_routers = {"Jupiter v6", "Raydium AMM", "Orca"}
            exclusive: dict[str, str] = {}
            for w, routers in wallet_routers.items():
                if len(routers) == 1:
                    (router,) = routers
                    if router not in major_routers:
                        exclusive[w] = router

            # Anomaly score.
            total_tx = sum(router_counts.values()) or 1
            hopper_ratio = len(hoppers) / max(len(wallet_routers), 1)
            exclusive_ratio = len(exclusive) / max(len(wallet_routers), 1)
            anomaly_score = round(
                min(0.5 * hopper_ratio + 0.5 * exclusive_ratio, 1.0), 4
            )

        except Exception:
            logger.exception("Error analysing router patterns")
            return {
                "router_distribution": {},
                "router_volume": {},
                "router_hopping_wallets": [],
                "exclusive_router_wallets": {},
                "anomaly_score": 0.0,
            }

        return {
            "router_distribution": dict(router_counts),
            "router_volume": {k: round(v, 4) for k, v in router_volume.items()},
            "router_hopping_wallets": sorted(hoppers),
            "exclusive_router_wallets": exclusive,
            "anomaly_score": anomaly_score,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _linear_slope(values: list[float]) -> float:
        """Compute the OLS slope of *values* indexed 0 … n-1.

        Uses the closed-form formula for simple linear regression to
        avoid pulling in a full regression library for a single
        coefficient.

        Args:
            values: Ordered numeric observations.

        Returns:
            Slope coefficient (units per index step).
        """
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(values, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = float(np.sum((x - x_mean) * (y - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))
        if denominator == 0:
            return 0.0
        return numerator / denominator
