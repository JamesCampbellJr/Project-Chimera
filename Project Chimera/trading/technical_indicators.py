# trading/technical_indicators.py
"""
Technical Indicators — computes classical technical analysis indicators
adapted for cryptocurrency trading on Solana.

All indicators operate on price history lists (newest last) and return
scalar values or dicts.  The module stores computed values in the
database for historical reference and pattern learning.

Supported indicators
--------------------
* RSI (Relative Strength Index)      — momentum oscillator (0–100)
* MACD (Moving Average Convergence Divergence) — trend/momentum
* Bollinger Bands                    — volatility bands around SMA
* Volume SMA                        — simple moving average of volume
* Price Momentum                    — rate-of-change indicator
* Volatility (Std Dev)              — rolling standard deviation of returns
"""

import logging
import math
from datetime import datetime, timezone
from typing import Optional

import config
from trading.database import init_db, insert_technical_indicator, get_latest_indicator

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Computes and stores technical analysis indicators for tokens.
    """

    def __init__(self):
        self.conn = init_db()
        logger.info("TechnicalIndicators initialised.")

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def compute_all(
        self,
        token_address: str,
        prices: list[float],
        volumes: Optional[list[float]] = None,
    ) -> dict:
        """
        Compute all indicators for a token given its price history.

        Parameters
        ----------
        token_address : str
            The token mint address.
        prices : list[float]
            Price history, oldest first, newest last.
        volumes : list[float] or None
            Volume history, same ordering as prices.

        Returns a dict of indicator names → values.
        """
        results: dict = {}

        if len(prices) < 2:
            logger.debug("Not enough price data for indicators (need >= 2).")
            return results

        # RSI
        rsi = self.compute_rsi(prices)
        if rsi is not None:
            results["RSI"] = rsi
            self._store(token_address, "RSI", value=rsi)

        # MACD
        macd = self.compute_macd(prices)
        if macd:
            results["MACD"] = macd
            self._store(
                token_address, "MACD",
                value=macd.get("macd_line"),
                signal_value=macd.get("signal_line"),
            )

        # Bollinger Bands
        bbands = self.compute_bollinger_bands(prices)
        if bbands:
            results["BBANDS"] = bbands
            self._store(
                token_address, "BBANDS",
                value=bbands.get("middle"),
                upper_band=bbands.get("upper"),
                lower_band=bbands.get("lower"),
            )

        # Volume SMA
        if volumes and len(volumes) >= 5:
            vol_sma = self.compute_volume_sma(volumes)
            results["VOLUME_SMA"] = vol_sma
            self._store(token_address, "VOLUME_SMA", value=vol_sma)

        # Price Momentum
        momentum = self.compute_momentum(prices)
        if momentum is not None:
            results["MOMENTUM"] = momentum
            self._store(token_address, "MOMENTUM", value=momentum)

        # Volatility
        volatility = self.compute_volatility(prices)
        if volatility is not None:
            results["VOLATILITY"] = volatility
            self._store(token_address, "VOLATILITY", value=volatility)

        logger.debug(
            "Computed %d indicators for %s.", len(results), token_address,
        )
        return results

    def get_trading_signal(
        self,
        token_address: str,
        prices: list[float],
        volumes: Optional[list[float]] = None,
    ) -> dict:
        """
        Generate a composite trading signal from technical indicators.

        Returns a dict with:
        - signal: 'BUY', 'SELL', or 'HOLD'
        - confidence: 0.0–1.0
        - reasons: list of contributing factors
        """
        indicators = self.compute_all(token_address, prices, volumes)

        if not indicators:
            return {"signal": "HOLD", "confidence": 0.0, "reasons": ["Insufficient data"]}

        buy_signals: list[str] = []
        sell_signals: list[str] = []

        # RSI analysis
        rsi = indicators.get("RSI")
        if rsi is not None:
            if rsi < config.RSI_OVERSOLD:
                buy_signals.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > config.RSI_OVERBOUGHT:
                sell_signals.append(f"RSI overbought ({rsi:.1f})")

        # MACD analysis
        macd = indicators.get("MACD")
        if macd:
            histogram = macd.get("histogram", 0)
            if histogram > 0:
                buy_signals.append(f"MACD bullish (histogram={histogram:.4f})")
            elif histogram < 0:
                sell_signals.append(f"MACD bearish (histogram={histogram:.4f})")

        # Bollinger Bands analysis
        bbands = indicators.get("BBANDS")
        if bbands and prices:
            current_price = prices[-1]
            lower = bbands.get("lower", 0)
            upper = bbands.get("upper", 0)
            if current_price <= lower:
                buy_signals.append("Price at lower Bollinger Band")
            elif current_price >= upper:
                sell_signals.append("Price at upper Bollinger Band")

        # Momentum analysis
        momentum = indicators.get("MOMENTUM")
        if momentum is not None:
            if momentum > config.MOMENTUM_BUY_THRESHOLD:
                buy_signals.append(f"Strong upward momentum ({momentum:.2f}%)")
            elif momentum < config.MOMENTUM_SELL_THRESHOLD:
                sell_signals.append(f"Strong downward momentum ({momentum:.2f}%)")

        # Determine composite signal
        total_signals = len(buy_signals) + len(sell_signals)
        if total_signals == 0:
            return {"signal": "HOLD", "confidence": 0.0, "reasons": ["No clear signals"]}

        if len(buy_signals) > len(sell_signals):
            confidence = len(buy_signals) / total_signals
            return {
                "signal": "BUY",
                "confidence": min(confidence, 1.0),
                "reasons": buy_signals,
            }
        elif len(sell_signals) > len(buy_signals):
            confidence = len(sell_signals) / total_signals
            return {
                "signal": "SELL",
                "confidence": min(confidence, 1.0),
                "reasons": sell_signals,
            }
        else:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reasons": ["Mixed signals"] + buy_signals + sell_signals,
            }

    # ---------------------------------------------------------------------- #
    # Individual indicator computations                                        #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def compute_rsi(prices: list[float], period: int = 14) -> Optional[float]:
        """
        Compute the Relative Strength Index (RSI).

        Returns a value between 0 and 100, or None if insufficient data.
        """
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        recent = deltas[-(period):]

        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]

        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0

        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def compute_macd(
        prices: list[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Optional[dict]:
        """
        Compute MACD line, signal line, and histogram.

        Returns a dict or None if insufficient data.
        """
        if len(prices) < slow_period + signal_period:
            return None

        def ema(data: list[float], period: int) -> list[float]:
            """Compute exponential moving average."""
            if not data:
                return []
            multiplier = 2.0 / (period + 1)
            result = [data[0]]
            for i in range(1, len(data)):
                val = (data[i] - result[-1]) * multiplier + result[-1]
                result.append(val)
            return result

        fast_ema = ema(prices, fast_period)
        slow_ema = ema(prices, slow_period)

        # MACD line = fast EMA - slow EMA
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]

        # Signal line = EMA of full MACD line history
        signal_line = ema(macd_line, signal_period)

        if not signal_line:
            return None

        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        histogram = current_macd - current_signal

        return {
            "macd_line": current_macd,
            "signal_line": current_signal,
            "histogram": histogram,
        }

    @staticmethod
    def compute_bollinger_bands(
        prices: list[float],
        period: int = 20,
        num_std: float = 2.0,
    ) -> Optional[dict]:
        """
        Compute Bollinger Bands (middle, upper, lower).

        Returns a dict or None if insufficient data.
        """
        if len(prices) < period:
            return None

        recent = prices[-period:]
        middle = sum(recent) / period

        variance = sum((p - middle) ** 2 for p in recent) / period
        std_dev = math.sqrt(variance)

        return {
            "upper": middle + num_std * std_dev,
            "middle": middle,
            "lower": middle - num_std * std_dev,
            "std_dev": std_dev,
        }

    @staticmethod
    def compute_volume_sma(
        volumes: list[float],
        period: int = 20,
    ) -> float:
        """Compute the simple moving average of volume."""
        recent = volumes[-period:] if len(volumes) >= period else volumes
        return sum(recent) / len(recent) if recent else 0.0

    @staticmethod
    def compute_momentum(
        prices: list[float],
        period: int = 10,
    ) -> Optional[float]:
        """
        Compute price momentum (rate of change) as a percentage.

        momentum = (current_price - price_N_periods_ago) / price_N_periods_ago * 100
        """
        if len(prices) < period + 1:
            return None

        old_price = prices[-(period + 1)]
        current_price = prices[-1]

        if old_price == 0:
            return None

        return (current_price - old_price) / old_price * 100.0

    @staticmethod
    def compute_volatility(
        prices: list[float],
        period: int = 20,
    ) -> Optional[float]:
        """
        Compute rolling volatility as the standard deviation of returns.

        Returns a float (higher = more volatile).
        """
        if len(prices) < period + 1:
            return None

        recent = prices[-(period + 1):]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] != 0:
                ret = (recent[i] - recent[i - 1]) / recent[i - 1]
                returns.append(ret)

        if not returns:
            return None

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _store(
        self,
        token_address: str,
        indicator_name: str,
        value: Optional[float] = None,
        signal_value: Optional[float] = None,
        upper_band: Optional[float] = None,
        lower_band: Optional[float] = None,
    ) -> None:
        """Persist an indicator value to the database."""
        insert_technical_indicator(self.conn, {
            "token_address": token_address,
            "indicator_name": indicator_name,
            "value": value,
            "signal_value": signal_value,
            "upper_band": upper_band,
            "lower_band": lower_band,
        })
