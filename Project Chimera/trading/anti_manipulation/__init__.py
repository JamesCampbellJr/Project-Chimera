"""
Anti-manipulation detection suite for Solana trading.

Provides anomaly detection, fake trade / honeypot identification,
and MEV bot detection to protect the paper-trading bot.
"""

from trading.anti_manipulation.anomaly_detector import AnomalyDetector
from trading.anti_manipulation.fake_trade_detector import FakeTradeDetector
from trading.anti_manipulation.mev_detector import MEVDetector

__all__ = ["AnomalyDetector", "FakeTradeDetector", "MEVDetector"]
