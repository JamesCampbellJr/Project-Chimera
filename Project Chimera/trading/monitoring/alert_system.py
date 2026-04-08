"""Alert system for trading bot infrastructure monitoring.

Provides configurable alert rules with threshold-based triggers, cooldown
periods to prevent alert fatigue, acknowledgment tracking, and webhook
notification support.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Severity levels for alerts, ordered by severity."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class AlertRule:
    """Definition of an alert rule.

    Attributes:
        name: Human-readable rule name.
        metric: Name of the metric to watch.
        condition: Comparison operator (``>``, ``<``, ``>=``, ``<=``, ``==``).
        threshold: Numeric threshold for the condition.
        level: Severity level when the rule fires.
        cooldown_seconds: Minimum seconds between consecutive alerts from this rule.
        message_template: Template string; ``{value}`` and ``{threshold}`` are substituted.
    """
    name: str
    metric: str
    condition: str
    threshold: float
    level: AlertLevel = AlertLevel.WARNING
    cooldown_seconds: int = 300
    message_template: str = "Alert: {name} - {metric} is {value} (threshold: {threshold})"


@dataclass
class Alert:
    """An alert instance generated when a rule fires."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str = ""
    level: AlertLevel = AlertLevel.WARNING
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    value: float = 0.0
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None


class AlertSystem:
    """Manages alert rules, evaluates metrics, and dispatches notifications.

    Args:
        webhook_url: Optional default webhook URL for alert notifications.
    """

    # Comparison operators
    _OPERATORS: Dict[str, Callable[[float, float], bool]] = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
    }

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._last_triggered: Dict[str, float] = {}
        self._webhook_url = webhook_url
        logger.info("AlertSystem initialised (webhook=%s)", "yes" if webhook_url else "no")

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule (overwrites existing rule with same name).

        Args:
            rule: The ``AlertRule`` to register.
        """
        self._rules[rule.name] = rule
        logger.info("Added alert rule '%s' on metric '%s' %s %s",
                     rule.name, rule.metric, rule.condition, rule.threshold)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule by name. Returns ``True`` if removed."""
        removed = self._rules.pop(name, None) is not None
        if removed:
            logger.info("Removed alert rule '%s'", name)
        return removed

    # ------------------------------------------------------------------
    # Alert evaluation
    # ------------------------------------------------------------------

    def check_alerts(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate all rules against the current metric values.

        Args:
            metrics: Mapping of metric name → current value.

        Returns:
            List of newly triggered ``Alert`` instances.
        """
        triggered: List[Alert] = []
        for rule in self._rules.values():
            value = metrics.get(rule.metric)
            if value is None:
                continue
            if self._evaluate_condition(value, rule.condition, rule.threshold):
                alert = self.trigger_alert(rule, value)
                if alert is not None:
                    triggered.append(alert)
        return triggered

    def trigger_alert(self, rule: AlertRule, value: float) -> Optional[Alert]:
        """Fire an alert for *rule* if the cooldown period has elapsed.

        Args:
            rule: The alert rule that was violated.
            value: The metric value that caused the violation.

        Returns:
            The new ``Alert`` if created, or ``None`` if still in cooldown.
        """
        now = time.time()
        last = self._last_triggered.get(rule.name, 0.0)
        if now - last < rule.cooldown_seconds:
            logger.debug("Rule '%s' is in cooldown (%.0fs remaining)",
                         rule.name, rule.cooldown_seconds - (now - last))
            return None

        message = rule.message_template.format(
            name=rule.name, metric=rule.metric, value=value, threshold=rule.threshold
        )
        alert = Alert(
            rule_name=rule.name,
            level=rule.level,
            message=message,
            timestamp=now,
            value=value,
        )
        self._alerts.append(alert)
        self._last_triggered[rule.name] = now
        logger.warning("Alert triggered: [%s] %s", rule.level.name, message)

        if self._webhook_url:
            try:
                asyncio.get_event_loop().create_task(
                    self.send_webhook(alert, self._webhook_url)
                )
            except RuntimeError:
                logger.debug("No running event loop; webhook skipped for '%s'", rule.name)

        return alert

    # ------------------------------------------------------------------
    # Alert management
    # ------------------------------------------------------------------

    def acknowledge_alert(self, alert_id: str, by: str = "system") -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: UUID of the alert.
            by: Identifier of the person or system acknowledging the alert.

        Returns:
            ``True`` if the alert was found and acknowledged.
        """
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = by
                logger.info("Alert %s acknowledged by %s", alert_id, by)
                return True
        logger.warning("Alert %s not found for acknowledgment", alert_id)
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Return all unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]

    def get_alert_history(self, period: Optional[float] = None) -> List[Alert]:
        """Return alert history within the given *period* (seconds).

        Args:
            period: Look-back window in seconds. ``None`` returns all history.
        """
        if period is None:
            return list(self._alerts)
        cutoff = time.time() - period
        return [a for a in self._alerts if a.timestamp >= cutoff]

    # ------------------------------------------------------------------
    # Webhook notifications
    # ------------------------------------------------------------------

    async def send_webhook(self, alert: Alert, webhook_url: str) -> bool:
        """Send an alert notification to a webhook URL.

        Uses ``aiohttp`` if available, otherwise falls back to ``urllib``.

        Args:
            alert: The alert to send.
            webhook_url: Destination URL.

        Returns:
            ``True`` if the webhook call succeeded.
        """
        payload = {
            "id": alert.id,
            "rule_name": alert.rule_name,
            "level": alert.level.name,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "value": alert.value,
        }
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    success = resp.status < 400
                    if success:
                        logger.info("Webhook sent for alert %s (status %d)", alert.id, resp.status)
                    else:
                        logger.error("Webhook failed for alert %s (status %d)", alert.id, resp.status)
                    return success
        except ImportError:
            logger.debug("aiohttp not available; using urllib fallback")
        except Exception as exc:
            logger.error("Webhook (aiohttp) failed for alert %s: %s", alert.id, exc)

        # Fallback to urllib
        try:
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                success = resp.status < 400
                logger.info("Webhook (urllib) sent for alert %s (status %d)", alert.id, resp.status)
                return success
        except Exception as exc:
            logger.error("Webhook (urllib) failed for alert %s: %s", alert.id, exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate a comparison condition."""
        op = self._OPERATORS.get(condition)
        if op is None:
            logger.error("Unknown condition operator: %s", condition)
            return False
        return op(value, threshold)
