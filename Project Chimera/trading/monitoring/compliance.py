"""Compliance and audit logging for trading bot.

Provides trade reporting for tax purposes (CSV export), a complete audit trail
for all trading decisions with reasoning chains, and an emergency shutdown
mechanism with state preservation.
"""

import csv
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single audit-trail entry."""
    timestamp: float = field(default_factory=time.time)
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    decision_reasoning: str = ""
    model_scores: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None


@dataclass
class TaxReport:
    """Aggregated tax report for a fiscal period."""
    period: str = ""
    trades: List[Dict[str, Any]] = field(default_factory=list)
    total_pnl: float = 0.0
    short_term_gains: float = 0.0
    long_term_gains: float = 0.0
    fees_paid: float = 0.0


class ComplianceLogger:
    """Compliance & audit-logging engine.

    Maintains an in-memory audit trail, supports CSV export for tax reporting,
    and provides an emergency shutdown mechanism that preserves state to disk.

    Args:
        state_dir: Directory used for emergency state dumps.  Created on demand.
    """

    # Holding period threshold (seconds) for short/long-term gains (1 year).
    _LONG_TERM_THRESHOLD_SECONDS: float = 365.25 * 24 * 3600

    def __init__(self, state_dir: str = "compliance_state") -> None:
        self._audit_trail: List[AuditEntry] = []
        self._trade_log: List[Dict[str, Any]] = []
        self._state_dir = state_dir
        self._is_shutdown = False
        logger.info("ComplianceLogger initialised (state_dir=%s)", state_dir)

    # ------------------------------------------------------------------
    # Decision / trade logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        action: str,
        reasoning: str,
        scores: Optional[Dict[str, float]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Log a trading decision with its reasoning chain.

        Args:
            action: Short action identifier (e.g. ``BUY``, ``SELL``, ``HOLD``).
            reasoning: Human-readable explanation of why the decision was made.
            scores: Optional model confidence scores.
            risk_assessment: Optional risk-analysis data.

        Returns:
            The created ``AuditEntry``.
        """
        entry = AuditEntry(
            action=action,
            decision_reasoning=reasoning,
            model_scores=scores or {},
            risk_assessment=risk_assessment or {},
        )
        self._audit_trail.append(entry)
        logger.info("Decision logged: %s – %s", action, reasoning[:80])
        return entry

    def log_trade(
        self,
        trade: Dict[str, Any],
        reasoning: str = "",
    ) -> AuditEntry:
        """Log an executed trade for audit and tax purposes.

        Args:
            trade: Dictionary with trade details (must include at least
                ``symbol``, ``side``, ``quantity``, ``price``).
            reasoning: Optional reasoning string.

        Returns:
            The created ``AuditEntry``.
        """
        trade_record = {
            "timestamp": time.time(),
            "symbol": trade.get("symbol", "UNKNOWN"),
            "side": trade.get("side", "UNKNOWN"),
            "quantity": float(trade.get("quantity", 0)),
            "price": float(trade.get("price", 0)),
            "fees": float(trade.get("fees", 0)),
            "pnl": float(trade.get("pnl", 0)),
            "holding_period_seconds": float(trade.get("holding_period_seconds", 0)),
        }
        self._trade_log.append(trade_record)

        entry = AuditEntry(
            action="TRADE_EXECUTED",
            details=trade_record,
            decision_reasoning=reasoning,
        )
        self._audit_trail.append(entry)
        logger.info(
            "Trade logged: %s %s %.4f @ %.6f (pnl=%.4f)",
            trade_record["side"], trade_record["symbol"],
            trade_record["quantity"], trade_record["price"],
            trade_record["pnl"],
        )
        return entry

    # ------------------------------------------------------------------
    # Tax reporting
    # ------------------------------------------------------------------

    def export_tax_report(self, year: int, fmt: str = "csv") -> str:
        """Generate a tax report for *year*.

        Args:
            year: The fiscal year to report on.
            fmt: Export format (``csv`` or ``json``).

        Returns:
            Report content as a string.
        """
        year_trades = self._filter_trades_by_year(year)
        total_pnl = sum(t["pnl"] for t in year_trades)
        fees_paid = sum(t["fees"] for t in year_trades)

        short_term = sum(
            t["pnl"] for t in year_trades
            if t["holding_period_seconds"] < self._LONG_TERM_THRESHOLD_SECONDS
        )
        long_term = sum(
            t["pnl"] for t in year_trades
            if t["holding_period_seconds"] >= self._LONG_TERM_THRESHOLD_SECONDS
        )

        report = TaxReport(
            period=str(year),
            trades=year_trades,
            total_pnl=total_pnl,
            short_term_gains=short_term,
            long_term_gains=long_term,
            fees_paid=fees_paid,
        )
        logger.info(
            "Tax report for %d: %d trades, PnL=%.4f, fees=%.4f",
            year, len(year_trades), total_pnl, fees_paid,
        )

        if fmt == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        return self._tax_report_to_csv(report)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_trail(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> List[AuditEntry]:
        """Return audit entries in the ``[start, end]`` time range.

        Args:
            start: Unix timestamp for range start (inclusive).
            end: Unix timestamp for range end (inclusive).

        Returns:
            Filtered list of ``AuditEntry`` instances.
        """
        entries = self._audit_trail
        if start is not None:
            entries = [e for e in entries if e.timestamp >= start]
        if end is not None:
            entries = [e for e in entries if e.timestamp <= end]
        return entries

    # ------------------------------------------------------------------
    # Compliance verification
    # ------------------------------------------------------------------

    def verify_compliance(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic compliance checks against a proposed trade.

        Args:
            trade: Trade details dictionary.

        Returns:
            Dictionary with ``compliant`` boolean and a list of ``violations``.
        """
        violations: List[str] = []
        quantity = float(trade.get("quantity", 0))
        price = float(trade.get("price", 0))

        if quantity <= 0:
            violations.append("Trade quantity must be positive")
        if price <= 0:
            violations.append("Trade price must be positive")

        max_position = float(trade.get("max_position_size", float("inf")))
        if quantity > max_position:
            violations.append(
                f"Position size {quantity} exceeds maximum {max_position}"
            )

        max_notional = float(trade.get("max_notional_value", float("inf")))
        notional = quantity * price
        if notional > max_notional:
            violations.append(
                f"Notional value {notional:.2f} exceeds maximum {max_notional:.2f}"
            )

        if self._is_shutdown:
            violations.append("System is in emergency shutdown state")

        result = {"compliant": len(violations) == 0, "violations": violations}
        if violations:
            logger.warning("Compliance check failed: %s", violations)
        return result

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a summary compliance report.

        Returns:
            Dictionary with trade counts, PnL breakdown, and audit statistics.
        """
        total_trades = len(self._trade_log)
        total_pnl = sum(t["pnl"] for t in self._trade_log)
        total_fees = sum(t["fees"] for t in self._trade_log)
        decisions = len(self._audit_trail)

        return {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_audit_entries": decisions,
            "system_status": "shutdown" if self._is_shutdown else "active",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Emergency shutdown
    # ------------------------------------------------------------------

    def emergency_shutdown(
        self,
        reason: str,
        preserve_state: bool = True,
    ) -> Dict[str, Any]:
        """Initiate an emergency shutdown and optionally preserve state to disk.

        Args:
            reason: Human-readable reason for the shutdown.
            preserve_state: Whether to dump state to ``state_dir``.

        Returns:
            Dictionary describing the shutdown result.
        """
        self._is_shutdown = True
        logger.critical("EMERGENCY SHUTDOWN initiated: %s", reason)

        entry = AuditEntry(
            action="EMERGENCY_SHUTDOWN",
            details={"reason": reason, "preserve_state": preserve_state},
            decision_reasoning=reason,
        )
        self._audit_trail.append(entry)

        state_file: Optional[str] = None
        if preserve_state:
            state_file = self._dump_state(reason)

        return {
            "shutdown": True,
            "reason": reason,
            "timestamp": time.time(),
            "state_file": state_file,
            "open_trades": len(self._trade_log),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_trades_by_year(self, year: int) -> List[Dict[str, Any]]:
        """Return trades whose timestamp falls within *year*."""
        results: List[Dict[str, Any]] = []
        for t in self._trade_log:
            trade_dt = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc)
            if trade_dt.year == year:
                results.append(t)
        return results

    @staticmethod
    def _tax_report_to_csv(report: TaxReport) -> str:
        """Serialise a ``TaxReport`` to CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Period", report.period])
        writer.writerow(["Total PnL", f"{report.total_pnl:.4f}"])
        writer.writerow(["Short-Term Gains", f"{report.short_term_gains:.4f}"])
        writer.writerow(["Long-Term Gains", f"{report.long_term_gains:.4f}"])
        writer.writerow(["Fees Paid", f"{report.fees_paid:.4f}"])
        writer.writerow([])
        writer.writerow(["Timestamp", "Symbol", "Side", "Quantity", "Price", "Fees", "PnL", "Holding Period (s)"])
        for trade in report.trades:
            writer.writerow([
                datetime.fromtimestamp(trade["timestamp"], tz=timezone.utc).isoformat(),
                trade["symbol"],
                trade["side"],
                f"{trade['quantity']:.6f}",
                f"{trade['price']:.8f}",
                f"{trade['fees']:.6f}",
                f"{trade['pnl']:.6f}",
                f"{trade['holding_period_seconds']:.0f}",
            ])
        return buf.getvalue()

    def _dump_state(self, reason: str) -> Optional[str]:
        """Persist current state to a JSON file on disk."""
        try:
            os.makedirs(self._state_dir, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self._state_dir, f"emergency_state_{ts}.json")
            state = {
                "reason": reason,
                "timestamp": time.time(),
                "trade_log": self._trade_log,
                "audit_trail_count": len(self._audit_trail),
                "audit_trail": [asdict(e) for e in self._audit_trail[-100:]],
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, default=str)
            logger.info("Emergency state saved to %s", filepath)
            return filepath
        except Exception as exc:
            logger.error("Failed to save emergency state: %s", exc)
            return None
