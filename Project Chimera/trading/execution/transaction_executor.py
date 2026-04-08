# trading/execution/transaction_executor.py
"""
Transaction Executor — executes trades on Solana with retry logic,
confirmation monitoring, failure handling, batch execution, and
comprehensive audit logging.

All network operations are async.  Failed transactions are tracked for
cost analysis and post-mortem review.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Constants                                                              #
# --------------------------------------------------------------------- #
MAX_RETRIES = 3
INITIAL_BACKOFF_SEC = 1.0
BACKOFF_MULTIPLIER = 2.0
CONFIRMATION_TIMEOUT_SEC = 60
CONFIRMATION_POLL_SEC = 2.0
BATCH_CONCURRENCY = 3

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"


# --------------------------------------------------------------------- #
# Data classes                                                           #
# --------------------------------------------------------------------- #
@dataclass
class ExecutionResult:
    """Outcome of a single trade execution attempt."""

    success: bool
    tx_signature: Optional[str]
    executed_price: float
    slippage_actual: float
    fees_paid: float
    latency_ms: float
    retries: int
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.executed_price = round(self.executed_price, 8)
        self.slippage_actual = round(self.slippage_actual, 6)
        self.fees_paid = round(self.fees_paid, 8)
        self.latency_ms = round(self.latency_ms, 2)


@dataclass
class ExecutionReport:
    """Aggregated execution statistics over a reporting period."""

    period: str
    total_trades: int
    successful: int
    failed: int
    total_fees: float
    avg_slippage: float
    avg_latency: float

    def __post_init__(self) -> None:
        self.total_fees = round(self.total_fees, 6)
        self.avg_slippage = round(self.avg_slippage, 6)
        self.avg_latency = round(self.avg_latency, 2)

    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.successful / self.total_trades


# --------------------------------------------------------------------- #
# Transaction Executor                                                   #
# --------------------------------------------------------------------- #
class TransactionExecutor:
    """
    Executes Solana transactions with automatic retry, confirmation
    polling, failure tracking, and audit logging.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
        initial_backoff: float = INITIAL_BACKOFF_SEC,
        confirmation_timeout: float = CONFIRMATION_TIMEOUT_SEC,
        batch_concurrency: int = BATCH_CONCURRENCY,
        timeout: int = 30,
    ) -> None:
        self.rpc_url = rpc_url or DEFAULT_RPC_URL
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.confirmation_timeout = confirmation_timeout
        self.batch_concurrency = batch_concurrency
        self.timeout = timeout

        # Audit log
        self._execution_log: List[ExecutionResult] = []
        self._failed_orders: List[Dict[str, Any]] = []

        logger.info(
            "TransactionExecutor ready  rpc=%s  retries=%d  "
            "backoff=%.1fs  confirm_timeout=%.0fs",
            self.rpc_url[:40], self.max_retries,
            self.initial_backoff, self.confirmation_timeout,
        )

    # ------------------------------------------------------------------ #
    # RPC helpers                                                          #
    # ------------------------------------------------------------------ #

    async def _rpc_call(
        self,
        method: str,
        params: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Make a JSON-RPC call to the Solana cluster."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.rpc_url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                ),
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.error("RPC call %s failed: %s", method, exc)
            return None

    # ------------------------------------------------------------------ #
    # Core execution                                                       #
    # ------------------------------------------------------------------ #

    async def execute_trade(self, order: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a trade with retry logic and exponential backoff.

        Parameters
        ----------
        order : dict
            Must include:
              - serialized_tx (str): base-58/64 encoded signed transaction
              - expected_price (float): expected fill price
              - expected_output (float): expected output amount
              - input_amount (float): input amount
              - token (str): token identifier for logging

        Returns
        -------
        ExecutionResult
        """
        token = order.get("token", "unknown")
        expected_price = float(order.get("expected_price", 0))
        expected_output = float(order.get("expected_output", 0))
        serialized_tx = order.get("serialized_tx", "")

        start_time = time.monotonic()
        last_error: Optional[str] = None
        tx_signature: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Executing trade for %s  attempt %d/%d",
                token, attempt, self.max_retries,
            )

            try:
                # Submit transaction
                result = await self._rpc_call(
                    "sendTransaction",
                    [
                        serialized_tx,
                        {"encoding": "base64", "skipPreflight": False},
                    ],
                )

                if result is None:
                    last_error = "RPC call returned None"
                    raise ConnectionError(last_error)

                if "error" in result:
                    error_msg = str(result["error"])
                    last_error = error_msg
                    logger.warning(
                        "Transaction error on attempt %d: %s", attempt, error_msg
                    )
                    raise RuntimeError(error_msg)

                tx_signature = result.get("result")
                if not tx_signature:
                    last_error = "No transaction signature in response"
                    raise RuntimeError(last_error)

                logger.info("Transaction submitted: %s", tx_signature)

                # Wait for confirmation
                confirmed = await self.confirm_transaction(
                    tx_signature, self.confirmation_timeout
                )

                if not confirmed:
                    last_error = f"Transaction {tx_signature} not confirmed within timeout"
                    logger.warning(last_error)
                    raise TimeoutError(last_error)

                # Success
                latency = (time.monotonic() - start_time) * 1000
                actual_output = float(order.get("actual_output", expected_output))
                slippage = 0.0
                if expected_output > 0:
                    slippage = (expected_output - actual_output) / expected_output

                fees = float(order.get("fees", 0.000005))  # base tx fee

                exec_result = ExecutionResult(
                    success=True,
                    tx_signature=tx_signature,
                    executed_price=expected_price,
                    slippage_actual=slippage,
                    fees_paid=fees,
                    latency_ms=latency,
                    retries=attempt - 1,
                )
                self.log_execution(exec_result)
                logger.info(
                    "Trade executed: %s  sig=%s  latency=%.0fms  retries=%d",
                    token, tx_signature[:16] if tx_signature else "?",
                    latency, attempt - 1,
                )
                return exec_result

            except (ConnectionError, RuntimeError, TimeoutError) as exc:
                last_error = str(exc)
                if attempt < self.max_retries:
                    backoff = self.initial_backoff * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    logger.info(
                        "Retrying in %.1fs (attempt %d/%d)…",
                        backoff, attempt, self.max_retries,
                    )
                    await asyncio.sleep(backoff)

        # All retries exhausted
        latency = (time.monotonic() - start_time) * 1000
        exec_result = ExecutionResult(
            success=False,
            tx_signature=tx_signature,
            executed_price=0.0,
            slippage_actual=0.0,
            fees_paid=0.0,
            latency_ms=latency,
            retries=self.max_retries,
            error=last_error,
        )
        self.log_execution(exec_result)
        await self.handle_failure(order, last_error or "Unknown error")
        return exec_result

    # ------------------------------------------------------------------ #
    # Confirmation                                                         #
    # ------------------------------------------------------------------ #

    async def confirm_transaction(
        self,
        signature: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Poll the cluster for transaction confirmation.

        Parameters
        ----------
        signature : str
            Transaction signature to monitor.
        timeout : float or None
            Maximum seconds to wait; defaults to instance setting.

        Returns
        -------
        bool
            True if the transaction is confirmed (finalized or confirmed
            commitment), False on timeout.
        """
        timeout = timeout if timeout is not None else self.confirmation_timeout
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            result = await self._rpc_call(
                "getSignatureStatuses",
                [[signature], {"searchTransactionHistory": True}],
            )

            if result and "result" in result:
                value = result["result"].get("value", [None])
                if value and value[0] is not None:
                    status = value[0]
                    confirmation = status.get("confirmationStatus", "")
                    err = status.get("err")

                    if err:
                        logger.warning(
                            "Transaction %s failed on-chain: %s",
                            signature[:16], err,
                        )
                        return False

                    if confirmation in ("confirmed", "finalized"):
                        logger.info(
                            "Transaction %s confirmed (%s).",
                            signature[:16], confirmation,
                        )
                        return True

            await asyncio.sleep(CONFIRMATION_POLL_SEC)

        logger.warning(
            "Transaction %s confirmation timed out after %.0fs.",
            signature[:16], timeout,
        )
        return False

    # ------------------------------------------------------------------ #
    # Failure handling                                                     #
    # ------------------------------------------------------------------ #

    async def handle_failure(
        self,
        order: Dict[str, Any],
        error: str,
    ) -> None:
        """
        Record a failed trade for post-mortem analysis and cost tracking.
        """
        failure_record = {
            "timestamp": time.time(),
            "order": {
                k: v for k, v in order.items()
                if k != "serialized_tx"  # don't log raw tx data
            },
            "error": error,
        }
        self._failed_orders.append(failure_record)

        # Keep bounded
        if len(self._failed_orders) > 500:
            self._failed_orders = self._failed_orders[-250:]

        logger.error(
            "Trade FAILED for %s: %s  (total failures: %d)",
            order.get("token", "unknown"), error, len(self._failed_orders),
        )

    # ------------------------------------------------------------------ #
    # Batch execution                                                      #
    # ------------------------------------------------------------------ #

    async def batch_execute(
        self,
        orders: List[Dict[str, Any]],
    ) -> List[ExecutionResult]:
        """
        Execute multiple orders with bounded concurrency.

        Returns results in the same order as the input orders.
        """
        if not orders:
            return []

        logger.info(
            "Batch executing %d orders (concurrency=%d).",
            len(orders), self.batch_concurrency,
        )

        semaphore = asyncio.Semaphore(self.batch_concurrency)

        async def _guarded(order: Dict[str, Any]) -> ExecutionResult:
            async with semaphore:
                return await self.execute_trade(order)

        results = await asyncio.gather(
            *[_guarded(o) for o in orders],
            return_exceptions=False,
        )

        succeeded = sum(1 for r in results if r.success)
        logger.info(
            "Batch complete: %d/%d succeeded.", succeeded, len(orders),
        )
        return list(results)

    # ------------------------------------------------------------------ #
    # Audit logging                                                        #
    # ------------------------------------------------------------------ #

    def log_execution(self, result: ExecutionResult) -> None:
        """Append an execution result to the audit log."""
        self._execution_log.append(result)

        # Keep bounded
        if len(self._execution_log) > 10_000:
            self._execution_log = self._execution_log[-5_000:]

        level = logging.INFO if result.success else logging.WARNING
        logger.log(
            level,
            "AUDIT  success=%s  sig=%s  price=%.8f  slippage=%.4f%%  "
            "fees=%.8f  latency=%.0fms  retries=%d  error=%s",
            result.success,
            (result.tx_signature or "")[:16],
            result.executed_price,
            result.slippage_actual * 100,
            result.fees_paid,
            result.latency_ms,
            result.retries,
            result.error or "none",
        )

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def get_execution_report(self, period: str = "all") -> ExecutionReport:
        """
        Generate an aggregated execution report.

        Parameters
        ----------
        period : str
            Label for the reporting period (e.g. "24h", "7d", "all").
            Currently returns stats over all logged executions.
        """
        log = self._execution_log
        total = len(log)
        if total == 0:
            return ExecutionReport(
                period=period,
                total_trades=0,
                successful=0,
                failed=0,
                total_fees=0.0,
                avg_slippage=0.0,
                avg_latency=0.0,
            )

        successful = sum(1 for r in log if r.success)
        failed = total - successful
        total_fees = sum(r.fees_paid for r in log)

        successful_results = [r for r in log if r.success]
        avg_slippage = (
            sum(r.slippage_actual for r in successful_results) / len(successful_results)
            if successful_results
            else 0.0
        )
        avg_latency = sum(r.latency_ms for r in log) / total

        report = ExecutionReport(
            period=period,
            total_trades=total,
            successful=successful,
            failed=failed,
            total_fees=total_fees,
            avg_slippage=avg_slippage,
            avg_latency=avg_latency,
        )

        logger.info(
            "Execution report [%s]: %d trades  %d ok  %d fail  "
            "fees=%.6f  slip=%.4f%%  latency=%.0fms",
            period, total, successful, failed,
            total_fees, avg_slippage * 100, avg_latency,
        )
        return report

    @property
    def failed_orders(self) -> List[Dict[str, Any]]:
        """Access the failed orders log for analysis."""
        return list(self._failed_orders)

    @property
    def execution_log(self) -> List[ExecutionResult]:
        """Access the full execution audit log."""
        return list(self._execution_log)
