"""Real-time data streams for Solana market monitoring.

Provides WebSocket-based connections to Solana RPC for account changes,
program log subscriptions, and real-time price polling with a callback
system for downstream consumers.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import aiohttp

logger = logging.getLogger(__name__)

SOLANA_WS_URL = "wss://api.mainnet-beta.solana.com"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"

Callback = Callable[..., Coroutine[Any, Any, None]]

# Reconnection parameters
_BACKOFF_BASE: float = 1.0
_BACKOFF_MAX: float = 60.0
_BACKOFF_FACTOR: float = 2.0


@dataclass
class PriceUpdate:
    """Real-time price update from a single source."""

    timestamp: float
    token: str
    price: float
    source: str
    volume: float


@dataclass
class AccountChange:
    """Notification of an on-chain account state change."""

    slot: int
    pubkey: str
    lamports: int
    data_hash: str


@dataclass
class WhaleMovement:
    """Large-value token transfer detected on-chain."""

    timestamp: float
    wallet: str
    token: str
    amount_usd: float
    direction: str  # "in" or "out"


class RealtimeDataStream:
    """Manage real-time WebSocket subscriptions and price polling.

    Connects to Solana RPC WebSocket for account and program
    subscriptions, polls prices from multiple sources, and detects
    whale movements.

    Usage::

        stream = RealtimeDataStream()
        await stream.connect()
        await stream.subscribe_account(pubkey, my_callback)
        # ... later ...
        await stream.disconnect()
    """

    def __init__(
        self,
        ws_url: str = SOLANA_WS_URL,
        rpc_url: str = SOLANA_RPC_URL,
    ) -> None:
        self._ws_url = ws_url
        self._rpc_url = rpc_url

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._connected: bool = False
        self._running: bool = False

        # Subscription bookkeeping
        self._next_id: int = 1
        self._subscriptions: dict[int, Callback] = {}  # subscription_id -> callback
        self._pending_subs: dict[int, Callback] = {}  # rpc_request_id -> callback
        self._account_callbacks: dict[str, Callback] = {}

        # Background tasks
        self._listener_task: asyncio.Task[None] | None = None
        self._polling_tasks: list[asyncio.Task[None]] = []
        self._whale_task: asyncio.Task[None] | None = None

        # Reconnection state
        self._backoff: float = _BACKOFF_BASE

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish a WebSocket connection to the Solana RPC node.

        Automatically creates an :class:`aiohttp.ClientSession` if one
        does not already exist and starts the background listener.
        """
        if self._connected:
            logger.warning("Already connected; skipping reconnect")
            return

        self._session = aiohttp.ClientSession()
        await self._connect_ws()
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Connected to Solana WebSocket at %s", self._ws_url)

    async def _connect_ws(self) -> None:
        """Open (or re-open) the raw WebSocket connection."""
        session = self._session
        if session is None:
            raise RuntimeError("No HTTP session available")
        self._ws = await session.ws_connect(
            self._ws_url, heartbeat=30.0, timeout=30.0,
        )
        self._connected = True
        self._backoff = _BACKOFF_BASE

    async def _reconnect(self) -> None:
        """Attempt reconnection with exponential back-off."""
        self._connected = False
        while self._running:
            wait = min(self._backoff, _BACKOFF_MAX)
            logger.info("Reconnecting in %.1fs …", wait)
            await asyncio.sleep(wait)
            self._backoff = min(self._backoff * _BACKOFF_FACTOR, _BACKOFF_MAX)
            try:
                await self._connect_ws()
                # Re-subscribe existing accounts
                await self._resubscribe_all()
                logger.info("Reconnected successfully")
                return
            except Exception:
                logger.warning("Reconnection attempt failed", exc_info=True)

    async def _resubscribe_all(self) -> None:
        """Re-issue all active subscriptions after a reconnect."""
        for pubkey, cb in list(self._account_callbacks.items()):
            await self._send_subscribe("accountSubscribe", [pubkey, {"encoding": "jsonParsed"}], cb)

    async def disconnect(self) -> None:
        """Gracefully close all connections and cancel background tasks."""
        self._running = False

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        for task in self._polling_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._polling_tasks.clear()

        if self._whale_task and not self._whale_task.done():
            self._whale_task.cancel()
            try:
                await self._whale_task
            except asyncio.CancelledError:
                pass

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._session and not self._session.closed:
            await self._session.close()

        self._connected = False
        self._subscriptions.clear()
        self._pending_subs.clear()
        self._account_callbacks.clear()
        logger.info("Disconnected from Solana WebSocket")

    # ------------------------------------------------------------------
    # Internal messaging
    # ------------------------------------------------------------------

    async def _send_subscribe(
        self, method: str, params: list[Any], callback: Callback
    ) -> int:
        """Send a subscription request and track the pending callback."""
        if not self._ws or self._ws.closed:
            raise RuntimeError("WebSocket is not connected")

        req_id = self._next_id
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        await self._ws.send_json(payload)
        self._pending_subs[req_id] = callback
        logger.debug("Sent %s (id=%d)", method, req_id)
        return req_id

    async def _listen_loop(self) -> None:
        """Continuously read messages from the WebSocket."""
        while self._running:
            if not self._ws or self._ws.closed:
                await self._reconnect()
                if not self._connected:
                    break
                continue

            try:
                msg = await self._ws.receive(timeout=60.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                logger.warning("WebSocket receive error", exc_info=True)
                await self._reconnect()
                continue

            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                logger.warning("WebSocket closed/error: %s", msg.data)
                await self._reconnect()
                continue

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON from WebSocket")
                continue

            await self._dispatch(data)

    async def _dispatch(self, data: dict[str, Any]) -> None:
        """Route an incoming WebSocket message to the appropriate handler."""
        # Subscription confirmation – map subscription ID to callback
        if "id" in data and "result" in data:
            req_id = data["id"]
            sub_id = data["result"]
            cb = self._pending_subs.pop(req_id, None)
            if cb is not None:
                self._subscriptions[sub_id] = cb
                logger.debug("Subscription %d confirmed (request %d)", sub_id, req_id)
            return

        # Subscription notification
        if data.get("method") == "accountNotification":
            await self._handle_account_notification(data)
        elif data.get("method") == "logsNotification":
            await self._handle_logs_notification(data)
        elif data.get("method") == "slotNotification":
            await self._handle_slot_notification(data)

    async def _handle_account_notification(self, data: dict[str, Any]) -> None:
        """Process an account change notification."""
        params = data.get("params", {})
        sub_id = params.get("subscription")
        result = params.get("result", {})

        context = result.get("context", {})
        value = result.get("value", {})

        raw_data = value.get("data", "")
        data_str = raw_data if isinstance(raw_data, str) else json.dumps(raw_data)

        change = AccountChange(
            slot=context.get("slot", 0),
            pubkey="",  # not provided in notification; caller tracks via subscription
            lamports=value.get("lamports", 0),
            data_hash=hashlib.sha256(data_str.encode()).hexdigest(),
        )

        cb = self._subscriptions.get(sub_id)
        if cb:
            try:
                await cb(change)
            except Exception:
                logger.exception("Account callback error (sub %d)", sub_id)

    async def _handle_logs_notification(self, data: dict[str, Any]) -> None:
        """Process a program logs notification."""
        params = data.get("params", {})
        sub_id = params.get("subscription")
        result = params.get("result", {})

        cb = self._subscriptions.get(sub_id)
        if cb:
            try:
                await cb(result)
            except Exception:
                logger.exception("Logs callback error (sub %d)", sub_id)

    async def _handle_slot_notification(self, data: dict[str, Any]) -> None:
        """Process a slot update notification."""
        params = data.get("params", {})
        sub_id = params.get("subscription")
        result = params.get("result", {})

        cb = self._subscriptions.get(sub_id)
        if cb:
            try:
                await cb(result)
            except Exception:
                logger.exception("Slot callback error (sub %d)", sub_id)

    # ------------------------------------------------------------------
    # Public subscription API
    # ------------------------------------------------------------------

    async def subscribe_account(self, pubkey: str, callback: Callback) -> int:
        """Subscribe to state changes for an on-chain account.

        Args:
            pubkey: Base-58 encoded public key of the account.
            callback: Async callable receiving an :class:`AccountChange`.

        Returns:
            The RPC request ID used for the subscription.
        """
        logger.info("Subscribing to account %s", pubkey)
        self._account_callbacks[pubkey] = callback
        return await self._send_subscribe(
            "accountSubscribe",
            [pubkey, {"encoding": "jsonParsed", "commitment": "confirmed"}],
            callback,
        )

    async def subscribe_program(self, program_id: str, callback: Callback) -> int:
        """Subscribe to log output for a specific program.

        Args:
            program_id: Base-58 program address to monitor.
            callback: Async callable receiving log result dicts.

        Returns:
            The RPC request ID used for the subscription.
        """
        logger.info("Subscribing to program logs for %s", program_id)
        return await self._send_subscribe(
            "logsSubscribe",
            [{"mentions": [program_id]}, {"commitment": "confirmed"}],
            callback,
        )

    async def start_price_polling(
        self,
        tokens: list[str],
        interval: float,
        callback: Callback,
    ) -> None:
        """Begin periodic price polling for a set of tokens.

        Polls CoinGecko ``/simple/price`` for each token at the given
        interval and delivers :class:`PriceUpdate` objects to *callback*.

        Args:
            tokens: CoinGecko token identifiers.
            interval: Seconds between polling cycles.
            callback: Async callable receiving a :class:`PriceUpdate`.
        """
        logger.info("Starting price polling for %s every %.1fs", tokens, interval)
        task = asyncio.create_task(self._poll_prices(tokens, interval, callback))
        self._polling_tasks.append(task)

    async def _poll_prices(
        self, tokens: list[str], interval: float, callback: Callback
    ) -> None:
        """Background loop that polls prices from CoinGecko."""
        session = self._session
        if session is None:
            raise RuntimeError("No HTTP session available")

        coingecko_url = "https://api.coingecko.com/api/v3/simple/price"

        while self._running:
            ids_param = ",".join(tokens)
            params = {
                "ids": ids_param,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            }
            try:
                async with session.get(
                    coingecko_url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                now = time.time()
                for token_id, info in data.items():
                    price = info.get("usd", 0.0)
                    volume = info.get("usd_24h_vol", 0.0)
                    update = PriceUpdate(
                        timestamp=now,
                        token=token_id,
                        price=float(price),
                        source="coingecko",
                        volume=float(volume),
                    )
                    try:
                        await callback(update)
                    except Exception:
                        logger.exception("Price callback error for %s", token_id)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Price polling cycle failed", exc_info=True)

            await asyncio.sleep(interval)

    async def monitor_whale_movements(
        self,
        threshold_usd: float,
        callback: Callback,
    ) -> None:
        """Monitor for large-value token transfers.

        Subscribes to slot notifications and inspects recent blocks for
        transfers exceeding *threshold_usd*.

        Args:
            threshold_usd: Minimum USD value to qualify as a whale movement.
            callback: Async callable receiving a :class:`WhaleMovement`.
        """
        logger.info("Starting whale monitoring (threshold=$%.0f)", threshold_usd)
        self._whale_task = asyncio.create_task(
            self._whale_loop(threshold_usd, callback)
        )

    async def _whale_loop(self, threshold_usd: float, callback: Callback) -> None:
        """Background loop that checks recent blocks for large transfers."""
        session = self._session
        if session is None:
            raise RuntimeError("No HTTP session available")

        last_slot: int = 0
        sol_price_usd: float = 0.0

        while self._running:
            try:
                # Get the current slot
                payload = {"jsonrpc": "2.0", "id": 1, "method": "getSlot", "params": []}
                async with session.post(
                    self._rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    current_slot = data.get("result", 0)

                if current_slot <= last_slot:
                    await asyncio.sleep(2.0)
                    continue

                # Refresh SOL price periodically from CoinGecko
                try:
                    async with session.get(
                        "https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": "solana", "vs_currencies": "usd"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as price_resp:
                        price_resp.raise_for_status()
                        price_data = await price_resp.json()
                        sol_price_usd = float(price_data.get("solana", {}).get("usd", sol_price_usd))
                except Exception:
                    logger.debug("Could not refresh SOL price; using cached value %.2f", sol_price_usd)

                if sol_price_usd <= 0:
                    await asyncio.sleep(2.0)
                    continue

                # Fetch the latest confirmed block
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBlock",
                    "params": [
                        current_slot,
                        {
                            "encoding": "jsonParsed",
                            "transactionDetails": "full",
                            "maxSupportedTransactionVersion": 0,
                        },
                    ],
                }
                async with session.post(
                    self._rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    resp.raise_for_status()
                    block_data = await resp.json()

                block = block_data.get("result")
                if not block:
                    last_slot = current_slot
                    await asyncio.sleep(2.0)
                    continue

                block_time = block.get("blockTime", time.time())
                transactions = block.get("transactions", [])

                for tx_envelope in transactions:
                    tx = tx_envelope.get("transaction", {})
                    meta = tx_envelope.get("meta", {})
                    if meta.get("err") is not None:
                        continue

                    pre_balances = meta.get("preBalances", [])
                    post_balances = meta.get("postBalances", [])
                    account_keys = tx.get("message", {}).get("accountKeys", [])

                    for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                        diff_lamports = post - pre
                        diff_sol = abs(diff_lamports) / 1e9
                        diff_usd = diff_sol * sol_price_usd

                        if diff_usd >= threshold_usd:
                            key = account_keys[i] if i < len(account_keys) else {}
                            wallet = key if isinstance(key, str) else key.get("pubkey", "unknown")
                            movement = WhaleMovement(
                                timestamp=float(block_time) if block_time else time.time(),
                                wallet=wallet,
                                token="SOL",
                                amount_usd=diff_usd,
                                direction="in" if diff_lamports > 0 else "out",
                            )
                            try:
                                await callback(movement)
                            except Exception:
                                logger.exception("Whale callback error")

                last_slot = current_slot

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Whale monitoring cycle failed", exc_info=True)

            await asyncio.sleep(2.0)
