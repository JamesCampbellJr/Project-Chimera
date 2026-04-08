"""Historical data pipeline for collecting and processing Solana market data.

Collects price history, volume data, liquidity info, mint/burn events,
and DEX trades from multiple sources including CoinGecko, Jupiter,
and Solana RPC.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
JUPITER_BASE_URL = "https://quote-api.jup.ag/v6"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"

RATE_LIMIT_INTERVAL = 0.5


@dataclass
class PriceCandle:
    """OHLCV price candle for a single time period."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class LiquiditySnapshot:
    """Point-in-time snapshot of pool liquidity metrics."""

    timestamp: float
    token: str
    pool: str
    tvl: float
    volume_24h: float


@dataclass
class MintBurnEvent:
    """Record of a token mint or burn event on-chain."""

    timestamp: float
    token: str
    event_type: str  # "mint" or "burn"
    amount: float
    tx_signature: str


class HistoricalDataPipeline:
    """Async pipeline for collecting historical Solana market data.

    Aggregates data from CoinGecko (prices), Jupiter (volume/liquidity),
    and Solana RPC (on-chain events) with built-in rate limiting and
    error handling.

    Usage::

        async with HistoricalDataPipeline() as pipeline:
            candles = await pipeline.fetch_price_history("solana", 30)
    """

    def __init__(
        self,
        coingecko_url: str = COINGECKO_BASE_URL,
        jupiter_url: str = JUPITER_BASE_URL,
        solana_rpc_url: str = SOLANA_RPC_URL,
        rate_limit: float = RATE_LIMIT_INTERVAL,
    ) -> None:
        self._coingecko_url = coingecko_url
        self._jupiter_url = jupiter_url
        self._solana_rpc_url = solana_rpc_url
        self._rate_limit = rate_limit
        self._last_request_time: float = 0.0
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "HistoricalDataPipeline":
        """Create an aiohttp session for the pipeline."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the aiohttp session."""
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Return the active session, creating one if necessary."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _rate_limit_wait(self) -> None:
        """Enforce minimum interval between outbound requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit:
            await asyncio.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Issue a rate-limited GET request and return the JSON body."""
        await self._rate_limit_wait()
        session = self._get_session()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientResponseError as exc:
            logger.error("HTTP %s from %s: %s", exc.status, url, exc.message)
            raise
        except aiohttp.ClientError as exc:
            logger.error("Request to %s failed: %s", url, exc)
            raise

    async def _rpc_call(self, method: str, params: list[Any] | None = None) -> Any:
        """Send a JSON-RPC 2.0 request to the Solana RPC endpoint."""
        await self._rate_limit_wait()
        session = self._get_session()
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }
        try:
            async with session.post(
                self._solana_rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if "error" in data:
                    logger.error("RPC error for %s: %s", method, data["error"])
                    raise RuntimeError(f"Solana RPC error: {data['error']}")
                return data.get("result")
        except aiohttp.ClientError as exc:
            logger.error("Solana RPC request %s failed: %s", method, exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_price_history(
        self, token_id: str, days: int = 30
    ) -> list[PriceCandle]:
        """Fetch OHLCV price history from CoinGecko.

        Args:
            token_id: CoinGecko token identifier (e.g. ``"solana"``).
            days: Number of historical days to retrieve.

        Returns:
            Chronologically-ordered list of :class:`PriceCandle` objects.
        """
        logger.info("Fetching %d-day price history for %s", days, token_id)

        url = f"{self._coingecko_url}/coins/{token_id}/ohlc"
        params: dict[str, Any] = {"vs_currency": "usd", "days": str(days)}

        try:
            data = await self._get_json(url, params)
        except Exception:
            logger.exception("Failed to fetch price history for %s", token_id)
            return []

        candles: list[PriceCandle] = []
        for entry in data:
            if len(entry) < 5:
                continue
            candles.append(
                PriceCandle(
                    timestamp=entry[0] / 1000.0,
                    open=float(entry[1]),
                    high=float(entry[2]),
                    low=float(entry[3]),
                    close=float(entry[4]),
                    volume=0.0,
                )
            )

        # Augment candles with volume from the market_chart endpoint
        vol_url = f"{self._coingecko_url}/coins/{token_id}/market_chart"
        try:
            vol_data = await self._get_json(vol_url, params)
            volume_map: dict[int, float] = {}
            for ts_ms, vol in vol_data.get("total_volumes", []):
                bucket = int(ts_ms) // 3_600_000
                volume_map[bucket] = float(vol)

            for candle in candles:
                bucket = int(candle.timestamp * 1000) // 3_600_000
                candle.volume = volume_map.get(bucket, 0.0)
        except Exception:
            logger.warning("Could not augment volume data for %s", token_id)

        logger.info("Retrieved %d candles for %s", len(candles), token_id)
        return candles

    async def fetch_volume_data(self, token_address: str) -> dict[str, Any]:
        """Fetch trading volume data from Jupiter.

        Args:
            token_address: Solana token mint address.

        Returns:
            Dictionary containing volume metrics keyed by DEX.
        """
        logger.info("Fetching volume data for token %s", token_address)

        url = f"{self._jupiter_url}/quote"
        # Request a nominal quote to probe available routes and volume
        params = {
            "inputMint": token_address,
            "outputMint": "So11111111111111111111111111111111111111112",  # wSOL
            "amount": "1000000",
            "slippageBps": "50",
        }

        try:
            data = await self._get_json(url, params)
        except Exception:
            logger.exception("Failed to fetch volume data for %s", token_address)
            return {}

        routes: list[dict[str, Any]] = []
        route_plan = data.get("routePlan", [])
        for step in route_plan:
            swap_info = step.get("swapInfo", {})
            routes.append(
                {
                    "dex": swap_info.get("label", "unknown"),
                    "in_amount": int(swap_info.get("inAmount", 0)),
                    "out_amount": int(swap_info.get("outAmount", 0)),
                    "fee_amount": int(swap_info.get("feeAmount", 0)),
                    "fee_mint": swap_info.get("feeMint", ""),
                }
            )

        return {
            "token": token_address,
            "in_amount": int(data.get("inAmount", 0)),
            "out_amount": int(data.get("outAmount", 0)),
            "price_impact_pct": float(data.get("priceImpactPct", 0)),
            "routes": routes,
        }

    async def fetch_liquidity_data(self, pool_address: str) -> LiquiditySnapshot | None:
        """Fetch liquidity data for a specific pool via Solana RPC.

        Args:
            pool_address: On-chain address of the liquidity pool.

        Returns:
            A :class:`LiquiditySnapshot` or ``None`` if unavailable.
        """
        logger.info("Fetching liquidity data for pool %s", pool_address)

        try:
            account_info = await self._rpc_call(
                "getAccountInfo",
                [pool_address, {"encoding": "jsonParsed"}],
            )
        except Exception:
            logger.exception("Failed to fetch liquidity for pool %s", pool_address)
            return None

        if not account_info or not account_info.get("value"):
            logger.warning("Pool account %s not found", pool_address)
            return None

        value = account_info["value"]
        lamports = value.get("lamports", 0)
        owner = value.get("owner", "")

        # Derive TVL from lamports (SOL value); real implementation would
        # parse the pool's specific program data.
        sol_tvl = lamports / 1e9

        return LiquiditySnapshot(
            timestamp=time.time(),
            token=owner,
            pool=pool_address,
            tvl=sol_tvl,
            volume_24h=0.0,
        )

    async def monitor_mint_burn(
        self, token_address: str, limit: int = 50
    ) -> list[MintBurnEvent]:
        """Retrieve recent mint and burn events for a token.

        Queries Solana RPC for confirmed signatures on the token mint
        address and inspects each transaction for supply-changing
        instructions (MintTo / Burn).

        Args:
            token_address: SPL token mint address.
            limit: Maximum number of signatures to inspect.

        Returns:
            List of :class:`MintBurnEvent` objects.
        """
        logger.info(
            "Monitoring mint/burn events for %s (limit=%d)", token_address, limit
        )

        try:
            signatures = await self._rpc_call(
                "getSignaturesForAddress",
                [token_address, {"limit": limit}],
            )
        except Exception:
            logger.exception("Failed to fetch signatures for %s", token_address)
            return []

        if not signatures:
            return []

        events: list[MintBurnEvent] = []
        for sig_info in signatures:
            sig = sig_info.get("signature", "")
            block_time = sig_info.get("blockTime")
            if sig_info.get("err") is not None:
                continue

            try:
                tx = await self._rpc_call(
                    "getTransaction",
                    [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                )
            except Exception:
                logger.warning("Could not fetch tx %s", sig)
                continue

            if not tx:
                continue

            instructions = (
                tx.get("transaction", {})
                .get("message", {})
                .get("instructions", [])
            )
            inner = tx.get("meta", {}).get("innerInstructions", []) or []
            all_ixs = list(instructions)
            for inner_group in inner:
                all_ixs.extend(inner_group.get("instructions", []))

            for ix in all_ixs:
                parsed = ix.get("parsed")
                if not isinstance(parsed, dict):
                    continue
                ix_type = parsed.get("type", "")
                info = parsed.get("info", {})

                if ix_type in ("mintTo", "mintToChecked"):
                    amount_raw = info.get("amount") or info.get("tokenAmount", {}).get("amount", "0")
                    events.append(
                        MintBurnEvent(
                            timestamp=float(block_time) if block_time else 0.0,
                            token=info.get("mint", token_address),
                            event_type="mint",
                            amount=float(amount_raw),
                            tx_signature=sig,
                        )
                    )
                elif ix_type in ("burn", "burnChecked"):
                    amount_raw = info.get("amount") or info.get("tokenAmount", {}).get("amount", "0")
                    events.append(
                        MintBurnEvent(
                            timestamp=float(block_time) if block_time else 0.0,
                            token=info.get("mint", token_address),
                            event_type="burn",
                            amount=float(amount_raw),
                            tx_signature=sig,
                        )
                    )

        logger.info("Found %d mint/burn events for %s", len(events), token_address)
        return events

    async def get_dex_trades(
        self, token_address: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Fetch recent DEX trades involving a token.

        Queries the Solana RPC for recent transactions on the token
        address and extracts swap-related instructions from Raydium,
        Orca, and Jupiter programs.

        Args:
            token_address: SPL token mint address.
            limit: Maximum number of signatures to scan.

        Returns:
            List of dicts with trade details (dex, amounts, signature, timestamp).
        """
        logger.info("Fetching DEX trades for %s (limit=%d)", token_address, limit)

        KNOWN_DEX_PROGRAMS: dict[str, str] = {
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium",
            "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "orca",
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "jupiter",
            "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "jupiter_v4",
        }

        try:
            signatures = await self._rpc_call(
                "getSignaturesForAddress",
                [token_address, {"limit": limit}],
            )
        except Exception:
            logger.exception("Failed to fetch signatures for %s", token_address)
            return []

        if not signatures:
            return []

        trades: list[dict[str, Any]] = []
        for sig_info in signatures:
            sig = sig_info.get("signature", "")
            block_time = sig_info.get("blockTime")
            if sig_info.get("err") is not None:
                continue

            try:
                tx = await self._rpc_call(
                    "getTransaction",
                    [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                )
            except Exception:
                logger.warning("Could not fetch tx %s", sig)
                continue

            if not tx:
                continue

            account_keys = (
                tx.get("transaction", {})
                .get("message", {})
                .get("accountKeys", [])
            )
            program_ids = set()
            for key_entry in account_keys:
                pubkey = key_entry if isinstance(key_entry, str) else key_entry.get("pubkey", "")
                if pubkey in KNOWN_DEX_PROGRAMS:
                    program_ids.add(pubkey)

            if not program_ids:
                continue

            pre_balances = tx.get("meta", {}).get("preTokenBalances", []) or []
            post_balances = tx.get("meta", {}).get("postTokenBalances", []) or []

            balance_changes: list[dict[str, Any]] = []
            for pre, post in zip(pre_balances, post_balances):
                pre_amount = float(
                    pre.get("uiTokenAmount", {}).get("uiAmount") or 0
                )
                post_amount = float(
                    post.get("uiTokenAmount", {}).get("uiAmount") or 0
                )
                if pre_amount != post_amount:
                    balance_changes.append(
                        {
                            "mint": post.get("mint", ""),
                            "change": post_amount - pre_amount,
                            "owner": post.get("owner", ""),
                        }
                    )

            for pid in program_ids:
                trades.append(
                    {
                        "dex": KNOWN_DEX_PROGRAMS[pid],
                        "signature": sig,
                        "timestamp": float(block_time) if block_time else 0.0,
                        "balance_changes": balance_changes,
                        "token": token_address,
                    }
                )

        logger.info("Found %d DEX trades for %s", len(trades), token_address)
        return trades
