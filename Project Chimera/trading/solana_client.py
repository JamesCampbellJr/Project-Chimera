# trading/solana_client.py
"""
Light-weight Solana RPC client for the paper-trading bot.

Uses the public JSON-RPC endpoint; no private key / signing required.
All blockchain interaction is *read-only*.
"""

import logging
import time
import random
from typing import Any, Optional

import requests

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Known DEX program IDs (mainnet-beta)                                  #
# --------------------------------------------------------------------- #
DEX_PROGRAMS = {
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium AMM",
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "Orca",
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter v6",
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc":  "Orca Whirlpool",
}

# Wrapped SOL mint address
WSOL = "So11111111111111111111111111111111111111112"

# CoinGecko token IDs keyed by mint address (subset for common tokens)
COINGECKO_IDS = {
    WSOL: "solana",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "usd-coin",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "tether",
}


class SolanaClient:
    """
    Read-only Solana RPC + price-feed client.

    All public methods return plain Python dicts/lists so that the rest of
    the bot does not need to import any Solana SDK.
    """

    def __init__(self, rpc_url: Optional[str] = None):
        self.rpc_url = rpc_url or config.SOLANA_RPC_URL
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._req_id = 0
        logger.info("SolanaClient connected to %s", self.rpc_url)

    # ------------------------------------------------------------------ #
    # Low-level RPC                                                        #
    # ------------------------------------------------------------------ #

    def _rpc(self, method: str, params: list, retries: int = 3) -> Any:
        """Send a JSON-RPC request with simple retry/back-off."""
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        for attempt in range(retries):
            try:
                resp = self._session.post(
                    self.rpc_url,
                    json=payload,
                    timeout=config.SOLANA_RPC_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    logger.warning("RPC error %s: %s", method, data["error"])
                    return None
                return data.get("result")
            except requests.RequestException as exc:
                wait = 2 ** attempt + random.uniform(0, 1)
                logger.warning(
                    "RPC call %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                    method, attempt + 1, retries, exc, wait,
                )
                time.sleep(wait)
        logger.error("All %d retries exhausted for RPC method %s", retries, method)
        return None

    # ------------------------------------------------------------------ #
    # Chain data helpers                                                   #
    # ------------------------------------------------------------------ #

    def get_sol_balance(self, address: str) -> float:
        """Return SOL balance (in SOL, not lamports) for *address*."""
        result = self._rpc("getBalance", [address])
        if result is None:
            return 0.0
        lamports = result.get("value", 0)
        return lamports / 1e9

    def get_transaction_signatures(
        self,
        address: str,
        limit: int = 100,
        before: Optional[str] = None,
    ) -> list[dict]:
        """
        Return up to *limit* transaction signatures for *address*.

        Each item is a dict with at minimum:
          signature, slot, blockTime, err
        """
        params: list = [address, {"limit": limit, "commitment": "finalized"}]
        if before:
            params[1]["before"] = before
        result = self._rpc("getSignaturesForAddress", params)
        return result or []

    def get_transaction(self, signature: str) -> Optional[dict]:
        """Fetch a fully-decoded transaction by its signature."""
        result = self._rpc(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": "finalized",
                },
            ],
        )
        return result

    def get_token_accounts(self, address: str) -> list[dict]:
        """Return all SPL token accounts owned by *address*."""
        result = self._rpc(
            "getTokenAccountsByOwner",
            [
                address,
                {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                {"encoding": "jsonParsed"},
            ],
        )
        if not result:
            return []
        return result.get("value", [])

    # ------------------------------------------------------------------ #
    # Transaction parsing                                                  #
    # ------------------------------------------------------------------ #

    def parse_swap_from_transaction(
        self, tx: dict, wallet_address: str
    ) -> Optional[dict]:
        """
        Attempt to extract swap details (token_in, token_out, amounts) from
        a decoded transaction.

        Returns None if the transaction is not a recognisable swap.
        """
        if not tx:
            return None

        meta = tx.get("meta", {})
        if meta and meta.get("err"):
            return None  # failed transaction

        # Detect which DEX program was involved
        account_keys = []
        tx_msg = tx.get("transaction", {}).get("message", {})
        for acct in tx_msg.get("accountKeys", []):
            if isinstance(acct, dict):
                account_keys.append(acct.get("pubkey", ""))
            else:
                account_keys.append(str(acct))

        dex_name = None
        for prog_id, name in DEX_PROGRAMS.items():
            if prog_id in account_keys:
                dex_name = name
                break

        if dex_name is None:
            return None  # not a known DEX swap

        # Extract pre/post token balances to infer swap direction
        pre_balances = {
            b["accountIndex"]: b
            for b in (meta.get("preTokenBalances") or [])
        }
        post_balances = {
            b["accountIndex"]: b
            for b in (meta.get("postTokenBalances") or [])
        }

        token_in = None
        token_out = None
        amount_in = 0.0
        amount_out = 0.0

        all_indices = set(pre_balances) | set(post_balances)
        for idx in all_indices:
            pre = pre_balances.get(idx, {})
            post = post_balances.get(idx, {})

            # Only consider token accounts owned by the wallet
            owner = post.get("owner") or pre.get("owner")
            if owner != wallet_address:
                continue

            mint = post.get("mint") or pre.get("mint")
            pre_amt = float(
                (pre.get("uiTokenAmount") or {}).get("uiAmount") or 0
            )
            post_amt = float(
                (post.get("uiTokenAmount") or {}).get("uiAmount") or 0
            )
            delta = post_amt - pre_amt

            if delta < 0:
                token_in = mint
                amount_in = abs(delta)
            elif delta > 0:
                token_out = mint
                amount_out = delta

        if not token_in or not token_out:
            return None

        return {
            "signature": tx.get("transaction", {})
            .get("signatures", [None])[0],
            "wallet_address": wallet_address,
            "block_time": tx.get("blockTime"),
            "token_in": token_in,
            "token_out": token_out,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "dex_program": dex_name,
            "price_usd": None,  # filled in by PriceFeed
        }

    # ------------------------------------------------------------------ #
    # Price feed (CoinGecko — free tier)                                  #
    # ------------------------------------------------------------------ #

    def get_token_price_usd(self, mint_address: str) -> Optional[float]:
        """
        Look up the current USD price for a token mint address.

        Falls back to CoinGecko's /simple/price endpoint.
        Returns None if price is unavailable.
        """
        cg_id = COINGECKO_IDS.get(mint_address)
        if not cg_id:
            logger.debug("No CoinGecko ID mapping for mint %s", mint_address)
            return None

        try:
            resp = self._session.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get(cg_id, {}).get("usd")
            return float(price) if price is not None else None
        except Exception as exc:
            logger.warning("CoinGecko price fetch failed for %s: %s", mint_address, exc)
            return None

    def get_sol_price_usd(self) -> float:
        """Return current SOL/USD price (defaults to 0.0 on failure)."""
        price = self.get_token_price_usd(WSOL)
        return price or 0.0
