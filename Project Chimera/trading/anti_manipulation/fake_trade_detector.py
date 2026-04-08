"""
Fake-trade and honeypot detection for Solana tokens.

Analyses on-chain data (sell/buy ratios, mint authority, holder
concentration) to identify tokens designed to trap buyers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from trading.solana_client import SolanaClient

logger = logging.getLogger(__name__)


# ====================================================================
# Data classes
# ====================================================================

@dataclass
class TokenRiskAssessment:
    """Comprehensive risk assessment for a single token."""

    token: str
    honeypot_risk: float
    mint_risk: float
    concentration_risk: float
    overall_risk: float
    details: dict = field(default_factory=dict)


# ====================================================================
# FakeTradeDetector
# ====================================================================

class FakeTradeDetector:
    """
    Detects honeypot tokens, dangerous mint/freeze authorities,
    and high holder-concentration risk on Solana.
    """

    def __init__(self, solana_client: Optional[SolanaClient] = None) -> None:
        """
        Parameters
        ----------
        solana_client : SolanaClient, optional
            An initialised :class:`SolanaClient`.  If ``None`` a fresh
            instance is created from the default config.
        """
        self.client = solana_client or SolanaClient()

    # ------------------------------------------------------------------ #
    # Honeypot detection                                                  #
    # ------------------------------------------------------------------ #

    async def detect_honeypot(self, token: str) -> dict:
        """
        Detect honeypot characteristics by analysing sell-to-buy ratios
        and checking for failed sell transactions.

        Parameters
        ----------
        token : str
            SPL token mint address.

        Returns
        -------
        dict
            Keys: ``is_honeypot``, ``sell_buy_ratio``, ``failed_sells``,
            ``confidence``.
        """
        result: dict[str, Any] = {
            "is_honeypot": False,
            "sell_buy_ratio": 0.0,
            "failed_sells": 0,
            "confidence": 0.0,
        }

        try:
            signatures = self.client.get_transaction_signatures(token, limit=100)
            if not signatures:
                logger.debug("No signatures found for token %s", token)
                return result

            buys = 0
            sells = 0
            failed_sells = 0

            for sig_info in signatures:
                sig = sig_info.get("signature")
                if not sig:
                    continue

                tx = self.client.get_transaction(sig)
                if tx is None:
                    continue

                meta = tx.get("meta", {})
                err = meta.get("err") if meta else None

                swap = self.client.parse_swap_from_transaction(tx, token)
                if swap is None:
                    # Try to infer direction from token balance changes
                    pre_balances = meta.get("preTokenBalances") or [] if meta else []
                    post_balances = meta.get("postTokenBalances") or [] if meta else []

                    for pre, post in _zip_token_balances(pre_balances, post_balances):
                        if pre.get("mint") != token and post.get("mint") != token:
                            continue
                        pre_amt = float(
                            (pre.get("uiTokenAmount") or {}).get("uiAmount") or 0
                        )
                        post_amt = float(
                            (post.get("uiTokenAmount") or {}).get("uiAmount") or 0
                        )
                        delta = post_amt - pre_amt
                        if delta > 0:
                            buys += 1
                        elif delta < 0:
                            sells += 1
                            if err is not None:
                                failed_sells += 1
                    continue

                # swap successfully parsed
                if swap.get("token_out") == token:
                    buys += 1
                elif swap.get("token_in") == token:
                    sells += 1
                    if err is not None:
                        failed_sells += 1

            total = buys + sells
            if total == 0:
                return result

            sell_buy_ratio = sells / max(buys, 1)
            result["sell_buy_ratio"] = round(sell_buy_ratio, 4)
            result["failed_sells"] = failed_sells

            # Honeypot heuristic: buys vastly outnumber sells (ratio < 0.2 → 5:1)
            is_honeypot = False
            confidence = 0.0

            if buys > 0 and sell_buy_ratio < 0.2:
                is_honeypot = True
                confidence = min(1.0, (0.2 - sell_buy_ratio) / 0.2 * 0.7)

            if failed_sells > 0:
                fail_rate = failed_sells / max(sells + failed_sells, 1)
                is_honeypot = is_honeypot or fail_rate > 0.5
                confidence = max(confidence, min(1.0, fail_rate))

            result["is_honeypot"] = is_honeypot
            result["confidence"] = round(confidence, 4)

        except Exception as exc:  # noqa: BLE001
            logger.error("Honeypot detection failed for %s: %s", token, exc)

        return result

    # ------------------------------------------------------------------ #
    # Mint / freeze authority                                             #
    # ------------------------------------------------------------------ #

    async def check_mint_authority(self, token: str) -> dict:
        """
        Check whether the token's mint or freeze authority is still active.

        Parameters
        ----------
        token : str
            SPL token mint address.

        Returns
        -------
        dict
            Keys: ``mint_authority``, ``freeze_authority``,
            ``is_renounced``, ``risk_level``.
        """
        result: dict[str, Any] = {
            "mint_authority": None,
            "freeze_authority": None,
            "is_renounced": True,
            "risk_level": "low",
        }

        try:
            account_info = self.client._rpc(
                "getAccountInfo",
                [token, {"encoding": "jsonParsed"}],
            )
            if account_info is None:
                logger.warning("getAccountInfo returned None for %s", token)
                return result

            value = account_info.get("value")
            if value is None:
                return result

            parsed = (
                value.get("data", {})
                .get("parsed", {})
                .get("info", {})
            )

            mint_authority = parsed.get("mintAuthority")
            freeze_authority = parsed.get("freezeAuthority")

            result["mint_authority"] = mint_authority
            result["freeze_authority"] = freeze_authority

            # Determine risk
            if mint_authority is not None:
                result["is_renounced"] = False
                result["risk_level"] = "high"
            elif freeze_authority is not None:
                result["is_renounced"] = False
                result["risk_level"] = "moderate"
            else:
                result["is_renounced"] = True
                result["risk_level"] = "low"

        except Exception as exc:  # noqa: BLE001
            logger.error("Mint authority check failed for %s: %s", token, exc)

        return result

    # ------------------------------------------------------------------ #
    # Ownership concentration  (HHI)                                      #
    # ------------------------------------------------------------------ #

    def calculate_ownership_concentration(self, holders: list[dict]) -> dict:
        """
        Compute the Herfindahl–Hirschman Index (HHI) of token holder
        concentration.

        Parameters
        ----------
        holders : list[dict]
            Each dict must have a ``"balance"`` key (float/int).

        Returns
        -------
        dict
            Keys: ``hhi``, ``top_10_pct``, ``is_concentrated``.
        """
        result: dict[str, Any] = {
            "hhi": 0.0,
            "top_10_pct": 0.0,
            "is_concentrated": False,
        }

        if not holders:
            return result

        try:
            balances = [float(h.get("balance", 0)) for h in holders]
            total_supply = sum(balances)
            if total_supply <= 0:
                return result

            shares = [b / total_supply for b in balances]

            # HHI: sum of squared market shares
            hhi = sum(s * s for s in shares)

            # Top-10 holder percentage
            sorted_shares = sorted(shares, reverse=True)
            top_10_pct = sum(sorted_shares[:10]) * 100.0

            result["hhi"] = round(hhi, 6)
            result["top_10_pct"] = round(top_10_pct, 2)
            result["is_concentrated"] = hhi > 0.25

        except Exception as exc:  # noqa: BLE001
            logger.error("Ownership concentration calculation failed: %s", exc)

        return result

    # ------------------------------------------------------------------ #
    # Rug-pull risk                                                       #
    # ------------------------------------------------------------------ #

    async def assess_rug_pull_risk(self, token: str) -> dict:
        """
        Combine honeypot, mint-authority, and concentration signals to
        produce an overall rug-pull risk score.

        Parameters
        ----------
        token : str
            SPL token mint address.

        Returns
        -------
        dict
            Keys: ``risk_score`` (0–1), ``risk_level``, ``factors``.
        """
        result: dict[str, Any] = {
            "risk_score": 0.0,
            "risk_level": "low",
            "factors": {},
        }

        try:
            honeypot = await self.detect_honeypot(token)
            mint_auth = await self.check_mint_authority(token)

            # Fetch holders for concentration check
            token_accounts = self.client.get_token_accounts(token)
            holders: list[dict] = []
            for acct in token_accounts:
                info = (
                    acct.get("account", {})
                    .get("data", {})
                    .get("parsed", {})
                    .get("info", {})
                )
                balance = float(
                    (info.get("tokenAmount") or {}).get("uiAmount") or 0
                )
                holders.append({"balance": balance})

            concentration = self.calculate_ownership_concentration(holders)

            # Weighted score
            honeypot_score = honeypot.get("confidence", 0.0)
            mint_score = {"low": 0.0, "moderate": 0.4, "high": 0.8}.get(
                mint_auth.get("risk_level", "low"), 0.0
            )
            conc_score = min(concentration.get("hhi", 0.0) / 0.5, 1.0)

            risk_score = (
                0.40 * honeypot_score
                + 0.35 * mint_score
                + 0.25 * conc_score
            )
            risk_score = round(min(risk_score, 1.0), 4)

            if risk_score >= 0.7:
                risk_level = "critical"
            elif risk_score >= 0.5:
                risk_level = "high"
            elif risk_score >= 0.3:
                risk_level = "moderate"
            else:
                risk_level = "low"

            result["risk_score"] = risk_score
            result["risk_level"] = risk_level
            result["factors"] = {
                "honeypot": honeypot,
                "mint_authority": mint_auth,
                "concentration": concentration,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Rug-pull risk assessment failed for %s: %s", token, exc)

        return result

    # ------------------------------------------------------------------ #
    # Comprehensive risk assessment                                       #
    # ------------------------------------------------------------------ #

    async def get_token_risk(self, token: str) -> TokenRiskAssessment:
        """
        Run all detection checks and return a :class:`TokenRiskAssessment`.

        Parameters
        ----------
        token : str
            SPL token mint address.

        Returns
        -------
        TokenRiskAssessment
        """
        try:
            rug = await self.assess_rug_pull_risk(token)
            factors = rug.get("factors", {})

            honeypot_risk = factors.get("honeypot", {}).get("confidence", 0.0)

            mint_info = factors.get("mint_authority", {})
            mint_risk = {"low": 0.0, "moderate": 0.4, "high": 0.8}.get(
                mint_info.get("risk_level", "low"), 0.0
            )

            conc = factors.get("concentration", {})
            concentration_risk = min(conc.get("hhi", 0.0) / 0.5, 1.0)

            return TokenRiskAssessment(
                token=token,
                honeypot_risk=round(honeypot_risk, 4),
                mint_risk=round(mint_risk, 4),
                concentration_risk=round(concentration_risk, 4),
                overall_risk=rug.get("risk_score", 0.0),
                details={
                    "honeypot": factors.get("honeypot", {}),
                    "mint_authority": factors.get("mint_authority", {}),
                    "concentration": factors.get("concentration", {}),
                    "risk_level": rug.get("risk_level", "low"),
                },
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Full token risk assessment failed for %s: %s", token, exc)
            return TokenRiskAssessment(
                token=token,
                honeypot_risk=0.0,
                mint_risk=0.0,
                concentration_risk=0.0,
                overall_risk=0.0,
                details={"error": str(exc)},
            )


# ====================================================================
# Helpers
# ====================================================================

def _zip_token_balances(
    pre: list[dict],
    post: list[dict],
) -> list[tuple[dict, dict]]:
    """
    Pair pre- and post-token-balance entries by ``accountIndex``.

    Returns a list of ``(pre_entry, post_entry)`` tuples.
    Missing entries on either side are represented as empty dicts.
    """
    pre_map = {b.get("accountIndex"): b for b in pre}
    post_map = {b.get("accountIndex"): b for b in post}
    indices = set(pre_map) | set(post_map)
    return [(pre_map.get(i, {}), post_map.get(i, {})) for i in indices]
