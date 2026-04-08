# trading/analysis/wallet_clustering.py
"""
Wallet clustering and entity-resolution engine.

Groups related Solana wallets into clusters using a Union-Find data
structure, timing correlation, and common-funding-source analysis.  The
resulting entity resolutions help the trading bot identify coordinated
activity (e.g. sybil farms, whale sub-wallets).
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

# Number of time bins used for transaction-timing correlation.
_DEFAULT_TIME_BINS = 48


# --------------------------------------------------------------------- #
#  Union-Find (Disjoint-Set)                                             #
# --------------------------------------------------------------------- #

class UnionFind:
    """Weighted quick-union with path compression.

    Supports arbitrary hashable elements.  Elements are lazily added on
    first use via :meth:`find`.
    """

    def __init__(self) -> None:
        self._parent: dict = {}
        self._rank: dict = {}

    def find(self, x: object) -> object:
        """Return the canonical root of *x*, creating a new set if needed.

        Applies full path compression so that subsequent lookups are
        nearly O(1).
        """
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x

        # Path compression: point every node directly at the root.
        root = x
        while self._parent[root] != root:
            root = self._parent[root]

        while self._parent[x] != root:
            next_x = self._parent[x]
            self._parent[x] = root
            x = next_x

        return root

    def union(self, x: object, y: object) -> None:
        """Merge the sets containing *x* and *y* (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Attach the smaller-rank tree under the larger-rank tree.
        if self._rank[rx] < self._rank[ry]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[ry]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1

    def connected(self, x: object, y: object) -> bool:
        """Return ``True`` if *x* and *y* belong to the same set."""
        return self.find(x) == self.find(y)

    def get_clusters(self) -> dict:
        """Return ``{root: set(members)}`` for every disjoint set."""
        clusters: dict[object, set] = defaultdict(set)
        for element in self._parent:
            root = self.find(element)
            clusters[root].add(element)
        return dict(clusters)


# --------------------------------------------------------------------- #
#  Data classes                                                          #
# --------------------------------------------------------------------- #

@dataclass
class WalletCluster:
    """A group of wallets believed to be operated by the same entity."""

    cluster_id: str
    wallets: list[str]
    common_funding: list[str]
    timing_correlation: float
    confidence: float


@dataclass
class EntityResolution:
    """Final entity-resolution output combining all evidence signals."""

    entity_id: str
    wallets: list[str]
    evidence: list[str]
    confidence: float


# --------------------------------------------------------------------- #
#  WalletClusterer                                                       #
# --------------------------------------------------------------------- #

class WalletClusterer:
    """Cluster related Solana wallets and resolve shared entities.

    The clustering pipeline proceeds in three stages:

    1. **Transaction-graph construction** – build an adjacency dict from
       shared address co-occurrence within decoded transactions.
    2. **Union-Find clustering** – merge addresses whose edge weight
       exceeds a configurable threshold.
    3. **Entity resolution** – enrich each cluster with common-funding
       and timing-correlation evidence and assign a confidence score.
    """

    def __init__(
        self,
        solana_client: Optional[SolanaClient] = None,
        timing_threshold: float = 0.7,
        funding_threshold: float = 0.8,
    ) -> None:
        self._client = solana_client or SolanaClient()
        self.timing_threshold = timing_threshold
        self.funding_threshold = funding_threshold

        # Cached intermediate results.
        self._clusters: list[WalletCluster] = []
        self._funding_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    #  Stage 1: Transaction graph                                          #
    # ------------------------------------------------------------------ #

    def build_transaction_graph(self, transactions: list[dict]) -> dict:
        """Build a weighted adjacency dict from transaction co-occurrence.

        Two wallet addresses are linked if they appear together in the
        same transaction's account keys.  The edge weight is the number
        of co-occurrences normalised by the total number of transactions
        analysed.

        Args:
            transactions: Decoded transaction dicts (as returned by
                :pymethod:`SolanaClient.get_transaction`).

        Returns:
            ``{address: {neighbour: weight, …}, …}`` adjacency dict.
        """
        co_occurrence: dict[tuple[str, str], int] = defaultdict(int)
        address_set: set[str] = set()

        for tx in transactions:
            try:
                msg = tx.get("transaction", {}).get("message", {})
                keys = msg.get("accountKeys", [])
                addrs: list[str] = []
                for k in keys:
                    addr = k.get("pubkey", k) if isinstance(k, dict) else str(k)
                    # Skip program / system addresses (short or well-known).
                    if len(addr) >= 32:
                        addrs.append(addr)

                address_set.update(addrs)

                # Count pairwise co-occurrence (undirected).
                for i in range(len(addrs)):
                    for j in range(i + 1, len(addrs)):
                        a, b = tuple(sorted((addrs[i], addrs[j])))
                        co_occurrence[(a, b)] += 1
            except Exception:
                logger.debug("Skipping malformed transaction in graph build")

        total_tx = max(len(transactions), 1)
        graph: dict[str, dict[str, float]] = defaultdict(dict)
        for (a, b), count in co_occurrence.items():
            weight = count / total_tx
            graph[a][b] = weight
            graph[b][a] = weight

        # Ensure isolated nodes still appear in the graph.
        for addr in address_set:
            graph.setdefault(addr, {})

        return dict(graph)

    # ------------------------------------------------------------------ #
    #  Stage 2: Union-Find clustering                                      #
    # ------------------------------------------------------------------ #

    def find_clusters(
        self, graph: dict, threshold: float = 0.5
    ) -> list[WalletCluster]:
        """Cluster graph nodes whose edge weight exceeds *threshold*.

        After merging via Union-Find, each resulting cluster is enriched
        with a timing-correlation score (average pairwise Pearson
        correlation of transaction-time histograms) and an overall
        confidence value.

        Args:
            graph: Adjacency dict produced by
                :meth:`build_transaction_graph`.
            threshold: Minimum normalised edge weight to merge two nodes.

        Returns:
            List of :class:`WalletCluster` objects.
        """
        uf = UnionFind()

        # Register every node and merge those above threshold.
        for node in graph:
            uf.find(node)

        for node, neighbours in graph.items():
            for neighbour, weight in neighbours.items():
                if weight >= threshold:
                    uf.union(node, neighbour)

        raw_clusters = uf.get_clusters()

        clusters: list[WalletCluster] = []
        for root, members in raw_clusters.items():
            wallet_list = sorted(members)
            if len(wallet_list) < 2:
                continue  # singletons are uninteresting

            # Compute pairwise timing correlations within the cluster.
            timing_corrs: list[float] = []
            for i in range(len(wallet_list)):
                for j in range(i + 1, len(wallet_list)):
                    times_a = self._get_wallet_tx_times(wallet_list[i])
                    times_b = self._get_wallet_tx_times(wallet_list[j])
                    corr = self.calculate_timing_correlation(times_a, times_b)
                    timing_corrs.append(corr)

            avg_timing = (
                sum(timing_corrs) / len(timing_corrs)
                if timing_corrs
                else 0.0
            )

            # Confidence: 40 % cluster size, 60 % timing correlation.
            size_factor = min(len(wallet_list) / 10.0, 1.0)
            confidence = 0.4 * size_factor + 0.6 * max(avg_timing, 0.0)

            clusters.append(
                WalletCluster(
                    cluster_id=f"cluster-{root[:8]}",
                    wallets=wallet_list,
                    common_funding=[],
                    timing_correlation=round(avg_timing, 4),
                    confidence=round(confidence, 4),
                )
            )

        self._clusters = clusters
        return clusters

    # ------------------------------------------------------------------ #
    #  Funding analysis                                                    #
    # ------------------------------------------------------------------ #

    def detect_common_funding(self, wallets: list[str]) -> list[str]:
        """Trace funding sources for *wallets* and return shared parents.

        For each wallet the method fetches recent transaction history,
        extracts inbound SOL transfer senders, and intersects the
        resulting sets.

        Args:
            wallets: Wallet addresses to analyse.

        Returns:
            Sorted list of addresses that funded two or more of the
            provided wallets.
        """
        funding_sources: dict[str, set[str]] = {}

        for wallet in wallets:
            sources: set[str] = set()
            try:
                sigs = self._client.get_transaction_signatures(
                    wallet, limit=50
                )
                for sig_info in sigs[:30]:
                    if sig_info.get("err"):
                        continue
                    tx = self._client.get_transaction(sig_info["signature"])
                    if not tx:
                        continue
                    sender = self._extract_sol_sender(tx, wallet)
                    if sender:
                        sources.add(sender)
                    time.sleep(config.RPC_RATE_LIMIT_DELAY)
            except Exception:
                logger.debug(
                    "Error fetching funding sources for %s", wallet
                )

            funding_sources[wallet] = sources
            self._funding_cache[wallet] = sorted(sources)

        # Find addresses that appear as funding sources for >= 2 wallets.
        all_sources = [s for s in funding_sources.values()]
        if len(all_sources) < 2:
            return []

        # Count how many wallets each source funded.
        source_counts: dict[str, int] = defaultdict(int)
        for src_set in all_sources:
            for addr in src_set:
                source_counts[addr] += 1

        common = sorted(
            addr for addr, cnt in source_counts.items() if cnt >= 2
        )
        return common

    # ------------------------------------------------------------------ #
    #  Timing correlation                                                  #
    # ------------------------------------------------------------------ #

    def calculate_timing_correlation(
        self,
        wallet_a_times: list[float],
        wallet_b_times: list[float],
    ) -> float:
        """Pearson correlation on binned transaction-time histograms.

        Both timestamp lists are binned into ``_DEFAULT_TIME_BINS``
        equal-width buckets spanning the union of both time ranges.
        Pearson *r* is then computed on the resulting count vectors.

        Args:
            wallet_a_times: Unix timestamps for wallet A's transactions.
            wallet_b_times: Unix timestamps for wallet B's transactions.

        Returns:
            Pearson correlation coefficient in [-1, 1], or 0.0 when
            the computation is undefined (e.g. constant vectors).
        """
        if not wallet_a_times or not wallet_b_times:
            return 0.0

        all_times = wallet_a_times + wallet_b_times
        t_min = min(all_times)
        t_max = max(all_times)
        if t_max == t_min:
            return 1.0  # identical single-point distributions

        bins = _DEFAULT_TIME_BINS
        bin_width = (t_max - t_min) / bins

        hist_a = [0] * bins
        hist_b = [0] * bins

        for t in wallet_a_times:
            idx = min(int((t - t_min) / bin_width), bins - 1)
            hist_a[idx] += 1

        for t in wallet_b_times:
            idx = min(int((t - t_min) / bin_width), bins - 1)
            hist_b[idx] += 1

        # Pearson correlation.
        arr_a = np.array(hist_a, dtype=np.float64)
        arr_b = np.array(hist_b, dtype=np.float64)

        std_a = np.std(arr_a)
        std_b = np.std(arr_b)
        if std_a == 0.0 or std_b == 0.0:
            return 0.0

        mean_a = np.mean(arr_a)
        mean_b = np.mean(arr_b)
        cov = np.mean((arr_a - mean_a) * (arr_b - mean_b))

        return float(cov / (std_a * std_b))

    # ------------------------------------------------------------------ #
    #  Entity resolution                                                   #
    # ------------------------------------------------------------------ #

    def resolve_entities(self) -> list[EntityResolution]:
        """Combine clustering, funding, and timing evidence into final
        entity resolutions.

        Must be called after :meth:`find_clusters`.  For each cluster
        the method:

        1. Looks up common funding sources.
        2. Evaluates timing correlation against *timing_threshold*.
        3. Aggregates evidence strings and computes a confidence score.

        Returns:
            List of :class:`EntityResolution` objects.
        """
        resolutions: list[EntityResolution] = []

        for cluster in self._clusters:
            evidence: list[str] = []
            confidence_components: list[float] = []

            # Evidence 1: cluster size.
            evidence.append(
                f"cluster_size={len(cluster.wallets)}"
            )
            confidence_components.append(
                min(len(cluster.wallets) / 10.0, 1.0)
            )

            # Evidence 2: common funding.
            try:
                common = self.detect_common_funding(cluster.wallets)
                cluster.common_funding = common
                if common:
                    evidence.append(
                        f"common_funding_sources={len(common)}"
                    )
                    confidence_components.append(
                        min(len(common) / 3.0, 1.0)
                    )
            except Exception:
                logger.debug(
                    "Funding analysis failed for cluster %s",
                    cluster.cluster_id,
                )

            # Evidence 3: timing correlation.
            if cluster.timing_correlation >= self.timing_threshold:
                evidence.append(
                    f"timing_corr={cluster.timing_correlation:.2f}"
                )
                confidence_components.append(cluster.timing_correlation)

            # Aggregate confidence: mean of all signal components.
            overall_confidence = (
                sum(confidence_components) / len(confidence_components)
                if confidence_components
                else 0.0
            )

            resolutions.append(
                EntityResolution(
                    entity_id=cluster.cluster_id.replace(
                        "cluster-", "entity-"
                    ),
                    wallets=cluster.wallets,
                    evidence=evidence,
                    confidence=round(overall_confidence, 4),
                )
            )

        return resolutions

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_wallet_tx_times(self, address: str) -> list[float]:
        """Fetch recent transaction timestamps for *address*."""
        try:
            sigs = self._client.get_transaction_signatures(
                address, limit=100
            )
            times = [
                float(s["blockTime"])
                for s in sigs
                if s.get("blockTime")
            ]
            return times
        except Exception:
            logger.debug(
                "Could not fetch tx times for wallet %s", address
            )
            return []

    @staticmethod
    def _extract_sol_sender(tx: dict, recipient: str) -> Optional[str]:
        """Extract the SOL sender address for a native transfer to *recipient*.

        Inspects the pre/post SOL balance deltas in the transaction
        metadata to identify who sent SOL to *recipient*.
        """
        try:
            meta = tx.get("meta", {})
            pre = meta.get("preBalances", [])
            post = meta.get("postBalances", [])
            keys_raw = (
                tx.get("transaction", {})
                .get("message", {})
                .get("accountKeys", [])
            )
            keys: list[str] = [
                (k.get("pubkey", k) if isinstance(k, dict) else str(k))
                for k in keys_raw
            ]

            for i, key in enumerate(keys):
                if i >= len(pre) or i >= len(post):
                    break
                delta = post[i] - pre[i]
                # Negative delta means this account *sent* lamports.
                if delta < 0 and key != recipient:
                    return key
        except Exception:
            pass
        return None
