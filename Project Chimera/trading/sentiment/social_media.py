"""Social media intelligence for crypto sentiment analysis.

Monitors Twitter/X, Reddit, Discord, and Telegram for token mentions,
tracks influencer activity, and detects coordinated shilling campaigns.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SocialPost:
    """A single social-media post across any platform."""
    platform: str           # "twitter", "reddit", "discord", "telegram"
    author: str
    content: str
    timestamp: datetime
    engagement: int         # likes + retweets / upvotes / reactions
    followers: int          # author's follower count (0 if unknown)


@dataclass(frozen=True)
class InfluencerMention:
    """Record of an influencer mentioning a token."""
    influencer: str
    token: str
    timestamp: datetime
    engagement_rate: float   # 0.0–1.0
    follower_count: int


@dataclass(frozen=True)
class ShillingAlert:
    """Alert raised when coordinated shilling is suspected."""
    token: str
    confidence: float        # 0.0–1.0
    evidence: list[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# SocialMediaAnalyzer
# ---------------------------------------------------------------------------

class SocialMediaAnalyzer:
    """Async social-media intelligence engine.

    Integrates with Twitter API v2, Reddit (OAuth2), Discord/Telegram
    webhooks, and provides influencer tracking with coordinated shilling
    detection based on z-score burst analysis.
    """

    # Default subreddits to monitor
    DEFAULT_SUBREDDITS = ("cryptocurrency", "solana", "CryptoMoonShots")

    def __init__(
        self,
        twitter_bearer_token: str | None = None,
        reddit_client_id: str | None = None,
        reddit_client_secret: str | None = None,
        reddit_user_agent: str = "ProjectChimera/1.0",
        session: aiohttp.ClientSession | None = None,
        request_timeout: float = 15.0,
        influencer_min_followers: int = 10_000,
        shilling_zscore_threshold: float = 2.5,
    ) -> None:
        """Initialise the SocialMediaAnalyzer.

        Args:
            twitter_bearer_token: Twitter API v2 bearer token.
            reddit_client_id: Reddit OAuth2 application client ID.
            reddit_client_secret: Reddit OAuth2 application secret.
            reddit_user_agent: User-Agent string for Reddit API.
            session: Reusable *aiohttp* client session.
            request_timeout: HTTP request timeout in seconds.
            influencer_min_followers: Minimum followers to count as influencer.
            shilling_zscore_threshold: Z-score above which a mention burst
                is flagged as potential shilling.
        """
        self._twitter_token = twitter_bearer_token
        self._reddit_client_id = reddit_client_id
        self._reddit_client_secret = reddit_client_secret
        self._reddit_user_agent = reddit_user_agent
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._influencer_min_followers = influencer_min_followers
        self._shilling_zscore_threshold = shilling_zscore_threshold
        self._reddit_access_token: str | None = None

    # -- session management -------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ======================================================================
    # Twitter / X
    # ======================================================================

    async def analyze_twitter(
        self,
        query: str,
        limit: int = 100,
    ) -> list[SocialPost]:
        """Search Twitter/X for recent tweets matching *query*.

        Uses the Twitter API v2 ``/tweets/search/recent`` endpoint with
        engagement metrics (``public_metrics``) and author expansion.

        Args:
            query: Search query string (supports Twitter search operators).
            limit: Maximum number of tweets to return (max 100 per request).

        Returns:
            List of :class:`SocialPost` instances.
        """
        if not self._twitter_token:
            logger.warning("Twitter bearer token not configured; skipping.")
            return []

        session = await self._get_session()
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self._twitter_token}"}
        params: dict[str, str] = {
            "query": query,
            "max_results": str(min(limit, 100)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "public_metrics,username",
        }

        posts: list[SocialPost] = []
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Twitter API error %d: %s", resp.status, body[:300])
                    return []
                data = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Twitter request failed: %s", exc)
            return []

        # Build author lookup
        users: dict[str, dict[str, Any]] = {}
        for user in data.get("includes", {}).get("users", []):
            users[user["id"]] = user

        for tweet in data.get("data", []):
            metrics = tweet.get("public_metrics", {})
            engagement = (
                metrics.get("like_count", 0)
                + metrics.get("retweet_count", 0)
                + metrics.get("reply_count", 0)
                + metrics.get("quote_count", 0)
            )

            author_id = tweet.get("author_id", "")
            author_info = users.get(author_id, {})
            author_name = author_info.get("username", author_id)
            followers = author_info.get("public_metrics", {}).get("followers_count", 0)

            created_at = _parse_iso(tweet.get("created_at", ""))

            posts.append(
                SocialPost(
                    platform="twitter",
                    author=author_name,
                    content=tweet.get("text", ""),
                    timestamp=created_at,
                    engagement=engagement,
                    followers=followers,
                )
            )

        logger.info("Fetched %d tweets for query '%s'", len(posts), query)
        return posts

    # ======================================================================
    # Influencer tracking
    # ======================================================================

    async def track_influencers(
        self,
        token: str,
    ) -> list[InfluencerMention]:
        """Identify influencer mentions of *token* on Twitter.

        An influencer is defined as an account with at least
        ``influencer_min_followers`` followers.  For each qualifying
        mention the engagement rate is calculated as
        ``engagement / followers``.

        Args:
            token: Token name or ticker to search for.

        Returns:
            List of :class:`InfluencerMention` sorted by engagement rate
            descending.
        """
        posts = await self.analyze_twitter(f"${token} OR #{token}", limit=100)

        mentions: list[InfluencerMention] = []
        for post in posts:
            if post.followers < self._influencer_min_followers:
                continue

            engagement_rate = (
                post.engagement / post.followers if post.followers > 0 else 0.0
            )

            mentions.append(
                InfluencerMention(
                    influencer=post.author,
                    token=token,
                    timestamp=post.timestamp,
                    engagement_rate=round(engagement_rate, 6),
                    follower_count=post.followers,
                )
            )

        mentions.sort(key=lambda m: m.engagement_rate, reverse=True)
        logger.info(
            "Found %d influencer mentions for '%s'", len(mentions), token,
        )
        return mentions

    # ======================================================================
    # Shilling detection
    # ======================================================================

    def detect_shilling_campaign(
        self,
        mentions: list[SocialPost],
    ) -> ShillingAlert | None:
        """Detect coordinated shilling using z-score burst analysis.

        The method buckets mentions into 1-hour windows, computes the mean
        and standard deviation of mention counts, and flags if the most
        recent window exceeds the configured z-score threshold.

        Args:
            mentions: List of :class:`SocialPost` to analyse.

        Returns:
            A :class:`ShillingAlert` if a campaign is detected, else *None*.
        """
        if len(mentions) < 5:
            return None

        # Bucket into 1-hour windows
        buckets: dict[str, int] = {}
        for post in mentions:
            bucket_key = post.timestamp.strftime("%Y-%m-%d-%H")
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

        if len(buckets) < 3:
            return None

        counts = list(buckets.values())
        mean = statistics.mean(counts)
        stdev = statistics.pstdev(counts)

        if stdev == 0:
            return None

        # Check the most recent bucket
        sorted_keys = sorted(buckets.keys())
        latest_count = buckets[sorted_keys[-1]]
        z_score = (latest_count - mean) / stdev

        if z_score < self._shilling_zscore_threshold:
            return None

        # Collect evidence
        evidence: list[str] = [
            f"Z-score: {z_score:.2f} (threshold: {self._shilling_zscore_threshold})",
            f"Latest hour mentions: {latest_count} (mean: {mean:.1f}, stdev: {stdev:.1f})",
        ]

        # Check for repeated authors
        author_counts: dict[str, int] = {}
        for post in mentions:
            author_counts[post.author] = author_counts.get(post.author, 0) + 1
        repeat_authors = {a: c for a, c in author_counts.items() if c >= 3}
        if repeat_authors:
            top = sorted(repeat_authors.items(), key=lambda x: x[1], reverse=True)[:5]
            evidence.append(
                f"Repeat authors: {', '.join(f'{a}({c}x)' for a, c in top)}"
            )

        # Check for low-follower swarm
        low_follower_posts = [p for p in mentions if 0 < p.followers < 500]
        if len(low_follower_posts) > len(mentions) * 0.5:
            evidence.append(
                f"Low-follower swarm: {len(low_follower_posts)}/{len(mentions)} "
                f"posts from accounts with <500 followers"
            )

        # Check for similar content (high duplication)
        contents_lower = [p.content.lower().strip() for p in mentions]
        unique_ratio = len(set(contents_lower)) / len(contents_lower) if contents_lower else 1.0
        if unique_ratio < 0.6:
            evidence.append(
                f"Content similarity: only {unique_ratio:.0%} unique messages"
            )

        # Confidence based on z-score magnitude
        confidence = min(1.0, z_score / (self._shilling_zscore_threshold * 2))

        # Infer token from most common word (simplified)
        token = _infer_token(mentions)

        alert = ShillingAlert(
            token=token,
            confidence=round(confidence, 4),
            evidence=evidence,
        )
        logger.warning("Shilling campaign detected: %s", alert)
        return alert

    # ======================================================================
    # Reddit
    # ======================================================================

    async def _reddit_auth(self) -> str | None:
        """Authenticate with Reddit OAuth2 and cache the access token."""
        if self._reddit_access_token:
            return self._reddit_access_token

        if not self._reddit_client_id or not self._reddit_client_secret:
            logger.warning("Reddit API credentials not configured; skipping.")
            return None

        session = await self._get_session()
        auth = aiohttp.BasicAuth(self._reddit_client_id, self._reddit_client_secret)
        data = {"grant_type": "client_credentials"}
        headers = {"User-Agent": self._reddit_user_agent}

        try:
            async with session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    logger.error("Reddit OAuth failed with status %d", resp.status)
                    return None
                body = await resp.json()
                self._reddit_access_token = body.get("access_token")
                return self._reddit_access_token
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Reddit OAuth request failed: %s", exc)
            return None

    async def analyze_reddit(
        self,
        subreddit: str,
        query: str,
        limit: int = 50,
    ) -> list[SocialPost]:
        """Search a subreddit for posts matching *query*.

        Args:
            subreddit: Subreddit name (without ``r/`` prefix).
            query: Search query.
            limit: Maximum posts to return.

        Returns:
            List of :class:`SocialPost` instances.
        """
        token = await self._reddit_auth()
        if not token:
            return []

        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self._reddit_user_agent,
        }
        params: dict[str, str] = {
            "q": query,
            "limit": str(min(limit, 100)),
            "sort": "new",
            "restrict_sr": "true",
            "t": "week",
        }
        url = f"https://oauth.reddit.com/r/{subreddit}/search"

        posts: list[SocialPost] = []
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.error("Reddit API error %d for r/%s", resp.status, subreddit)
                    return []
                data = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Reddit request failed: %s", exc)
            return []

        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            created_utc = post_data.get("created_utc", 0)
            timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

            engagement = (
                post_data.get("score", 0)
                + post_data.get("num_comments", 0)
            )

            posts.append(
                SocialPost(
                    platform="reddit",
                    author=post_data.get("author", "[deleted]"),
                    content=post_data.get("title", ""),
                    timestamp=timestamp,
                    engagement=engagement,
                    followers=0,
                )
            )

        logger.info(
            "Fetched %d Reddit posts from r/%s for '%s'",
            len(posts), subreddit, query,
        )
        return posts

    # ======================================================================
    # Discord / Telegram webhook monitoring
    # ======================================================================

    async def monitor_discord(
        self,
        webhook_url: str,
    ) -> dict[str, Any]:
        """Fetch Discord channel info and recent messages via a webhook.

        For monitoring purposes this hits the webhook info endpoint to
        retrieve guild and channel metadata.  Message content requires a
        bot token, so this returns metadata only.

        Args:
            webhook_url: Discord webhook URL.

        Returns:
            Dict with ``guild_id``, ``channel_id``, ``name``, and
            ``guild_name``.
        """
        session = await self._get_session()

        try:
            async with session.get(webhook_url) as resp:
                if resp.status != 200:
                    logger.error("Discord webhook returned %d", resp.status)
                    return {}
                data = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Discord webhook request failed: %s", exc)
            return {}

        result = {
            "guild_id": data.get("guild_id"),
            "channel_id": data.get("channel_id"),
            "name": data.get("name", ""),
            "guild_name": data.get("guild", {}).get("name", "") if isinstance(data.get("guild"), dict) else "",
        }
        logger.info("Discord webhook info: %s", result.get("name", "unknown"))
        return result

    # ======================================================================
    # Growth velocity
    # ======================================================================

    @staticmethod
    def calculate_growth_velocity(
        member_counts: list[tuple[datetime, int]],
    ) -> dict[str, float]:
        """Calculate community growth velocity via linear regression.

        Args:
            member_counts: Time-series of ``(timestamp, member_count)``
                pairs in chronological order.

        Returns:
            Dict with ``slope`` (members/hour), ``r_squared``,
            ``growth_rate_pct`` (hourly percentage change), and
            ``velocity_label`` ("accelerating", "steady", "declining").
        """
        if len(member_counts) < 2:
            return {
                "slope": 0.0,
                "r_squared": 0.0,
                "growth_rate_pct": 0.0,
                "velocity_label": "insufficient_data",
            }

        # Convert timestamps to hours since first data-point
        t0 = member_counts[0][0]
        xs = [(ts - t0).total_seconds() / 3600.0 for ts, _ in member_counts]
        ys = [float(count) for _, count in member_counts]

        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return {
                "slope": 0.0,
                "r_squared": 0.0,
                "growth_rate_pct": 0.0,
                "velocity_label": "steady",
            }

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_mean = sum_y / n
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Hourly growth rate relative to initial
        initial = ys[0] if ys[0] != 0 else 1.0
        growth_rate_pct = (slope / initial) * 100.0

        if growth_rate_pct > 0.5:
            label = "accelerating"
        elif growth_rate_pct < -0.5:
            label = "declining"
        else:
            label = "steady"

        return {
            "slope": round(slope, 4),
            "r_squared": round(r_squared, 4),
            "growth_rate_pct": round(growth_rate_pct, 4),
            "velocity_label": label,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp, falling back to *now*."""
    if not text:
        return datetime.now(timezone.utc)
    try:
        text = text.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        logger.debug("Unable to parse ISO timestamp: '%s'", text)
        return datetime.now(timezone.utc)


def _infer_token(mentions: list[SocialPost]) -> str:
    """Infer the most likely token name from mention content.

    Looks for $TICKER patterns and falls back to the most frequent
    capitalised word of length 2-6.
    """
    import re

    ticker_counts: dict[str, int] = {}
    for post in mentions:
        tickers = re.findall(r"\$([A-Za-z]{2,6})", post.content)
        for t in tickers:
            key = t.upper()
            ticker_counts[key] = ticker_counts.get(key, 0) + 1

    if ticker_counts:
        return max(ticker_counts, key=ticker_counts.get)  # type: ignore[arg-type]

    # Fallback: most frequent capitalised word
    word_counts: dict[str, int] = {}
    for post in mentions:
        words = re.findall(r"\b[A-Z]{2,6}\b", post.content)
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

    if word_counts:
        return max(word_counts, key=word_counts.get)  # type: ignore[arg-type]

    return "UNKNOWN"
