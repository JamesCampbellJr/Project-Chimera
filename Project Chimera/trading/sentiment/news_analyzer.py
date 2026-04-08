"""News and media monitoring for crypto sentiment analysis.

Provides RSS feed parsing, custom VADER-style sentiment analysis,
Google Trends tracking, and GitHub activity monitoring.
"""

from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom VADER-style sentiment lexicon
# ---------------------------------------------------------------------------

POSITIVE_LEXICON: dict[str, float] = {
    # General positive
    "good": 1.9, "great": 3.1, "excellent": 3.3, "amazing": 3.1,
    "awesome": 3.1, "fantastic": 3.1, "wonderful": 3.1, "positive": 1.5,
    "love": 2.5, "best": 3.0, "happy": 2.5, "growth": 1.8,
    "profit": 2.2, "gain": 2.0, "success": 2.5, "improve": 1.8,
    "upgrade": 1.8, "strong": 2.0, "win": 2.5, "support": 1.5,
    "opportunity": 1.8, "innovative": 2.0, "breakthrough": 2.8,
    "promising": 2.0, "optimistic": 2.2, "confident": 2.0,
    # Crypto-specific positive
    "bullish": 3.0, "moon": 2.8, "mooning": 3.2, "pump": 2.0,
    "rally": 2.5, "surge": 2.8, "soar": 2.8, "breakout": 2.5,
    "ath": 3.0, "adoption": 2.5, "partnership": 2.2, "listing": 2.0,
    "mainnet": 2.2, "launch": 1.8, "upgrade": 1.8, "staking": 1.5,
    "airdrop": 2.0, "defi": 1.5, "yield": 1.5, "tvl": 1.2,
    "accumulate": 1.8, "hodl": 2.0, "diamond": 2.0, "gem": 2.2,
    "undervalued": 2.0, "whale": 1.5, "recovery": 2.0,
    "integration": 1.8, "milestone": 2.0, "ecosystem": 1.5,
}

NEGATIVE_LEXICON: dict[str, float] = {
    # General negative
    "bad": -1.9, "terrible": -3.1, "horrible": -3.1, "awful": -3.0,
    "worst": -3.1, "poor": -1.9, "negative": -1.5, "hate": -2.7,
    "fail": -2.5, "failure": -2.8, "loss": -2.2, "lose": -2.0,
    "decline": -2.0, "drop": -1.8, "crash": -3.0, "risk": -1.5,
    "danger": -2.2, "threat": -2.0, "problem": -1.5, "issue": -1.0,
    "concern": -1.2, "fear": -2.0, "worried": -1.8, "uncertain": -1.5,
    "weak": -1.8, "vulnerable": -2.0, "exploit": -2.5,
    # Crypto-specific negative
    "bearish": -3.0, "dump": -2.5, "dumping": -2.8, "rug": -3.5,
    "rugpull": -3.8, "scam": -3.5, "fraud": -3.5, "hack": -3.2,
    "hacked": -3.5, "exploit": -3.0, "vulnerability": -2.5,
    "delist": -2.8, "ban": -2.5, "regulation": -1.5, "sec": -1.5,
    "lawsuit": -2.5, "ponzi": -3.5, "bubble": -2.0, "fud": -2.0,
    "sell": -1.0, "selloff": -2.5, "liquidation": -2.8,
    "capitulation": -3.0, "rekt": -3.0, "bagholding": -2.0,
    "rug-pull": -3.8, "exit-scam": -3.8, "depeg": -3.0,
    "insolvent": -3.5, "bankrupt": -3.5, "delay": -1.5,
}

INTENSITY_MODIFIERS: dict[str, float] = {
    "very": 1.3, "extremely": 1.5, "incredibly": 1.4, "absolutely": 1.4,
    "really": 1.2, "so": 1.2, "super": 1.3, "highly": 1.3,
    "massively": 1.4, "totally": 1.3, "completely": 1.3,
    "slightly": 0.6, "somewhat": 0.7, "barely": 0.5, "hardly": 0.5,
    "kind": 0.7, "kinda": 0.7, "sort": 0.7, "little": 0.6,
    "marginally": 0.6, "partly": 0.7,
}

NEGATION_WORDS: set[str] = {
    "not", "no", "never", "neither", "nobody", "nothing",
    "nowhere", "nor", "cannot", "can't", "won't", "wouldn't",
    "shouldn't", "couldn't", "doesn't", "didn't", "isn't",
    "aren't", "wasn't", "weren't", "don't", "haven't", "hasn't",
    "hadn't", "without", "lack", "lacking",
}

# Default RSS feeds
DEFAULT_RSS_FEEDS: dict[str, str] = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NewsArticle:
    """Represents a parsed news article."""
    title: str
    source: str
    url: str
    published_at: datetime
    content_snippet: str


@dataclass(frozen=True)
class SentimentResult:
    """Result of sentiment analysis on a piece of text."""
    score: float          # -1.0 to 1.0
    magnitude: float      # 0.0+ (absolute strength)
    label: str            # "positive", "negative", "neutral"
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NewsAnalyzer
# ---------------------------------------------------------------------------

class NewsAnalyzer:
    """Async news monitoring and sentiment analysis engine.

    Fetches crypto news via RSS, analyses text sentiment with a custom
    VADER-style lexicon, tracks Google Trends interest, and monitors
    GitHub repository activity.
    """

    def __init__(
        self,
        rss_feeds: dict[str, str] | None = None,
        github_token: str | None = None,
        session: aiohttp.ClientSession | None = None,
        request_timeout: float = 15.0,
        time_decay_hours: float = 48.0,
    ) -> None:
        """Initialise the NewsAnalyzer.

        Args:
            rss_feeds: Mapping of source name → RSS URL.  Falls back to
                the built-in default feeds when *None*.
            github_token: Optional GitHub personal access token for higher
                rate limits.
            session: Reusable *aiohttp* client session.
            request_timeout: Timeout in seconds for HTTP requests.
            time_decay_hours: Half-life (in hours) for time-decay weighting.
        """
        self.rss_feeds = rss_feeds or dict(DEFAULT_RSS_FEEDS)
        self.github_token = github_token
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._time_decay_hours = time_decay_hours

    # -- session management -------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (and lazily create) the shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # -- RSS fetching -------------------------------------------------------

    @staticmethod
    def _parse_rss_items(
        xml_text: str,
        source: str,
        limit: int,
    ) -> list[NewsArticle]:
        """Parse RSS/Atom XML and return :class:`NewsArticle` instances."""
        articles: list[NewsArticle] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.warning("Failed to parse RSS XML from %s", source)
            return articles

        # Support both RSS 2.0 (<item>) and Atom (<entry>)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        for item in items[:limit]:
            title = (
                _el_text(item, "title")
                or _el_text(item, "atom:title", ns)
                or ""
            )
            link = (
                _el_text(item, "link")
                or _el_attr(item, "atom:link", "href", ns)
                or ""
            )
            pub_text = (
                _el_text(item, "pubDate")
                or _el_text(item, "atom:updated", ns)
                or ""
            )
            description = (
                _el_text(item, "description")
                or _el_text(item, "atom:summary", ns)
                or ""
            )

            published_at = _parse_datetime(pub_text)
            snippet = _strip_html(description)[:500]

            articles.append(
                NewsArticle(
                    title=title,
                    source=source,
                    url=link,
                    published_at=published_at,
                    content_snippet=snippet,
                )
            )

        return articles

    async def fetch_news(
        self,
        query: str,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """Fetch news articles matching *query* from configured RSS feeds.

        Articles are filtered by whether *query* appears (case-insensitive)
        in the title or content snippet, then sorted newest-first.

        Args:
            query: Search term (e.g. a token name).
            limit: Maximum number of articles to return *per feed*.

        Returns:
            List of :class:`NewsArticle` instances, newest first.
        """
        session = await self._get_session()
        all_articles: list[NewsArticle] = []
        query_lower = query.lower()

        for source, url in self.rss_feeds.items():
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "RSS feed %s returned status %d", source, resp.status,
                        )
                        continue
                    xml_text = await resp.text()
            except (aiohttp.ClientError, TimeoutError) as exc:
                logger.error("Error fetching RSS feed %s: %s", source, exc)
                continue

            articles = self._parse_rss_items(xml_text, source, limit * 3)
            for article in articles:
                if (
                    query_lower in article.title.lower()
                    or query_lower in article.content_snippet.lower()
                ):
                    all_articles.append(article)

        all_articles.sort(key=lambda a: a.published_at, reverse=True)
        return all_articles[:limit]

    # -- VADER-style sentiment analysis ------------------------------------

    def analyze_text_sentiment(self, text: str) -> SentimentResult:
        """Analyse *text* and return a :class:`SentimentResult`.

        The analyser uses a crypto-specific lexicon with valence scores,
        intensity modifiers, and negation handling to produce a compound
        score normalised to the **-1.0 … +1.0** range.
        """
        if not text or not text.strip():
            return SentimentResult(
                score=0.0, magnitude=0.0, label="neutral", keywords=[],
            )

        tokens = _tokenize(text)
        sentiments: list[float] = []
        matched_keywords: list[str] = []

        for idx, token in enumerate(tokens):
            valence = _lookup_valence(token)
            if valence == 0.0:
                continue

            # Intensity modifier check (preceding word)
            modifier = 1.0
            if idx > 0:
                prev = tokens[idx - 1]
                if prev in INTENSITY_MODIFIERS:
                    modifier = INTENSITY_MODIFIERS[prev]

            # Negation check (within 3-word window before)
            negated = False
            for j in range(max(0, idx - 3), idx):
                if tokens[j] in NEGATION_WORDS:
                    negated = True
                    break

            adjusted = valence * modifier
            if negated:
                adjusted *= -0.75

            sentiments.append(adjusted)
            matched_keywords.append(token)

        if not sentiments:
            return SentimentResult(
                score=0.0, magnitude=0.0, label="neutral", keywords=[],
            )

        raw_sum = sum(sentiments)
        # Normalise using VADER-style formula: x / sqrt(x^2 + alpha)
        alpha = 15.0
        compound = raw_sum / math.sqrt(raw_sum ** 2 + alpha)
        compound = max(-1.0, min(1.0, compound))

        magnitude = sum(abs(s) for s in sentiments)

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            score=round(compound, 4),
            magnitude=round(magnitude, 4),
            label=label,
            keywords=matched_keywords,
        )

    # -- Google Trends ------------------------------------------------------

    async def get_trending_score(self, keyword: str) -> float:
        """Return a 0–100 interest score for *keyword* via Google Trends.

        Uses the public Google Trends embed endpoint which returns a
        simplified JSON-like payload without requiring an API key.

        Returns:
            Interest score 0–100, or 0.0 on failure.
        """
        session = await self._get_session()
        encoded = quote_plus(keyword)
        url = (
            "https://trends.google.com/trends/api/dailytrends"
            f"?hl=en-US&tz=0&geo=US&ns=15&q={encoded}"
        )
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Google Trends returned status %d for '%s'",
                        resp.status, keyword,
                    )
                    return 0.0
                body = await resp.text()

            # The response starts with ")]}'\n" – strip the prefix.
            body = body.lstrip(")]}'\n")

            import json
            data = json.loads(body)
            trending = data.get("default", {}).get("trendingSearchesDays", [])
            for day in trending:
                for search in day.get("trendingSearches", []):
                    title = search.get("title", {}).get("query", "").lower()
                    if keyword.lower() in title:
                        traffic_str = search.get("formattedTraffic", "0")
                        return _parse_traffic(traffic_str)

            return 0.0

        except (aiohttp.ClientError, TimeoutError, ValueError, KeyError) as exc:
            logger.error("Error fetching Google Trends for '%s': %s", keyword, exc)
            return 0.0

    # -- GitHub monitoring --------------------------------------------------

    async def monitor_github(self, repo_url: str) -> dict[str, Any]:
        """Return activity metrics for a GitHub repository.

        Args:
            repo_url: Full GitHub URL (e.g. ``https://github.com/owner/repo``).

        Returns:
            Dict with keys ``stars``, ``forks``, ``open_issues``,
            ``recent_commits``, ``watchers``, and ``activity_score``.
        """
        owner, repo = _parse_github_url(repo_url)
        if not owner or not repo:
            logger.error("Invalid GitHub URL: %s", repo_url)
            return _empty_github_metrics()

        session = await self._get_session()
        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        api_base = f"https://api.github.com/repos/{owner}/{repo}"

        # Fetch repo info
        repo_info: dict[str, Any] = {}
        try:
            async with session.get(api_base, headers=headers) as resp:
                if resp.status != 200:
                    logger.warning("GitHub API returned %d for %s", resp.status, repo_url)
                    return _empty_github_metrics()
                repo_info = await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.error("Error fetching GitHub repo info: %s", exc)
            return _empty_github_metrics()

        # Fetch recent commits (last 30 days)
        recent_commits = 0
        try:
            commits_url = f"{api_base}/commits?per_page=100&since={_days_ago_iso(30)}"
            async with session.get(commits_url, headers=headers) as resp:
                if resp.status == 200:
                    commits_data = await resp.json()
                    recent_commits = len(commits_data)
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.debug("Error fetching commits for %s: %s", repo_url, exc)

        stars = repo_info.get("stargazers_count", 0)
        forks = repo_info.get("forks_count", 0)
        open_issues = repo_info.get("open_issues_count", 0)
        watchers = repo_info.get("subscribers_count", 0)

        # Composite activity score (0–100)
        activity_score = min(
            100.0,
            (
                min(stars, 10000) / 100.0
                + min(forks, 2000) / 40.0
                + min(recent_commits, 200) / 2.0
                + min(watchers, 5000) / 100.0
            ),
        )

        return {
            "stars": stars,
            "forks": forks,
            "open_issues": open_issues,
            "recent_commits": recent_commits,
            "watchers": watchers,
            "activity_score": round(activity_score, 2),
        }

    # -- High-level convenience --------------------------------------------

    async def get_news_sentiment(self, token_name: str) -> SentimentResult:
        """Fetch recent news for *token_name* and return an aggregate sentiment.

        Recent articles are weighted higher (exponential time-decay).
        """
        articles = await self.fetch_news(token_name)
        if not articles:
            logger.info("No news articles found for '%s'", token_name)
            return SentimentResult(
                score=0.0, magnitude=0.0, label="neutral", keywords=[],
            )

        weighted_score = 0.0
        total_weight = 0.0
        all_keywords: list[str] = []
        total_magnitude = 0.0

        now = datetime.now(timezone.utc)
        for article in articles:
            text = f"{article.title}. {article.content_snippet}"
            result = self.analyze_text_sentiment(text)

            hours_old = max(
                (now - article.published_at).total_seconds() / 3600.0,
                0.01,
            )
            weight = math.exp(-0.693 * hours_old / self._time_decay_hours)

            weighted_score += result.score * weight
            total_magnitude += result.magnitude * weight
            total_weight += weight
            all_keywords.extend(result.keywords)

        if total_weight == 0.0:
            return SentimentResult(
                score=0.0, magnitude=0.0, label="neutral", keywords=[],
            )

        avg_score = weighted_score / total_weight
        avg_magnitude = total_magnitude / total_weight
        avg_score = max(-1.0, min(1.0, avg_score))

        if avg_score >= 0.05:
            label = "positive"
        elif avg_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # Deduplicate keywords preserving order
        seen: set[str] = set()
        unique_keywords: list[str] = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return SentimentResult(
            score=round(avg_score, 4),
            magnitude=round(avg_magnitude, 4),
            label=label,
            keywords=unique_keywords,
        )

    # -- time-decay helper (exposed for testing) ----------------------------

    def _time_decay_weight(self, hours_old: float) -> float:
        """Return an exponential decay weight (1.0 → 0.0)."""
        return math.exp(-0.693 * hours_old / self._time_decay_hours)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lower-case tokenisation keeping contractions intact."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\-]+", " ", text)
    return text.split()


def _lookup_valence(token: str) -> float:
    """Return the valence score for *token*, or 0.0 if unknown."""
    if token in POSITIVE_LEXICON:
        return POSITIVE_LEXICON[token]
    if token in NEGATIVE_LEXICON:
        return NEGATIVE_LEXICON[token]
    return 0.0


def _strip_html(text: str) -> str:
    """Remove HTML tags from *text*."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _el_text(
    parent: ET.Element,
    tag: str,
    ns: dict[str, str] | None = None,
) -> str | None:
    """Return the text of a child element, or *None*."""
    el = parent.find(tag, ns) if ns else parent.find(tag)
    return el.text.strip() if el is not None and el.text else None


def _el_attr(
    parent: ET.Element,
    tag: str,
    attr: str,
    ns: dict[str, str] | None = None,
) -> str | None:
    """Return an attribute of a child element, or *None*."""
    el = parent.find(tag, ns) if ns else parent.find(tag)
    return el.get(attr) if el is not None else None


def _parse_datetime(text: str) -> datetime:
    """Best-effort parse of an RSS/Atom date string."""
    if not text:
        return datetime.now(timezone.utc)

    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    logger.debug("Unable to parse date: '%s'; falling back to now.", text)
    return datetime.now(timezone.utc)


def _parse_traffic(traffic_str: str) -> float:
    """Convert Google Trends traffic strings like '100K+' to a float 0–100."""
    traffic_str = traffic_str.replace("+", "").replace(",", "").strip()
    multiplier = 1.0
    if traffic_str.endswith("K"):
        multiplier = 1_000
        traffic_str = traffic_str[:-1]
    elif traffic_str.endswith("M"):
        multiplier = 1_000_000
        traffic_str = traffic_str[:-1]

    try:
        value = float(traffic_str) * multiplier
    except ValueError:
        return 0.0

    # Normalise to 0–100 (1M+ → 100)
    return min(100.0, value / 10_000)


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        url.strip(),
    )
    if match:
        return match.group(1), match.group(2)
    return "", ""


def _days_ago_iso(days: int) -> str:
    """Return an ISO-8601 timestamp *days* in the past."""
    from datetime import timedelta
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_github_metrics() -> dict[str, Any]:
    """Return a zeroed-out GitHub metrics dict."""
    return {
        "stars": 0,
        "forks": 0,
        "open_issues": 0,
        "recent_commits": 0,
        "watchers": 0,
        "activity_score": 0.0,
    }
