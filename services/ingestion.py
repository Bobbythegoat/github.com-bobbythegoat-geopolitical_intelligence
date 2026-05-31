"""
services/ingestion.py
---------------------
Phase 1: Pull raw articles from external sources and store them.

Supported sources
-----------------
1. RSS feeds (via feedparser) — any public RSS URL
2. Per-ticker Yahoo Finance RSS with STRICT relevance gating
3. SEC EDGAR full-text search RSS
4. Manual / test data

Relevance Gating
----------------
Every article from a general feed (Yahoo Finance, CNBC, MarketWatch, etc.)
must pass a MINIMUM_RELEVANCE_SCORE before it is stored.  This score is the
fraction of MUST_MATCH_KEYWORDS (i.e. at least one from the required set) AND
a minimum hit-count of TICKER_RELEVANCE_KEYWORDS that directly maps the story
to one of the tracked tickers or macro themes.

Articles about earnings recaps, lifestyle content, sponsored articles, or
personal-finance advice are explicitly blocked by a NOISE_BLOCKLIST.

After storing, each article is automatically:
  - Clustered into an event  (clustering.cluster_article)
  - Processed for entities   (processing.process_article)
  - Checked for alerts       (alerts.try_trigger_alert)
"""

import hashlib
import re
import socket
from datetime import datetime, timedelta
from typing import List, Optional, Set


# Maximum age (hours) of an article to accept from RSS feeds.
# Articles published more than this many hours ago are discarded — they no longer
# move markets and only inflate "active events" with stale signal.
MAX_ARTICLE_AGE_HOURS: int = 36

# Global timeout for all RSS fetches (seconds). Prevents any single feed
# from hanging startup. feedparser respects socket timeouts.
_FEED_TIMEOUT_SECONDS = 10

from sqlalchemy.orm import Session

import database as db
from services.clustering import cluster_article
from services.processing import process_article
from services.alerts import try_trigger_alert

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Relevance filter configuration
# ---------------------------------------------------------------------------

# Minimum number of TRACKED_KEYWORDS that must appear in headline+summary
# for a general-purpose feed article to be accepted.
# Raised from 2 → 3 to eliminate borderline articles that mention one
# generic term (e.g. "war" in a sports context, "china" in a travel piece).
MIN_RELEVANCE_HITS = 3

# These are the high-value macro/geopolitical/market keywords we actually care
# about.  An article must match at least MIN_RELEVANCE_HITS of these to pass.
TRACKED_KEYWORDS: Set[str] = {
    # ── Macro / geopolitical ──────────────────────────────────────────────
    "war", "conflict", "sanction", "sanctions", "tariff", "tariffs",
    "embargo", "trade war", "geopolit", "invasion", "missile", "nuclear",
    "nato", "ukraine", "russia", "china", "taiwan", "iran", "israel",
    "hamas", "hezbollah", "middle east", "north korea", "dprk",
    "pentagon", "defense", "defence", "military", "arms", "treaty",
    "ceasefire", "cease-fire", "cease fire",
    "strait", "hormuz", "red sea", "houthi", "opec",
    # ── US executive / policy actions (directly move markets) ────────────
    "white house", "executive order", "oval office",
    "peace deal", "peace talks", "peace agreement",
    "trump", "administration announce", "us president",
    "us tariff", "us sanction", "trade policy",
    "us-china", "us-russia", "us-taiwan",
    "g7", "g20", "un security council", "wto ruling",
    # ── Macro / monetary ─────────────────────────────────────────────────
    "federal reserve", "fed rate", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp", "recession", "yield curve",
    "quantitative", "central bank", "treasury", "bond yield",
    "debt ceiling", "bank failure", "banking crisis",
    # ── Energy ───────────────────────────────────────────────────────────
    "oil", "crude", "brent", "wti", "petroleum", "lng", "natural gas",
    "pipeline", "refinery", "opec", "drilling", "energy crisis",
    "exxon", "chevron", "gazprom", "nordstream",
    # ── Technology / semiconductors ───────────────────────────────────────
    "semiconductor", "chip", "export ban", "export control",
    "nvidia", "tsmc", "apple", "foxconn", "supply chain",
    "artificial intelligence", "ai regulation", "tech war",
    # ── Defense / aerospace ───────────────────────────────────────────────
    "lockheed", "raytheon", "boeing", "defense contract", "f-35",
    "fighter jet", "hypersonic", "air defense",
    # ── Financial sector ─────────────────────────────────────────────────
    "goldman sachs", "jpmorgan", "jp morgan", "investment bank",
    "credit suisse", "silicon valley bank", "bank run",
    "capital markets", "ipo", "merger", "acquisition",
    # ── Commodities / agriculture ─────────────────────────────────────────
    "wheat", "grain", "food security", "famine", "fertilizer",
    "drought", "black sea", "agriculture", "gold", "bullion",
    # ── Regulatory / official ─────────────────────────────────────────────
    "sec filing", "8-k", "10-k", "earnings", "guidance cut",
    "guidance raise", "restatement", "fda", "regulatory", "antitrust",
    "doj", "sec investigation", "fraud",
}

# Articles whose headline or summary contain ANY of these patterns are
# immediately discarded as noise regardless of other keyword matches.
NOISE_BLOCKLIST: List[str] = [
    # Personal finance noise
    "best credit card", "personal loan", "mortgage rate", "refinance your",
    "save money", "budget tips", "retirement savings", "401k", "roth ira",
    "crypto tip", "bitcoin price prediction", "altcoin",
    # Lifestyle / clickbait
    "celebrities", "celebrity", "hollywood", "relationship advice",
    "weight loss", "health tip", "recipe", "travel deal",
    "horoscope", "quiz",
    # Sponsored / ad-adjacent
    "sponsored", "partner content", "advertisement", "promo code",
    "discount code", "affiliate",
    # Earnings-recap fluff with no macro signal
    "quarterly earnings beat expectations",  # too generic
    # Sports
    "nfl", "nba", "mlb", "nhl", "soccer", "football score",
    # Political entertainment / personality gossip (no market impact)
    "reality show", "faces backlash", "television appearance", "book deal",
    "speaking tour", "political rally", "campaign launch", "running for",
    "presidential ambition", "senate race", "election campaign",
    # Mass casualty events with no market link
    "mass shooting", "school shooting", "shooting remembered", "vigil for",
    # Sports misclassifications
    "world cup qualifier", "casting votes", "election result",
    "voting begins", "polls open", "ballot count",
]

# Compiled for speed
_NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NOISE_BLOCKLIST]

# Company/sector anchor terms — at least one must appear for the article
# to pass the company-relevance gate.  These are specific enough that a
# single match reliably signals the article is about our tracked universe.
COMPANY_RELEVANCE_ANCHORS: Set[str] = {
    # Semiconductor companies & products
    "nvidia", "nvda", "tsmc", "taiwan semiconductor", "asml", "applied materials",
    "amat", "lam research", "lrcx", "kla", "klac", "micron", "broadcom", "avgo",
    "qualcomm", "qcom", "intel", "intc", "amd", "advanced micro devices",
    "arm holdings", "marvell", "mrvl", "synopsys", "snps", "cadence", "cdns",
    "teradyne", "onsemi", "on semiconductor", "nxp semiconductors", "nxpi",
    "texas instruments", "txn", "analog devices", "super micro", "smci",
    "globalfoundries", "gfs", "united microelectronics", "umc", "microchip technology",
    # Semiconductor products / events
    "h100", "a100", "blackwell", "hopper", "mi300", "euv lithography",
    "semiconductor equipment", "wafer fab", "chip export", "chip ban",
    "export control", "chip restriction", "ai chip", "gpu cluster",
    "foundry capacity", "advanced node", "3nm", "2nm", "5nm", "hbm memory",
    "high bandwidth memory", "chip shortage", "semiconductor shortage",
    # Energy companies
    "exxon", "exxonmobil", "chevron", "opec", "aramco",
    # Defense companies
    "lockheed", "raytheon", "boeing", "northrop", "general dynamics",
    # Financial companies
    "goldman sachs", "jpmorgan", "jp morgan", "morgan stanley",
    # Tech companies
    "apple inc", "apple revenue", "foxconn", "iphone sales",
    # Key geopolitical + market events that reliably move our universe
    "taiwan strait", "strait of taiwan", "china taiwan", "pla navy",
    "china export", "china sanction", "china tech", "china chip",
    "hormuz", "oil sanction", "iran nuclear", "iran oil",
    "fed rate", "rate decision", "fomc", "treasury yield",
    "ukraine russia", "russia sanction",
    # US executive actions — these directly move markets
    "white house", "executive order", "ceasefire", "cease fire", "cease-fire",
    "peace deal", "peace agreement", "peace talks",
    "trump tariff", "trump sanction", "trump executive",
    "us-china trade", "us china relations",
    "pentagon statement", "state department",
    "wto", "g7 statement", "g20 communique",
    # ── AI Infrastructure ─────────────────────────────────────────────────────
    "microsoft azure", "google cloud", "aws", "meta ai", "oracle cloud",
    "palantir", "cloudflare", "snowflake", "crowdstrike", "vertiv",
    "dell server", "hpe greenlake", "openai", "anthropic",
    # ── Defense ───────────────────────────────────────────────────────────────
    "northrop", "general dynamics", "huntington ingalls", "l3harris",
    "leidos", "kratos", "pentagon contract", "defense budget", "dod award",
    "defense department", "air force contract", "navy contract",
    # ── Energy ────────────────────────────────────────────────────────────────
    "conocophillips", "eog resources", "schlumberger", "marathon petroleum",
    "phillips 66", "occidental", "halliburton", "oil output", "lng export",
    "natural gas export", "crude production", "refinery capacity",
}


# Market transmission keywords — at least ONE must appear for non-strict sources
# (unless the article already matches a company anchor) to ensure the article
# describes an actual economic mechanism, not just a geopolitical existence.
MARKET_TRANSMISSION_KEYWORDS: Set[str] = {
    "trade", "export", "import", "sanction", "tariff", "embargo", "ban",
    "supply chain", "revenue", "earnings", "profit", "loss", "contract",
    "restriction", "chip", "semiconductor", "oil", "crude", "gas",
    "interest rate", "fed rate", "rate hike", "rate cut", "bond yield",
    "market", "stock", "share price", "investor", "investment",
    "gdp", "recession", "inflation", "capital", "merger", "acquisition",
    "ipo", "filing", "guidance", "forecast", "subsidy", "penalty", "fine",
    "regulation", "antitrust", "export control", "technology ban",
    "defense contract", "weapons sale", "arms deal",
}


# ---------------------------------------------------------------------------
# Default RSS sources (curated for geopolitical/market intelligence)
# ---------------------------------------------------------------------------

DEFAULT_FEEDS = [
    # ── Breaking News Wire Services (highest priority — catches presidential actions) ──
    # Reuters retired feeds.reuters.com in 2020. Use the Yahoo-hosted Reuters
    # mirror plus the Google News topic feed as drop-in replacements.
    {
        "name": "Reuters via Google News (World)",
        "url": "https://news.google.com/rss/search?q=site:reuters.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 1,
        "strict_filter": False,   # wire service: news judgment is already editorial
    },
    {
        "name": "Reuters via Google News (Business)",
        "url": "https://news.google.com/rss/search?q=site:reuters.com+business+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "Reuters via Google News (Technology)",
        "url": "https://news.google.com/rss/search?q=site:reuters.com+technology+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        # rsshub.app is rate-limited and currently returns HTML error pages
        # that feedparser rejects with a syntax error. Use Google News' AP
        # topic mirror, which is stable and authenticated-free.
        "name": "AP News Top Stories",
        "url": "https://news.google.com/rss/search?q=site:apnews.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 1,
        "strict_filter": False,   # AP is a primary wire — catch everything that passes noise filter
    },
    {
        "name": "AP Business News",
        "url": "https://news.google.com/rss/search?q=site:apnews.com+business+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 1,
        "strict_filter": False,
    },
    # ── Geopolitical / World News ─────────────────────────────────────────
    # These sources are EXPLICITLY geopolitical — strict_filter=False means
    # we accept any article that passes the keyword threshold (3+ tracked terms).
    # We do NOT require a company name match because a ceasefire announcement,
    # executive order, or peace deal is inherently market-relevant even without
    # mentioning NVDA or TSMC directly.
    {
        "name": "BBC World News",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "is_primary": 1,
        "strict_filter": False,   # geopolitical source — keyword threshold is sufficient
    },
    {
        "name": "BBC Business",
        "url": "https://feeds.bbci.co.uk/news/business/rss.xml",
        "is_primary": 1,
        "strict_filter": True,    # business content — require company relevance
    },
    {
        "name": "The Guardian World",
        "url": "https://www.theguardian.com/world/rss",
        "is_primary": 1,
        "strict_filter": False,   # geopolitical source
    },
    {
        "name": "Al Jazeera English",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "is_primary": 1,
        "strict_filter": False,   # geopolitical source — strong for Middle East / Asia
    },
    {
        "name": "NPR World News",
        "url": "https://feeds.npr.org/1004/rss.xml",
        "is_primary": 1,
        "strict_filter": False,   # geopolitical source
    },
    {
        "name": "Politico",
        "url": "https://rss.politico.com/politics-news.xml",
        "is_primary": 1,
        "strict_filter": False,   # policy/political news — executive orders, trade policy, etc.
    },
    # ── Financial / Market News — strict: must have company or strong keyword signal ──
    {
        "name": "CNBC World News",
        "url": "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        "name": "CNBC Finance",
        "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        "name": "CNBC Technology",
        "url": "https://www.cnbc.com/id/19854910/device/rss/rss.html",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        "name": "MarketWatch Top Stories",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        "name": "Seeking Alpha Market News",
        "url": "https://seekingalpha.com/market_currents.xml",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        # Investopedia's feedbuilder endpoint now serves a malformed XML page;
        # mirror via Google News until they fix it.
        "name": "Investopedia News",
        "url": "https://news.google.com/rss/search?q=site:investopedia.com+when:1d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 0,
        "strict_filter": True,
    },
    # ── Semiconductor / Tech specific ─────────────────────────────────────
    {
        "name": "EE Times",
        "url": "https://www.eetimes.com/feed/",
        "is_primary": 0,
        "strict_filter": False,   # semiconductor trade press — always relevant
    },
    {
        "name": "Tom's Hardware",
        "url": "https://www.tomshardware.com/feeds/all",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        # AnandTech's site shut down editorial in mid-2024 and the feed now
        # returns malformed XML; mirror via Google News.
        "name": "AnandTech",
        "url": "https://news.google.com/rss/search?q=site:anandtech.com+when:7d&hl=en-US&gl=US&ceid=US:en",
        "is_primary": 0,
        "strict_filter": True,
    },
    # ── Regulatory / Official ─────────────────────────────────────────────
    {
        "name": "SEC EDGAR 8-K Filings",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=20&output=atom",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "Federal Reserve Press Releases",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "is_primary": 1,
        "strict_filter": False,
    },
    # ── Defence / Energy ─────────────────────────────────────────────────
    {
        "name": "Defense News",
        "url": "https://www.defensenews.com/arc/outboundfeeds/rss/?outputType=xml",
        "is_primary": 0,
        "strict_filter": False,
    },
    {
        "name": "Breaking Defense",
        "url": "https://breakingdefense.com/feed/",
        "is_primary": 0,
        "strict_filter": False,
    },
    # ── Geopolitical analysis ─────────────────────────────────────────────
    {
        "name": "Foreign Policy",
        "url": "https://foreignpolicy.com/feed/",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "War on the Rocks",
        "url": "https://warontherocks.com/feed/",
        "is_primary": 1,
        "strict_filter": False,
    },
    # ── SEC EDGAR Company Filings (primary official source) ───────────────────
    {
        "name": "SEC EDGAR Semiconductors 8-K",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=semiconductor&output=atom",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "SEC EDGAR AI 8-K",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=artificial+intelligence&output=atom",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "SEC EDGAR Defense 8-K",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=defense+contract&output=atom",
        "is_primary": 1,
        "strict_filter": False,
    },
    {
        "name": "SEC EDGAR Energy 8-K",
        "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=oil+energy&output=atom",
        "is_primary": 1,
        "strict_filter": False,
    },
    # ── Corporate News Wires ──────────────────────────────────────────────────
    {
        "name": "PR Newswire Technology",
        "url": "https://www.prnewswire.com/rss/news-releases-list.rss?category=technology",
        "is_primary": 0,
        "strict_filter": True,
    },
    {
        "name": "BusinessWire Technology",
        "url": "https://feed.businesswire.com/rss/home/?rss=G22",
        "is_primary": 0,
        "strict_filter": True,
    },
    # ── Sector Trade Media ────────────────────────────────────────────────────
    {
        "name": "Oil Price News",
        "url": "https://oilprice.com/rss/main",
        "is_primary": 0,
        "strict_filter": True,
    },
]


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def _relevance_score(headline: str, content: str) -> int:
    """
    Count how many TRACKED_KEYWORDS appear in the combined headline + content.
    Returns an integer hit count.
    """
    text = (headline + " " + content).lower()
    return sum(1 for kw in TRACKED_KEYWORDS if kw in text)


def _is_noise(headline: str, content: str) -> bool:
    """Return True if any noise pattern matches headline or summary."""
    text = headline + " " + content
    return any(p.search(text) for p in _NOISE_PATTERNS)


def _has_company_relevance(headline: str, content: str) -> bool:
    """
    Return True if the article directly mentions at least one tracked company
    or a high-specificity sector event.  This is the second gate that prevents
    generic world news (sports results, political gossip, lifestyle) from
    entering the pipeline even if it passes the keyword count check.
    """
    text = (headline + " " + content).lower()
    return any(anchor in text for anchor in COMPANY_RELEVANCE_ANCHORS)


def _has_market_impact(headline: str, content: str) -> bool:
    """
    Return True if at least one MARKET_TRANSMISSION_KEYWORDS term appears in
    the headline or content.  This ensures non-strict source articles describe
    an actual economic mechanism rather than pure geopolitical existence.
    """
    text = (headline + " " + content).lower()
    return any(kw in text for kw in MARKET_TRANSMISSION_KEYWORDS)


def _passes_relevance_gate(headline: str, content: str,
                            strict: bool) -> bool:
    """
    Two-stage gate — both stages must pass:

    Stage 1 — Noise rejection:
      Always discard articles matching NOISE_BLOCKLIST patterns.

    Stage 2 — Company + topic relevance:
      An article must either:
        (a) Mention at least one company/sector anchor (COMPANY_RELEVANCE_ANCHORS), OR
        (b) Score MIN_RELEVANCE_HITS on TRACKED_KEYWORDS
      AND (if strict=True) also meet the MIN_RELEVANCE_HITS keyword threshold.

    The result: general news about sports, politics, lifestyle, or vague
    "war" references that have no connection to our tracked universe are
    filtered out at ingestion time before they can pollute stock scores.
    """
    if _is_noise(headline, content):
        return False

    has_company = _has_company_relevance(headline, content)
    hits = _relevance_score(headline, content)

    if strict:
        # Strict feeds: must have company relevance AND keyword threshold
        return has_company and hits >= MIN_RELEVANCE_HITS

    # Non-strict feeds (official sources, trade press):
    # Accept if:
    #   (a) Company anchor present AND at least 2 keyword hits, OR
    #   (b) Market transmission keyword present AND at least 3 keyword hits.
    # Pure geopolitical news without any economic mechanism gets filtered.
    has_market = _has_market_impact(headline, content)
    return (has_company and hits >= 2) or (has_market and hits >= MIN_RELEVANCE_HITS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _article_fingerprint(headline: str, source: str) -> str:
    """Deduplication hash based on headline + source."""
    return hashlib.md5(f"{source}::{headline}".lower().encode()).hexdigest()


def _is_duplicate(fingerprint: str, session: Session) -> bool:
    return (
        session.query(db.Article)
        .filter(db.Article.url == fingerprint)
        .first()
    ) is not None


def _post_process(article_id: int, session: Session):
    """Run the full pipeline on a newly stored article."""
    cluster_article(article_id, session)
    process_article(article_id, session)

    article = session.get(db.Article, article_id)
    if article and article.event_id:
        try_trigger_alert(article.event_id, session)


# ---------------------------------------------------------------------------
# Core ingest functions
# ---------------------------------------------------------------------------

def ingest_rss_feed(feed_url: str, source_name: str, session: Session,
                    is_primary: int = 0,
                    strict_filter: bool = False) -> List[int]:
    """
    Fetch an RSS feed and store new articles that pass the relevance gate.
    Returns list of newly created article IDs.

    Parameters
    ----------
    strict_filter : bool
        If True, apply MIN_RELEVANCE_HITS keyword gating (for general
        financial feeds like Yahoo Finance).  If False, accept everything
        that is not in the noise blocklist.
    """
    if not FEEDPARSER_AVAILABLE:
        print("feedparser not installed — skipping RSS ingestion.")
        return []

    print(f"  Fetching: {source_name} ...")

    # Apply socket timeout so a single slow/hung feed can't block everything
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(_FEED_TIMEOUT_SECONDS)
    try:
        # SEC EDGAR rejects requests without a real User-Agent (returns an
        # error page that fails XML parsing). Several other publishers also
        # block the default feedparser UA. Pass a descriptive UA on every
        # fetch — required per https://www.sec.gov/os/accessing-edgar-data.
        feed = feedparser.parse(
            feed_url,
            agent=(
                "GeopoliticalIntelligence/1.0 "
                "(contact: augustus.soedarmono@gmail.com)"
            ),
        )
    except Exception as e:
        print(f"  ERROR fetching {source_name}: {e}")
        return []
    finally:
        socket.setdefaulttimeout(old_timeout)

    # feedparser marks malformed feeds with bozo=True; log but continue
    if getattr(feed, "bozo", False) and not feed.entries:
        exc = getattr(feed, "bozo_exception", "unknown")
        print(f"  {source_name}: feed parse warning ({exc}) — 0 usable entries")
        return []

    if not feed.entries:
        # Check HTTP status if available
        status = getattr(feed, "status", None)
        if status and status >= 400:
            print(f"  {source_name}: HTTP {status} — skipping")
        else:
            print(f"  {source_name}: 0 entries (feed may be empty or gated)")
        return []

    new_ids = []
    filtered_out = 0

    for entry in feed.entries:
        headline = entry.get("title", "").strip()
        content  = entry.get("summary", entry.get("description", "")).strip()
        url      = entry.get("link", "")
        fp       = _article_fingerprint(headline, source_name)

        if not headline:
            continue

        # ── Relevance gate ────────────────────────────────────────────────
        if not _passes_relevance_gate(headline, content, strict=strict_filter):
            filtered_out += 1
            continue

        if _is_duplicate(fp, session):
            continue

        # ── Safe date parsing ─────────────────────────────────────────────
        pub = entry.get("published_parsed")
        ts  = datetime.utcnow()
        if pub:
            try:
                # struct_time elements: (year, month, day, hour, min, sec, ...)
                # Guard against invalid values (month=0, day=0, etc.)
                year, month, day, hour, minute, second = pub[:6]
                month  = max(1, min(12, month))
                day    = max(1, min(31, day))
                hour   = max(0, min(23, hour))
                minute = max(0, min(59, minute))
                second = max(0, min(59, second))
                ts = datetime(year, month, day, hour, minute, second)
            except Exception:
                ts = datetime.utcnow()

        # ── Freshness gate ────────────────────────────────────────────────
        # Reject articles published more than MAX_ARTICLE_AGE_HOURS ago.
        # Stale articles no longer move markets and add noise to event clusters.
        age_hours = (datetime.utcnow() - ts).total_seconds() / 3600.0
        if age_hours > MAX_ARTICLE_AGE_HOURS:
            filtered_out += 1
            continue

        article = db.Article(
            source=source_name,
            url=fp,
            headline=headline,
            content=content,
            timestamp=ts,
            is_primary_source=is_primary,
        )
        session.add(article)
        session.flush()
        new_ids.append(article.id)

        # Classify article by event type
        try:
            from services.company_event_classifier import classify_article
            article.event_class = classify_article(
                article.headline or "",
                article.content or "",
            )
        except Exception as _ce:
            pass  # non-critical — event_class stays at default

        # ── SEC EDGAR deep filing analysis ────────────────────────────────
        # Only run for EDGAR sources — fetches the actual filing document and
        # enriches the article content with a structured analysis block.
        if "SEC EDGAR" in source_name or "sec.gov" in feed_url:
            try:
                from services.sec_filing_analyzer import analyze_filing
                analysis = analyze_filing(entry, session)
                if analysis and analysis.material_score > 0.3:
                    article = session.get(db.Article, article.id)
                    if article:
                        direction_label = (
                            "bullish" if analysis.impact_direction > 0
                            else "bearish" if analysis.impact_direction < 0
                            else "neutral"
                        )
                        disclosures_block = "\n".join(
                            f"- {s}" for s in analysis.key_disclosures[:5]
                        )
                        guidance_line = ""
                        if analysis.guidance_signals.get("raised"):
                            guidance_line = "Guidance: RAISED\n"
                        elif analysis.guidance_signals.get("lowered"):
                            guidance_line = "Guidance: LOWERED\n"
                        risk_line = ""
                        if analysis.risk_flags:
                            risk_line = (
                                "Risk Flags: "
                                + "; ".join(analysis.risk_flags[:3])
                                + "\n"
                            )
                        article.content = (
                            f"[SEC FILING ANALYSIS]\n"
                            f"Company: {analysis.company_name} ({analysis.ticker})\n"
                            f"Form: {analysis.form_type} | Items: {', '.join(analysis.items_found)}\n"
                            f"Impact: {analysis.primary_event_class} | Direction: {direction_label}\n\n"
                            f"Summary: {analysis.analysis_summary}\n\n"
                            f"Key Disclosures:\n{disclosures_block}\n\n"
                            + guidance_line
                            + risk_line
                        )
                        session.commit()
            except Exception as sec_e:
                pass  # non-critical — article is still stored without analysis

        try:
            _post_process(article.id, session)
        except Exception as e:
            # Roll back the failed article's transaction so the session stays usable
            try:
                session.rollback()
            except Exception:
                pass
            print(f"  Post-process error ({source_name} / {headline[:40]}): {e}")
            continue   # skip to next article, don't break the whole feed

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"  Commit error ({source_name}): {e}")
    print(
        f"  {source_name}: {len(new_ids)} saved"
        + (f", {filtered_out} filtered" if filtered_out else "")
    )
    return new_ids


def ingest_all_feeds(session: Session) -> dict:
    """
    Pull all feeds one by one.
    If one feed fails, the others continue running.
    Returns a summary dict of {source_name: article_count}.
    """
    print("\n=== Starting feed ingestion ===")
    results = {}
    for feed in DEFAULT_FEEDS:
        try:
            ids = ingest_rss_feed(
                feed["url"],
                feed["name"],
                session,
                feed.get("is_primary", 0),
                strict_filter=feed.get("strict_filter", False),
            )
            results[feed["name"]] = len(ids)
        except Exception as e:
            print(f"  SKIPPED {feed['name']}: {e}")
            results[feed["name"]] = 0
    total = sum(results.values())
    print(f"=== Ingestion complete: {total} new articles across {len(DEFAULT_FEEDS)} feeds ===\n")

    # Invalidate the stock recommender cache so the next /stocks/emerging call
    # scans the freshly ingested articles rather than serving stale results.
    if total > 0:
        try:
            from services.stock_recommender import invalidate_cache
            invalidate_cache()
        except Exception:
            pass

    return results


def ingest_manual_article(
    headline: str,
    content: str,
    source: str,
    session: Session,
    url: Optional[str] = None,
    is_primary: int = 0,
    has_official_confirm: int = 0,
) -> db.Article:
    """
    Manually submit an article (e.g. from the API POST endpoint or tests).
    Manual submissions bypass the relevance gate — the user is assumed to
    have already curated the article.
    """
    fp = _article_fingerprint(headline, source)
    if _is_duplicate(fp, session):
        return session.query(db.Article).filter_by(url=fp).first()

    article = db.Article(
        source=source,
        url=url or fp,
        headline=headline,
        content=content,
        is_primary_source=is_primary,
        has_official_confirm=has_official_confirm,
    )
    session.add(article)
    session.flush()
    _post_process(article.id, session)
    session.commit()
    session.refresh(article)
    return article
