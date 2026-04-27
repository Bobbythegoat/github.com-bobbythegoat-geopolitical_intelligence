"""
company_event_classifier.py

Lightweight, rule-based article classifier for the geopolitical intelligence system.
Classifies articles into one of 8 event types using keyword pattern matching only.
No LLM, no external calls — fast and deterministic.

Example usage:
    # classify_article("NVIDIA sued over GPU patent", "") == "legal"
    # classify_article("ASML export license revoked by Dutch government", "") == "regulatory"
    # classify_article("AMD reports Q3 earnings beat", "") == "earnings"
    # classify_article("Taiwan Strait tensions escalate", "") == "geopolitical"
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority order (index 0 = highest priority)
# ---------------------------------------------------------------------------

PRIORITY_ORDER = [
    "regulatory",
    "legal",
    "filing",
    "ma",
    "earnings",
    "executive",
    "product_launch",
    "geopolitical",
]

# ---------------------------------------------------------------------------
# Keywords per class
# ---------------------------------------------------------------------------

CLASS_KEYWORDS = {
    "regulatory": [
        "export control",
        "export ban",
        "export license",
        "ban",
        "sanction",
        "sanctions",
        "restriction",
        "restrictions",
        "cfius",
        "fcc",
        "fda approval",
        "blocked",
        "prohibited",
        "license revoked",
    ],
    "legal": [
        "lawsuit",
        "lawsuits",
        "sued",
        "doj",
        "sec investigation",
        "fraud",
        "settlement",
        "antitrust",
        "indictment",
        "class action",
        "litigation",
        "charges filed",
    ],
    "filing": [
        "8-k",
        "10-k",
        "10-q",
        "proxy",
        "def 14a",
        "form 4",
        "insider buying",
        "insider selling",
        "annual report",
        "quarterly report",
        "sec filing",
    ],
    "ma": [
        "acquires",
        "acquired",
        "merger",
        "acquisition",
        "takeover",
        "buyout",
        "buys for",
        "deal closed",
        "to buy",
        "to acquire",
    ],
    "earnings": [
        "earnings",
        "eps",
        "beat",
        "miss",
        "quarterly results",
        "guidance",
        "raised guidance",
        "lowered guidance",
        "outlook",
        "revenue beat",
        "revenue miss",
        "revenue",
    ],
    "executive": [
        "ceo",
        "cfo",
        "cto",
        "coo",
        "appointed",
        "resigned",
        "steps down",
        "named president",
        "promoted to",
        "new chief",
        "leadership change",
    ],
    "product_launch": [
        "announces",
        "unveiled",
        "unveils",
        "launches",
        "launched",
        "new chip",
        "new model",
        "next-gen",
        "collaboration",
        "new product",
        "software update",
        "firmware update",
    ],
    "geopolitical": [
        "war",
        "conflict",
        "invasion",
        "missile",
        "nuclear",
        "nato",
        "military",
        "tariff",
        "tariffs",
        "trade war",
        "taiwan",
        "china",
        "russia",
        "ukraine",
        "sanctions",
    ],
}

# ---------------------------------------------------------------------------
# Human-readable descriptions
# ---------------------------------------------------------------------------

EVENT_CLASS_DESCRIPTIONS = {
    "regulatory": (
        "Government or regulatory body action affecting company operations, "
        "including export controls, sanctions, bans, and license revocations."
    ),
    "legal": (
        "Legal proceedings involving the company, including lawsuits, DOJ/SEC "
        "investigations, fraud allegations, settlements, and class actions."
    ),
    "filing": (
        "Official regulatory filings submitted to the SEC or equivalent body, "
        "including 8-K, 10-K, 10-Q, proxy statements, and insider transactions."
    ),
    "ma": (
        "Mergers, acquisitions, takeovers, buyouts, and other corporate "
        "combination or deal activity."
    ),
    "earnings": (
        "Quarterly or annual financial results, EPS beats/misses, guidance "
        "changes, revenue performance, and forward outlook updates."
    ),
    "executive": (
        "Leadership changes at the C-suite level, including appointments, "
        "resignations, promotions, and other named executive transitions."
    ),
    "product_launch": (
        "New product announcements, unveilings, launches, next-generation "
        "hardware or software releases, and strategic partnerships."
    ),
    "geopolitical": (
        "Macro geopolitical developments including wars, trade conflicts, "
        "tariffs, military events, and country-level political risk."
    ),
}


# ---------------------------------------------------------------------------
# Keyword matching helper
# ---------------------------------------------------------------------------

def _match(keyword: str, text: str) -> bool:
    """Match a keyword against text using word boundaries for short keywords."""
    if len(keyword) <= 4:
        return bool(re.search(r'\b' + re.escape(keyword) + r'\b', text))
    return keyword in text


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------

def classify_article(headline: str, content: str = "") -> str:
    """
    Classify an article into one of 8 event classes.

    Uses the first 500 chars of content combined with the full headline.
    Returns the FIRST matching class in priority order (1 -> 8).

    Returns one of:
        regulatory | legal | filing | ma | earnings |
        executive | product_launch | geopolitical

    Default (no match): 'geopolitical'

    Examples:
        >>> classify_article("NVIDIA sued over GPU patent", "")
        'legal'
        >>> classify_article("ASML export license revoked by Dutch government", "")
        'regulatory'
        >>> classify_article("AMD reports Q3 earnings beat", "")
        'earnings'
        >>> classify_article("Taiwan Strait tensions escalate", "")
        'geopolitical'
    """
    try:
        # Build the text window: full headline + first 500 chars of content
        text = f"{headline} {content[:500]}".lower()

        for event_class in PRIORITY_ORDER:
            keywords = CLASS_KEYWORDS[event_class]
            for keyword in keywords:
                if _match(keyword, text):
                    return event_class

        # Default fallback
        return "geopolitical"

    except Exception as e:
        logger.debug("classify_article failed: %s", e)
        return "geopolitical"


def classify_batch(articles: list) -> list:
    """
    Classify a batch of article dicts.

    Each dict must have a 'headline' key and optionally a 'content' key.

    Returns a list of (article, event_class) tuples in the same order as
    the input list.

    Example:
        articles = [
            {"headline": "NVIDIA sued over GPU patent", "content": ""},
            {"headline": "Taiwan tensions escalate", "content": ""},
        ]
        results = classify_batch(articles)
        # [({'headline': 'NVIDIA sued...'}, 'legal'),
        #  ({'headline': 'Taiwan tensions...'}, 'geopolitical')]
    """
    results = []
    for article in articles:
        headline = article.get("headline", "")
        content = article.get("content", "")
        event_class = classify_article(headline, content)
        results.append((article, event_class))
    return results


def get_event_class_description(event_class: str) -> str:
    """
    Return a human-readable description of an event class.

    Returns an empty string if the event class is not recognised.
    """
    return EVENT_CLASS_DESCRIPTIONS.get(event_class, "")
