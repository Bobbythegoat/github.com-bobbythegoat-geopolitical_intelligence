"""
services/stock_recommender.py
------------------------------
Real-time stock recommendation engine driven by live news events.

Instead of relying solely on the 30 hardcoded tickers, this module:
  1. Scans recent articles for company/ticker mentions (via regex NER)
  2. Validates each candidate via yfinance (real price + market cap gate)
  3. Scores each by: event credibility × mention frequency × confidence
  4. Returns a ranked list of emerging opportunities outside the core universe

Results are cached for CACHE_TTL seconds so repeated frontend polls don't
hammer yfinance.  The cache is invalidated automatically when new articles
are ingested (call invalidate_cache() from the ingestion pipeline).
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

import database as db
from services.stock_discovery import (
    extract_company_mentions,
    validate_ticker_yfinance,
    _compute_confidence,
    MIN_SHOW_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_TTL        = 300    # seconds — 5-minute result cache
MAX_ARTICLES     = 200    # articles to scan per call
MAX_VALIDATE     = 50     # max yfinance validation calls per refresh
MIN_MENTIONS     = 1      # minimum article mentions to consider
MAX_RESULTS      = 30     # maximum returned recommendations
DEFAULT_HOURS    = 24     # hours of articles to scan (default)

# ── Module-level cache ────────────────────────────────────────────────────────

_cache: dict = {}   # {"results": [...], "expires_at": float, "hours": int}


def invalidate_cache() -> None:
    """Call this from the ingestion pipeline when new articles arrive."""
    _cache.clear()
    logger.debug("Stock recommender cache invalidated")


def _is_cache_valid(hours: int) -> bool:
    return (
        _cache.get("hours") == hours
        and _cache.get("expires_at", 0) > time.time()
        and _cache.get("results") is not None
    )


# ── Main recommendation function ──────────────────────────────────────────────

def get_emerging_recommendations(
    session: Session,
    hours: int = DEFAULT_HOURS,
    min_confidence: float = MIN_SHOW_CONFIDENCE,
    limit: int = MAX_RESULTS,
    force_refresh: bool = False,
    exclude_extended: bool = False,
) -> List[Dict]:
    """
    Scan recent articles for stock mentions and return scored recommendations.

    Returns a list of dicts:
    {
      "ticker":        str,
      "company_name":  str,
      "sector":        str,
      "market_cap":    float,
      "score":         float,     # composite recommendation score 0–1
      "confidence":    float,     # yfinance validation confidence
      "mentions":      int,       # article mention count in the window
      "event_ids":     List[int], # event IDs driving the mention
      "reason":        str,       # one-line human explanation
      "direction":     str,       # "bullish" | "bearish" | "mixed" | "neutral"
      "discovered_at": str,       # ISO UTC timestamp
    }
    """

    if not force_refresh and _is_cache_valid(hours):
        cached = _cache["results"]
        return _apply_filters(cached, min_confidence, exclude_extended, limit)

    since = datetime.utcnow() - timedelta(hours=hours)

    # ── Step 1: Fetch recent articles ─────────────────────────────────────────
    articles = (
        session.query(db.Article)
        .filter(db.Article.timestamp >= since)
        .order_by(db.Article.timestamp.desc())
        .limit(MAX_ARTICLES)
        .all()
    )

    if not articles:
        _cache.update({"results": [], "expires_at": time.time() + CACHE_TTL, "hours": hours})
        return []

    # ── Step 2: Pre-load events and impacts for article batch ─────────────────
    event_ids = {a.event_id for a in articles if a.event_id}
    events_map: Dict[int, db.Event] = {}
    impact_map: Dict[int, List[db.EventTickerImpact]] = defaultdict(list)

    if event_ids:
        for ev in session.query(db.Event).filter(db.Event.event_id.in_(event_ids)).all():
            events_map[ev.event_id] = ev
        for imp in (
            session.query(db.EventTickerImpact)
            .filter(db.EventTickerImpact.event_id.in_(event_ids))
            .all()
        ):
            impact_map[imp.event_id].append(imp)

    # Tickers already in the tracked universe (skip them — they show in /stocks/)
    known_tickers = {r.ticker for r in session.query(db.Ticker.ticker).all()}

    # ── Step 3: Extract and aggregate mentions ────────────────────────────────
    # mention_data maps raw candidate string → aggregated signal data
    mention_data: Dict[str, Dict] = defaultdict(lambda: {
        "articles":          [],    # article IDs where this candidate appears
        "events":            set(), # event IDs linked to those articles
        "raw_names":         set(), # alternate name forms seen in text
        "total_credibility": 0.0,
        "total_impact":      0.0,
        "directions":        [],    # "bullish" or "bearish" per impact
    })

    for article in articles:
        text = f"{article.headline} {article.content or ''}".strip()
        if not text:
            continue

        mentions = extract_company_mentions(text)
        if not mentions:
            continue

        event   = events_map.get(article.event_id) if article.event_id else None
        cred    = float(event.credibility_score) if (event and event.credibility_score is not None) else 0.3
        e_imps  = impact_map.get(article.event_id, []) if article.event_id else []

        for raw_name in mentions:
            # Canonical key: use the ticker form as-is, or title-case company name
            key = raw_name.upper() if (len(raw_name) <= 5 and raw_name.isupper()) else raw_name
            d   = mention_data[key]
            d["articles"].append(article.id)
            d["raw_names"].add(raw_name)
            if event:
                d["events"].add(event.event_id)
            d["total_credibility"] += cred

            # Collect directional signal from causal impacts
            for imp in e_imps:
                if imp.impact_score is not None:
                    d["total_impact"] += float(imp.impact_score)
                    d["directions"].append(
                        "bullish" if imp.impact_score > 0.10 else
                        "bearish" if imp.impact_score < -0.10 else None
                    )

    if not mention_data:
        _cache.update({"results": [], "expires_at": time.time() + CACHE_TTL, "hours": hours})
        return []

    # ── Step 4: Sort candidates by signal strength, validate via yfinance ─────
    sorted_candidates = sorted(
        mention_data.items(),
        key=lambda x: (len(x[1]["articles"]), x[1]["total_credibility"]),
        reverse=True,
    )

    results: List[Dict] = []
    validated_count = 0

    for raw_key, data in sorted_candidates:
        if validated_count >= MAX_VALIDATE:
            break
        if len(data["articles"]) < MIN_MENTIONS:
            continue
        if raw_key in known_tickers:
            continue

        # Try the canonical key first, then alternate name forms
        validated = validate_ticker_yfinance(raw_key)
        if not validated:
            for alt in sorted(data["raw_names"], key=len)[:3]:
                if alt == raw_key:
                    continue
                validated = validate_ticker_yfinance(alt)
                if validated:
                    break

        if not validated:
            continue
        validated_count += 1

        ticker = validated["ticker"]
        if ticker in known_tickers:
            continue  # promoted to main universe already

        # ── Confidence score ──────────────────────────────────────────────────
        best_event_id = max(data["events"]) if data["events"] else None
        best_event    = events_map.get(best_event_id) if best_event_id else None

        if best_event:
            confidence = _compute_confidence(ticker, validated["market_cap"], best_event)
        else:
            mcap = validated["market_cap"]
            confidence  = 0.35 if mcap >= 10_000_000_000 else 0.25 if mcap >= 1_000_000_000 else 0.10
            confidence += 0.20 if (len(ticker) <= 5 and ticker.isupper()) else 0.0
            confidence  = min(confidence, 1.0)

        if confidence < min_confidence:
            continue

        # ── Composite recommendation score ────────────────────────────────────
        #   40% average event credibility  — are the events driving this mention trustworthy?
        #   30% normalised mention count   — is it mentioned repeatedly?
        #   20% discovery confidence       — is the ticker validation solid?
        #   10% absolute causal impact     — is there a measurable impact signal?
        mentions_count  = len(data["articles"])
        avg_cred        = data["total_credibility"] / max(mentions_count, 1)
        mention_norm    = min(mentions_count / 10.0, 1.0)
        avg_impact_abs  = abs(data["total_impact"]) / max(mentions_count, 1)
        base_score = (
            avg_cred       * 0.40
            + mention_norm * 0.30
            + confidence   * 0.20
            + avg_impact_abs * 0.10
        )

        # ── Price positioning: down-rank names that already exploded ──────────
        # The news signal can be identical for a fresh name and one that already
        # ran 40% — but the fresh-entry asymmetry is not.  We fetch live price
        # context, classify how extended the stock is, and apply a penalty so
        # already-moved names sink in the ranking.  We do NOT hide them: the
        # label + the underlying numbers travel with the row so the user can
        # make their own call and research further.
        try:
            from services.price_positioning import classify, status_label
            from services.price_context import fetch_price_context
            pos = classify(fetch_price_context(ticker))
        except Exception:
            pos = {
                "extension_score": 0.0, "freshness_score": 1.0,
                "price_status": "unknown", "is_extended": False,
                "penalty_mult": 1.0, "rsi_14": None,
                "pct_from_52w_high": None, "momentum_30d": None,
                "note": "Price data unavailable — positioning not assessed.",
            }

        composite_score = round(base_score * pos.get("penalty_mult", 1.0), 4)

        # ── Direction label ───────────────────────────────────────────────────
        valid_dirs = [d for d in data["directions"] if d is not None]
        bull = valid_dirs.count("bullish")
        bear = valid_dirs.count("bearish")
        if bull + bear == 0:
            direction = "neutral"
        elif bull > bear * 1.5:
            direction = "bullish"
        elif bear > bull * 1.5:
            direction = "bearish"
        else:
            direction = "mixed"

        # ── Human-readable reason ─────────────────────────────────────────────
        event_note = (
            f", {len(data['events'])} linked event{'s' if len(data['events']) > 1 else ''}"
            if data["events"] else ""
        )
        reason = (
            f"{mentions_count} mention{'s' if mentions_count > 1 else ''} in the last {hours}h"
            f"{event_note}, avg credibility {avg_cred:.0%}."
        )

        results.append({
            "ticker":        ticker,
            "company_name":  validated["company_name"],
            "sector":        validated["sector"],
            "market_cap":    validated["market_cap"],
            "score":         composite_score,
            "base_score":    round(base_score, 4),   # before price-extension penalty
            "confidence":    round(confidence, 4),
            "mentions":      mentions_count,
            "event_ids":     sorted(data["events"]),
            "reason":        reason,
            "direction":     direction,
            # ── Price positioning (issue #1: avoid already-exploded names) ────
            "price_status":      pos.get("price_status", "unknown"),
            "price_status_label": status_label(pos["price_status"]) if pos.get("price_status") and pos.get("price_status") != "unknown" else "Not assessed",
            "extension_score":   pos.get("extension_score"),
            "freshness_score":   pos.get("freshness_score"),
            "is_extended":       pos.get("is_extended", False),
            "rsi_14":            pos.get("rsi_14"),
            "pct_from_52w_high": pos.get("pct_from_52w_high"),
            "momentum_30d":      pos.get("momentum_30d"),
            "entry_note":        pos.get("note", ""),
            "discovered_at": datetime.utcnow().isoformat(),
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    # Store in cache
    _cache.update({
        "results":    results,
        "expires_at": time.time() + CACHE_TTL,
        "hours":      hours,
    })

    logger.info(
        "Stock recommender: scanned %d articles, validated %d candidates, returning %d results",
        len(articles), validated_count, len(results),
    )
    return _apply_filters(results, min_confidence, exclude_extended, limit)


def _apply_filters(
    rows: List[Dict],
    min_confidence: float,
    exclude_extended: bool,
    limit: int,
) -> List[Dict]:
    """Filter cached/fresh recommendation rows by confidence and (optionally)
    drop overextended names, preserving the score-descending order."""
    out = [r for r in rows if r.get("confidence", 0) >= min_confidence]
    if exclude_extended:
        # Only drop the clearly already-exploded names (overextended); keep the
        # merely 'extended' ones visible-but-labelled.
        out = [r for r in out if r.get("price_status") != "overextended"]
    return out[:limit]
