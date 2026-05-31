"""
services/stock_discovery.py
-----------------------------
Discovers new stock tickers mentioned in article text beyond the
hardcoded universe. Uses regex-based NER + yfinance validation.

Pipeline:
  1. Extract candidate company/ticker mentions from article text
  2. Validate each candidate via yfinance (real price + market cap)
  3. Score confidence (market cap tier + event credibility + format)
  4. Store in DiscoveredTicker table
  5. Auto-promote to main Ticker universe if confidence >= 0.75
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session
import database as db
from services.universe_manager import check_and_promote_discovered

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

AUTO_PROMOTE_CONFIDENCE = 0.75
MIN_SHOW_CONFIDENCE     = 0.45
MIN_MARKET_CAP          = 500_000_000   # $500M

# Company name + legal suffix pattern
_COMPANY_RE = re.compile(
    r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+'
    r'(?:Inc\.|Corp\.|Ltd\.|plc|N\.V\.|S\.A\.|AG|GmbH|'
    r'Holdings?|Technologies?|Systems?|Semiconductor|'
    r'Microsystems?|Solutions?|Laboratories?|Industries?|'
    r'Enterprises?|Group|Capital|Photonics?|Electronics?)\b'
)

# Ticker in parentheses immediately following a (likely) company name token.
# We require a parenthesised ALL-CAPS ticker preceded by a capitalised word —
# this avoids matching standalone acronyms like "UCLA," or "OPEC."
_PAREN_TICKER_RE = re.compile(
    r'(?:[A-Z][A-Za-z][A-Za-z\.\-&]*|\.)\s*'
    r'\((?:NYSE|NASDAQ|LSE|TSX|HKEX|TYO|TSE|SHA|SHE)?:?\s*([A-Z]{1,5})\)'
)

# Explicit exchange prefix: "NYSE: TICK" / "NASDAQ:TICK"
_EXCHANGE_RE = re.compile(
    r'\b(?:NYSE|NASDAQ|LSE|TSX|HKEX|TYO|TSE|SHA|SHE):\s*([A-Z]{1,5})\b'
)

# Cashtag form used on financial sites/socials: "$NVDA"
_CASHTAG_RE = re.compile(r'\$([A-Z]{1,5})\b')

_NOISE = {
    "The United", "The Federal", "The European", "The American",
    "North Korea", "South Korea", "New York", "Hong Kong",
    "Silicon Valley", "United States", "White House", "Wall Street",
    "Main Street", "Federal Reserve", "European Union", "United Nations",
}

# Acronyms that look like tickers but are organisations, products, weapons,
# economic concepts, country codes, etc. yfinance will happily 404 on these
# (or worse, return a stale low-volume foreign listing). Filter at extraction
# time so we never call yfinance for them.
_ACRONYM_BLOCKLIST = {
    # Government / military / political / NGO acronyms
    "USA", "USD", "UK", "EU", "UN", "NATO", "OPEC", "WTO", "IMF", "WHO",
    "CIA", "FBI", "NSA", "DOJ", "DOD", "DOE", "EPA", "FDA", "FTC", "SEC",
    "FCC", "CDC", "IRS", "DHS", "ICE", "TSA", "GSA", "NIH", "USDA", "FAA",
    "CBP", "ATF", "OFAC", "BIS", "CFIUS", "CFPB", "GAO", "OMB", "USAID",
    "PRC", "DPRK", "ROK", "ROC", "IRGC", "PLA", "IDF", "RUF", "FSA",
    "ISIS", "ISIL", "PKK", "PFLP", "RNA", "DNA",
    "GDP", "GNP", "CPI", "PPI", "PMI", "M1", "M2", "PCE", "FOMC",
    "ECB", "BOE", "BOJ", "PBOC", "RBI", "RBNZ", "RBA", "BIS", "BOC",
    # Universities / NGOs / press orgs
    "MIT", "NYU", "USC", "UCLA", "UCSD", "UCSB", "UCSF", "UCB", "UCI",
    "USC", "BYU", "TCU", "SMU", "VCU", "BU", "GW", "FSU", "UF", "UM",
    "AHRC", "NDIS", "SPLC", "ACLU", "NRA", "NAACP", "AARP", "RNC", "DNC",
    "GOP", "PAC", "SuperPAC",
    # Cartels, terror groups, military programs (will not have tickers)
    "CJNG", "MS13", "FARC", "ELN", "AQ", "AQAP", "AQIM",
    # Weapons / military programs / standards
    "JDAM", "MOSA", "JADC2", "JEDI", "PNT", "AESA", "SAM", "ATGM",
    "ICBM", "MIRV", "SLBM", "ASW", "AWACS", "RPV", "UAV", "UAS",
    "IED", "VBIED", "EW",
    # Common english and tech acronyms that yfinance tries to resolve
    "AI", "ML", "NLP", "AR", "VR", "XR", "IT", "HR", "PR", "QA",
    "PDF", "HTML", "CSS", "JS", "API", "SDK", "GUI", "CLI", "OS",
    "URL", "DNS", "IP", "TCP", "UDP", "SSL", "TLS", "VPN", "CDN",
    "FAQ", "TOS", "EULA", "CEO", "CFO", "CTO", "COO", "CMO", "CIO",
    "CSO", "EVP", "SVP", "VP", "PM", "PR", "BD", "RD", "QC",
    # Misc commonly mis-extracted
    "SUSE", "RHEL", "RHEL", "MSCIO", "SPR", "BRICS",
    # File / code formats
    "JSON", "XML", "YAML", "YML", "CSV", "TSV", "JPG", "PNG", "GIF",
    # Time / date abbreviations
    "EST", "EDT", "PST", "PDT", "GMT", "UTC", "CST", "MST",
    # Other
    "SAR", "SAS", "BAR", "AAR", "RIP",
}


# ── Entity extraction ────────────────────────────────────────────────────────

def extract_company_mentions(text: str) -> List[str]:
    """
    Extract candidate company names / tickers from raw article text.

    The bare 2-5 letter pattern is intentionally NOT used here — too many
    organisation/military/economic acronyms (UCLA, OPEC, JDAM, GDP, …) match
    that shape and waste yfinance lookups on guaranteed 404s. We only accept:
      • Capitalised company names with a recognised legal suffix
      • Tickers wrapped in parentheses *after* a likely company name token
      • Tickers with an explicit exchange prefix (NYSE: TICK)
      • Cashtags ($NVDA)
    Each candidate is filtered against `_ACRONYM_BLOCKLIST`.
    """
    candidates: set = set()

    for m in _COMPANY_RE.finditer(text):
        name = m.group(0).strip()
        if name not in _NOISE and len(name) > 4:
            candidates.add(name)

    for m in _PAREN_TICKER_RE.finditer(text):
        sym = m.group(1)
        if sym in _ACRONYM_BLOCKLIST:
            continue
        if len(sym) < 2:        # skip single-letter false positives
            continue
        candidates.add(sym)

    for m in _EXCHANGE_RE.finditer(text):
        sym = m.group(1)
        if sym in _ACRONYM_BLOCKLIST or len(sym) < 2:
            continue
        candidates.add(sym)

    for m in _CASHTAG_RE.finditer(text):
        sym = m.group(1)
        if sym in _ACRONYM_BLOCKLIST or len(sym) < 2:
            continue
        candidates.add(sym)

    return list(candidates)


# ── yfinance validation ───────────────────────────────────────────────────────

def validate_ticker_yfinance(symbol_or_name: str) -> Optional[Dict]:
    """
    Validate a symbol or company name with yfinance.
    Returns {ticker, company_name, sector, market_cap} or None.
    """
    if symbol_or_name in _ACRONYM_BLOCKLIST:
        return None
    try:
        import yfinance as yf

        # Direct ticker lookup
        if re.match(r'^[A-Z]{2,5}$', symbol_or_name):
            info = yf.Ticker(symbol_or_name).info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if price:
                mcap = info.get("marketCap", 0) or 0
                if mcap < MIN_MARKET_CAP:
                    return None
                return {
                    "ticker":       symbol_or_name,
                    "company_name": info.get("longName") or info.get("shortName", symbol_or_name),
                    "sector":       info.get("sector", "Unknown"),
                    "market_cap":   mcap,
                }

        # Company name search
        results = yf.Search(symbol_or_name, max_results=3).quotes
        if not results:
            return None
        for r in results:
            ticker = r.get("symbol", "")
            if not ticker or len(ticker) > 6:
                continue
            info = yf.Ticker(ticker).info
            mcap = info.get("marketCap", 0) or 0
            if mcap >= MIN_MARKET_CAP:
                return {
                    "ticker":       ticker,
                    "company_name": info.get("longName") or r.get("longname", ticker),
                    "sector":       info.get("sector", "Unknown"),
                    "market_cap":   mcap,
                }
        return None
    except Exception as e:
        logger.debug("yfinance validation failed for %s: %s", symbol_or_name, e)
        return None


# ── Confidence scoring ────────────────────────────────────────────────────────

def _compute_confidence(ticker: str, market_cap: float, event: db.Event) -> float:
    score = 0.0
    # Market cap tier
    if   market_cap >= 10_000_000_000:  score += 0.35
    elif market_cap >= 1_000_000_000:   score += 0.25
    else:                               score += 0.10
    # Event credibility
    score += (event.credibility_score or 0.5) * 0.30
    # Ticker format (US-style: 2-5 uppercase letters)
    if re.match(r'^[A-Z]{2,5}$', ticker):
        score += 0.20
    # Bonus for known sector keywords in event
    event_text = f"{event.title} {event.summary or ''}".lower()
    if any(kw in event_text for kw in ["semiconductor", "chip", "ai", "defense", "energy"]):
        score += 0.15
    return min(score, 1.0)


# ── Promotion ─────────────────────────────────────────────────────────────────

def _promote_to_universe(disc: db.DiscoveredTicker, session: Session):
    """Add a high-confidence ticker to the main Ticker universe."""
    if not session.get(db.Ticker, disc.ticker):
        session.add(db.Ticker(
            ticker=disc.ticker,
            company_name=disc.company_name,
            sector=disc.sector or "Unknown",
        ))
    disc.is_promoted = True
    logger.info("Promoted discovered ticker %s (%s) to main universe", disc.ticker, disc.company_name)


# ── Main discovery function ────────────────────────────────────────────────────

def discover_tickers_from_event(event_id: int, session: Session) -> List[str]:
    """
    Extract + validate new ticker mentions from an event's articles.
    Stores validated discoveries; auto-promotes high-confidence ones.
    Returns list of newly discovered ticker symbols.
    """
    event = session.get(db.Event, event_id)
    if not event:
        return []

    articles = session.query(db.Article).filter_by(event_id=event_id).all()
    all_text = " ".join(f"{a.headline} {a.content or ''}" for a in articles)
    if not all_text.strip():
        all_text = f"{event.title} {event.summary or ''}"

    candidates = extract_company_mentions(all_text)
    if not candidates:
        return []

    # Build sets of tickers already known
    existing = {r.ticker for r in session.query(db.Ticker.ticker).all()}
    try:
        already_disc = {r.ticker for r in session.query(db.DiscoveredTicker.ticker).all()}
    except Exception:
        already_disc = set()

    new_discoveries: List[str] = []

    for candidate in candidates[:15]:  # cap per event
        if candidate in existing or candidate in already_disc:
            continue

        validated = validate_ticker_yfinance(candidate)
        if not validated:
            continue

        ticker = validated["ticker"]
        if ticker in existing or ticker in already_disc:
            continue

        confidence = _compute_confidence(ticker, validated["market_cap"], event)
        if confidence < MIN_SHOW_CONFIDENCE:
            continue

        try:
            disc = db.DiscoveredTicker(
                ticker=ticker,
                company_name=validated["company_name"],
                sector=validated["sector"],
                market_cap=validated.get("market_cap"),
                first_seen_at=datetime.utcnow(),
                discovery_event_id=event_id,
                confidence_score=confidence,
                article_count=1,
                is_promoted=False,
            )
            session.add(disc)
            session.flush()
            already_disc.add(ticker)
            new_discoveries.append(ticker)

            if confidence >= AUTO_PROMOTE_CONFIDENCE:
                _promote_to_universe(disc, session)

            logger.info(
                "Discovered %s (%s) conf=%.2f promoted=%s",
                ticker, validated["company_name"], confidence, confidence >= AUTO_PROMOTE_CONFIDENCE
            )
        except Exception as e:
            logger.debug("Discovery store failed for %s: %s", ticker, e)
            session.rollback()

    if new_discoveries:
        session.commit()
        for ticker_symbol in new_discoveries:
            try:
                check_and_promote_discovered(ticker_symbol, session)
            except Exception as e:
                logger.debug("check_and_promote_discovered failed for %s: %s", ticker_symbol, e)

    return new_discoveries


def get_discovered_tickers(
    session: Session,
    min_confidence: float = MIN_SHOW_CONFIDENCE,
    limit: int = 100,
) -> List[db.DiscoveredTicker]:
    """Return discovered tickers above confidence threshold, newest first."""
    try:
        return (
            session.query(db.DiscoveredTicker)
            .filter(db.DiscoveredTicker.confidence_score >= min_confidence)
            .order_by(db.DiscoveredTicker.confidence_score.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        return []
