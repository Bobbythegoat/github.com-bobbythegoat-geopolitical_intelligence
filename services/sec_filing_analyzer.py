"""
services/sec_filing_analyzer.py
---------------------------------
Deep SEC filing analyzer.

Pipeline:
  1. Parse EDGAR RSS entry to extract CIK, accession number, filing URL
  2. Fetch the filing index page (EDGAR ATOM feed → index URL)
  3. Download the primary document (8-K, 10-K, 10-Q, etc.)
  4. Parse filing by item type — each Item maps to a specific event class
  5. Extract company name, ticker, material disclosures
  6. Classify economic impact and generate causal analysis
  7. Return structured FilingAnalysis for ingestion pipeline

Supports:
  - 8-K: Material events (all item types mapped)
  - 10-K: Annual report highlights (risk factors, guidance)
  - 10-Q: Quarterly report highlights

Scales to 6000+ stocks via SEC EDGAR CIK→ticker registry.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC 8-K Item Classification
# ---------------------------------------------------------------------------

ITEM_CLASSIFICATIONS: Dict[str, Dict] = {
    # Item 1.01 — Entry into material definitive agreement
    "1.01": {
        "event_class": "merger_acquisition_tech",
        "direction": 0.6,
        "material": True,
        "description": "Material contract or agreement signed",
    },
    # Item 1.02 — Termination of material definitive agreement
    "1.02": {
        "event_class": "merger_acquisition_tech",
        "direction": -0.5,
        "material": True,
        "description": "Material contract terminated",
    },
    # Item 1.03 — Bankruptcy
    "1.03": {
        "event_class": "company_earnings_guidance",
        "direction": -1.0,
        "material": True,
        "description": "Bankruptcy or receivership",
    },
    # Item 2.01 — Completion of acquisition or disposition
    "2.01": {
        "event_class": "merger_acquisition_tech",
        "direction": 0.7,
        "material": True,
        "description": "Acquisition or merger completed",
    },
    # Item 2.02 — Results of operations / earnings
    "2.02": {
        "event_class": "company_earnings_guidance",
        "direction": 0.0,
        "material": True,
        "description": "Earnings results released",
    },
    # Item 2.05 — Costs associated with exit/disposal activities (layoffs)
    "2.05": {
        "event_class": "company_earnings_guidance",
        "direction": -0.4,
        "material": True,
        "description": "Restructuring or layoffs",
    },
    # Item 2.06 — Material impairments
    "2.06": {
        "event_class": "company_earnings_guidance",
        "direction": -0.7,
        "material": True,
        "description": "Material asset impairment",
    },
    # Item 3.01 — Notice of delisting
    "3.01": {
        "event_class": "regulatory_antitrust",
        "direction": -0.9,
        "material": True,
        "description": "Delisting notice",
    },
    # Item 4.01 — Changes in auditor
    "4.01": {
        "event_class": "regulatory_antitrust",
        "direction": -0.3,
        "material": True,
        "description": "Auditor change",
    },
    # Item 4.02 — Non-reliance on prior financial statements
    "4.02": {
        "event_class": "regulatory_antitrust",
        "direction": -0.8,
        "material": True,
        "description": "Financial restatement required",
    },
    # Item 5.01 — Changes in control
    "5.01": {
        "event_class": "merger_acquisition_tech",
        "direction": 0.5,
        "material": True,
        "description": "Change of control",
    },
    # Item 5.02 — Departure/appointment of directors or executive officers
    "5.02": {
        "event_class": "company_earnings_guidance",
        "direction": -0.2,
        "material": False,
        "description": "Executive change",
    },
    # Item 5.03 — Amendments to certificate of incorporation
    "5.03": {
        "event_class": "regulatory_antitrust",
        "direction": 0.0,
        "material": False,
        "description": "Corporate governance change",
    },
    # Item 7.01 — Regulation FD disclosure (voluntary guidance)
    "7.01": {
        "event_class": "company_earnings_guidance",
        "direction": 0.3,
        "material": True,
        "description": "Management guidance or disclosure",
    },
    # Item 8.01 — Other events (catch-all for material events)
    "8.01": {
        "event_class": "company_earnings_guidance",
        "direction": 0.2,
        "material": True,
        "description": "Other material event",
    },
    # Item 9.01 — Financial statements and exhibits
    "9.01": {
        "event_class": "company_earnings_guidance",
        "direction": 0.0,
        "material": False,
        "description": "Financial exhibits attached",
    },
}

# ---------------------------------------------------------------------------
# FilingAnalysis dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilingAnalysis:
    ticker: str
    company_name: str
    cik: str
    form_type: str                        # "8-K", "10-K", "10-Q"
    filed_date: str                       # ISO date string
    accession_number: str
    filing_url: str
    items_found: List[str]                # e.g. ["2.02", "9.01"]
    primary_event_class: str             # highest material item's event_class
    impact_direction: float              # -1.0 to 1.0
    material_score: float                # 0-1, how material this filing is
    key_disclosures: List[str]           # extracted important sentences
    revenue_mentions: List[str]          # sentences mentioning revenue/guidance numbers
    guidance_signals: dict               # {"raised": bool, "lowered": bool, "withdrawn": bool, "keywords": List[str]}
    risk_flags: List[str]                # phrases indicating downside risk
    full_text_snippet: str              # first 2000 chars of filing body
    analysis_summary: str               # generated 2-3 sentence summary


# ---------------------------------------------------------------------------
# CIK → Ticker Registry
# ---------------------------------------------------------------------------

_CIK_TICKER_CACHE: Dict[str, str] = {}    # cik → ticker
_TICKER_CIK_CACHE: Dict[str, str] = {}    # ticker → cik
_REGISTRY_LOADED: bool = False

_SEC_HEADERS = {
    "User-Agent": "GeopoliticalIntelligence/1.0 (contact: augustus.soedarmono@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
}


def load_sec_company_registry(max_companies: int = 6000) -> Dict[str, str]:
    """
    Download SEC EDGAR full company list and build CIK→ticker mapping.
    URL: https://www.sec.gov/files/company_tickers.json
    Returns dict: {cik: ticker}

    This is the foundation for scaling to 6000+ stocks.
    """
    global _CIK_TICKER_CACHE, _TICKER_CIK_CACHE, _REGISTRY_LOADED

    if _REGISTRY_LOADED and _CIK_TICKER_CACHE:
        return dict(_CIK_TICKER_CACHE)

    try:
        import requests

        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # JSON structure: {index: {"cik_str": "0000320193", "ticker": "AAPL", "title": "Apple Inc."}}
        count = 0
        for _idx, record in data.items():
            if count >= max_companies:
                break
            cik_raw = str(record.get("cik_str", "")).strip()
            ticker = str(record.get("ticker", "")).strip().upper()
            title = str(record.get("title", "")).strip()
            if not cik_raw or not ticker:
                continue
            # Normalise CIK to zero-padded 10-digit string
            cik_norm = cik_raw.lstrip("0") or "0"
            _CIK_TICKER_CACHE[cik_norm] = ticker
            _CIK_TICKER_CACHE[cik_raw] = ticker           # also store with leading zeros
            _TICKER_CIK_CACHE[ticker] = cik_norm
            # store company name variant for lookup
            if title:
                _TICKER_CIK_CACHE[title.upper()] = cik_norm
            count += 1

        _REGISTRY_LOADED = True
        logger.info("sec_filing_analyzer: loaded %d company records from SEC registry", count)
        return dict(_CIK_TICKER_CACHE)

    except Exception as exc:
        logger.warning("sec_filing_analyzer: could not load SEC company registry: %s", exc)
        return {}


def get_ticker_for_cik(cik: str) -> Optional[str]:
    """Look up ticker for a CIK. Loads registry if not cached."""
    if not _REGISTRY_LOADED:
        load_sec_company_registry()
    cik_norm = str(cik).lstrip("0") or "0"
    return _CIK_TICKER_CACHE.get(cik_norm) or _CIK_TICKER_CACHE.get(str(cik))


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up CIK for a ticker. Loads registry if not cached."""
    if not _REGISTRY_LOADED:
        load_sec_company_registry()
    return _TICKER_CIK_CACHE.get(ticker.upper())


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

def _resolve_ticker_from_cik(cik: str, company_name: str) -> str:
    """
    Attempt to resolve a stock ticker from a CIK number.

    Resolution order:
      1. In-memory cache (populated by load_sec_company_registry)
      2. SEC EDGAR company search API
      3. yfinance name search (last resort)

    Returns best-guess ticker string or empty string.
    """
    # Step 1: registry cache
    ticker = get_ticker_for_cik(cik)
    if ticker:
        return ticker

    # Step 2: SEC EDGAR full-text search
    try:
        import requests

        safe_name = re.sub(r"[^a-zA-Z0-9 ]", "", company_name)[:60]
        search_url = (
            "https://efts.sec.gov/LATEST/search-index"
            f"?q=%22{safe_name.replace(' ', '%20')}%22"
            "&dateRange=custom&startdt=2024-01-01"
            "&category=form-type&forms=8-K"
        )
        time.sleep(0.1)
        resp = requests.get(search_url, headers=_SEC_HEADERS, timeout=15)
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            if hits:
                src = hits[0].get("_source", {})
                tickers = src.get("period_of_report", "") or ""
                # EDGAR sometimes includes ticker in the entity name; extract it
                entity = src.get("entity_name", "") or company_name
                # crude: take first word if all-caps and ≤5 chars
                first_word = entity.split()[0] if entity else ""
                if first_word.isupper() and 1 <= len(first_word) <= 5:
                    return first_word
    except Exception as exc:
        logger.debug("sec_filing_analyzer: EDGAR search for %s failed: %s", company_name, exc)

    # Step 3: yfinance fallback
    try:
        import yfinance as yf

        result = yf.Search(company_name, max_results=1)
        quotes = result.quotes if hasattr(result, "quotes") else []
        if quotes:
            return quotes[0].get("symbol", "")
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# EDGAR HTTP helpers
# ---------------------------------------------------------------------------

def _sec_get(url: str, session=None) -> Optional[str]:
    """
    Fetch a URL from SEC EDGAR with proper headers and rate limiting.
    Returns response text or None on failure.
    Uses requests.Session if provided for connection pooling.
    """
    try:
        import requests

        time.sleep(0.1)
        req_session = session if (session is not None and hasattr(session, "get")) else requests
        resp = req_session.get(url, headers=_SEC_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        logger.debug("sec_filing_analyzer: HTTP %d for %s", resp.status_code, url)
        return None
    except Exception as exc:
        logger.debug("sec_filing_analyzer: request error for %s: %s", url, exc)
        return None


def _strip_html(html: str) -> str:
    """Strip HTML tags using simple regex — no BeautifulSoup dependency."""
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# EDGAR RSS entry parsing
# ---------------------------------------------------------------------------

def parse_edgar_rss_entry(entry: dict) -> dict:
    """
    Parse a feedparser entry dict from an EDGAR RSS feed.

    Returns:
        {
            "cik": str,
            "accession": str,      # formatted with dashes: XXXXXXXXXX-YY-ZZZZZZ
            "form_type": str,
            "company_name": str,
            "index_url": str,
        }
    """
    result = {
        "cik": "",
        "accession": "",
        "form_type": "",
        "company_name": "",
        "index_url": "",
    }

    # --- company name ---
    result["company_name"] = (
        entry.get("title", "")
        or entry.get("author", "")
        or ""
    ).strip()

    # --- form type ---
    # EDGAR ATOM feeds often have form type in the category or summary
    summary = entry.get("summary", entry.get("description", ""))
    form_match = re.search(
        r"\b(8-K|10-K|10-Q|6-K|S-1|S-3|DEF\s*14A|SC\s*13[DG]|4)\b",
        summary + " " + result["company_name"],
        re.IGNORECASE,
    )
    if form_match:
        result["form_type"] = form_match.group(1).upper().replace(" ", "")
    else:
        result["form_type"] = "8-K"

    # --- CIK and accession from link ---
    link = entry.get("link", entry.get("id", ""))
    # Pattern: https://www.sec.gov/Archives/edgar/data/{cik}/{accession-formatted}/
    arc_match = re.search(
        r"/Archives/edgar/data/(\d+)/([\d\-]+)/",
        link,
    )
    if arc_match:
        result["cik"] = arc_match.group(1)
        result["accession"] = arc_match.group(2)
    else:
        # Try the EDGAR browse URL: CIK=XXXXXXXXXX
        cik_match = re.search(r"[?&]CIK=(\d+)", link, re.IGNORECASE)
        if cik_match:
            result["cik"] = cik_match.group(1)
        # Try accession in the URL path
        acc_match = re.search(r"(\d{10}-\d{2}-\d{6})", link)
        if acc_match:
            result["accession"] = acc_match.group(1)

    # Also check the entry id field for CIK
    if not result["cik"]:
        entry_id = entry.get("id", "")
        cik_match = re.search(r"/data/(\d+)/", entry_id)
        if cik_match:
            result["cik"] = cik_match.group(1)

    # --- index URL ---
    # The EDGAR filing index is at:
    # https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/
    if result["cik"] and result["accession"]:
        acc_clean = result["accession"].replace("-", "")
        result["index_url"] = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{result['cik']}/{acc_clean}/"
        )
    elif result["cik"]:
        # Fall back to browse URL
        result["index_url"] = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={result['cik']}"
            f"&type={result['form_type']}&dateb=&owner=include&count=5"
            f"&search_text=&output=atom"
        )

    return result


# ---------------------------------------------------------------------------
# Filing document fetching
# ---------------------------------------------------------------------------

def fetch_filing_document(
    index_url: str,
    cik: str,
    accession: str,
) -> Optional[str]:
    """
    Fetch and parse the primary document from an EDGAR filing.

    Steps:
      1. Fetch the filing index page (HTML directory listing)
      2. Find the primary .htm / .txt document (the 8-K body itself)
      3. Download and strip HTML to return plain text

    Returns raw plain text or None on failure.
    """
    try:
        import requests as req_lib

        http = req_lib.Session()

        # Build clean accession number
        acc_clean = accession.replace("-", "")
        acc_dashed = (
            accession
            if "-" in accession
            else f"{acc_clean[:10]}-{acc_clean[10:12]}-{acc_clean[12:]}"
        )

        # Fetch the filing index
        index_html = _sec_get(index_url, session=http)
        if not index_html:
            return None

        # Parse document links from the index
        # EDGAR index lists files in a table; look for the primary document
        # Priority: 8-K body (.htm / .txt) that is NOT an exhibit
        doc_candidates: List[tuple] = []

        # Extract all hrefs
        href_pattern = re.compile(
            r'href="(/Archives/edgar/data/[^"]+\.(htm|txt|html))"',
            re.IGNORECASE,
        )
        for m in href_pattern.finditer(index_html):
            path = m.group(1)
            filename = path.split("/")[-1].lower()
            # Exclude exhibits and known non-primary docs
            if re.search(r"ex[\-_]?\d|exhibit|proxy|def14a|cover", filename, re.IGNORECASE):
                continue
            # Prefer files with the accession number in the name (primary doc convention)
            is_primary = acc_dashed.replace("-", "") in filename or acc_clean in filename
            doc_candidates.append((is_primary, path))

        if not doc_candidates:
            # Fallback: try standard EDGAR index file naming convention
            # Primary 8-K is often named {accession}.htm or {acc_clean}.htm
            for ext in [".htm", ".txt"]:
                candidate_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik}/{acc_clean}/{acc_clean}{ext}"
                )
                candidate_html = _sec_get(candidate_url, session=http)
                if candidate_html:
                    return _strip_html(candidate_html)
            return None

        # Sort: primary docs first
        doc_candidates.sort(key=lambda x: (not x[0], len(x[1])))
        doc_path = doc_candidates[0][1]
        doc_url = f"https://www.sec.gov{doc_path}"

        doc_html = _sec_get(doc_url, session=http)
        if not doc_html:
            return None

        return _strip_html(doc_html)

    except Exception as exc:
        logger.debug("sec_filing_analyzer: fetch_filing_document error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Filing text extraction
# ---------------------------------------------------------------------------

def extract_filing_items(text: str, form_type: str) -> List[str]:
    """
    Parse filing body to find which 8-K items are present.

    Recognises patterns:
      "Item 2.02", "ITEM 2.02", "Item 2.02.", "Item 2.02 —"

    Returns list of item numbers found (e.g. ["2.02", "9.01"]).
    """
    if form_type not in ("8-K", "8K"):
        return []

    pattern = re.compile(
        r"\bITEM\s+(\d+\.\d{2})\b",
        re.IGNORECASE,
    )
    found = []
    seen = set()
    for m in pattern.finditer(text):
        item_num = m.group(1)
        if item_num not in seen:
            seen.add(item_num)
            found.append(item_num)
    return found


def extract_key_sentences(text: str, max_sentences: int = 10) -> List[str]:
    """
    Extract sentences most likely to contain material financial information.

    Preference:
      - sentences containing numbers (dollar amounts, percentages)
      - sentences with financial keywords
    """
    financial_keywords = {
        "revenue", "guidance", "growth", "decline", "expect", "forecast",
        "million", "billion", "percent", "quarter", "year", "earnings",
        "profit", "loss", "sales", "outlook", "results", "income",
    }

    # Split into sentences (naive but sufficient for SEC filings)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    scored: List[tuple] = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 500:
            continue
        sent_lower = sent.lower()
        keyword_hits = sum(1 for kw in financial_keywords if kw in sent_lower)
        if keyword_hits == 0:
            continue
        # Bonus for sentences with numbers
        has_numbers = bool(re.search(r"\$[\d,]+|\d+\.?\d*\s*(?:million|billion|percent|%)", sent, re.IGNORECASE))
        score = keyword_hits + (2 if has_numbers else 0)
        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:max_sentences]]


def extract_guidance_signals(text: str) -> dict:
    """
    Detect whether a company raised, lowered, or withdrew guidance.

    Returns:
        {
            "raised": bool,
            "lowered": bool,
            "withdrawn": bool,
            "keywords": List[str],   # matched keyword phrases
        }
    """
    text_lower = text.lower()

    raised_patterns = [
        "raises guidance", "raises its guidance",
        "increases outlook", "increased outlook",
        "raises forecast", "raised forecast",
        "above consensus", "beat consensus",
        r"\bbeat\b", r"\bbeats\b",
        "exceeds", "exceeded",
        "strong demand", "record revenue",
        "raised its", "raising guidance",
        "above expectations", "better than expected",
        "raised guidance",
    ]

    lowered_patterns = [
        "lowered guidance", "lowered its guidance",
        "reduces outlook", "reduced outlook",
        "below expectations", "below expectation",
        r"\bmiss\b", r"\bmisses\b", r"\bmissed\b",
        "shortfall",
        "weak demand", "weakening demand",
        "inventory correction",
        r"\bheadwinds\b",
        "guidance cut", "cut guidance",
        "lower guidance", "lowers guidance",
        "lowered its forecast",
        "reduced its forecast",
    ]

    withdrawn_patterns = [
        "withdraws guidance", "withdraw guidance",
        "withdrew guidance",
        "suspends outlook", "suspended outlook",
        "unable to provide guidance",
        "cannot provide guidance",
        "withdrawing guidance",
    ]

    matched_keywords: List[str] = []

    raised = any(re.search(p, text_lower) for p in raised_patterns)
    lowered = any(re.search(p, text_lower) for p in lowered_patterns)
    withdrawn = any(re.search(p, text_lower) for p in withdrawn_patterns)

    # Collect matched keyword phrases for transparency
    for p in raised_patterns:
        m = re.search(p, text_lower)
        if m:
            matched_keywords.append(m.group(0))
    for p in lowered_patterns:
        m = re.search(p, text_lower)
        if m:
            matched_keywords.append(m.group(0))
    for p in withdrawn_patterns:
        m = re.search(p, text_lower)
        if m:
            matched_keywords.append(m.group(0))

    return {
        "raised": raised,
        "lowered": lowered,
        "withdrawn": withdrawn,
        "keywords": list(set(matched_keywords))[:10],
    }


def extract_risk_flags(text: str) -> List[str]:
    """
    Find sentences containing high-risk language.

    Looks for: regulatory action, litigation, restatements, going concern,
    export restrictions, sanctions, and material weakness disclosures.

    Returns up to 5 most relevant sentences.
    """
    risk_keywords = [
        "may adversely", "could materially", "significant risk",
        "substantial uncertainty", "going concern", "material weakness",
        "restatement", "litigation", "investigation", "subpoena",
        "export restriction", "sanction", "tariff impact",
        "class action", "regulatory action", "enforcement",
        "penalty", "fine", "adverse", "impairment",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    flagged: List[str] = []
    seen_phrases: set = set()

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 600:
            continue
        sent_lower = sent.lower()
        for kw in risk_keywords:
            if kw in sent_lower:
                # Deduplicate by first 60 chars
                key = sent[:60]
                if key not in seen_phrases:
                    seen_phrases.add(key)
                    flagged.append(sent)
                break  # one match per sentence is enough

    return flagged[:5]


# ---------------------------------------------------------------------------
# Material score calculation
# ---------------------------------------------------------------------------

def _calculate_material_score(items_found: List[str], form_type: str) -> float:
    """
    Score how materially significant this filing is on a 0.0–1.0 scale.

    Logic:
      - 10-K / 10-Q are always moderately material (0.5)
      - 8-K score is based on highest-impact item present
      - Multiple material items boost the score
    """
    if form_type in ("10-K", "10K"):
        return 0.5
    if form_type in ("10-Q", "10Q"):
        return 0.4

    if not items_found:
        return 0.1

    max_score = 0.0
    material_count = 0

    for item_num in items_found:
        classification = ITEM_CLASSIFICATIONS.get(item_num, {})
        if not classification:
            continue
        if classification.get("material", False):
            material_count += 1
            # Map direction magnitude to a score contribution
            direction_abs = abs(classification.get("direction", 0.0))
            item_score = 0.3 + direction_abs * 0.7
            if item_score > max_score:
                max_score = item_score

    # Bonus for multiple material items
    if material_count > 1:
        max_score = min(1.0, max_score + 0.1 * (material_count - 1))

    return round(max_score, 3)


def _determine_primary_event(items_found: List[str]) -> tuple:
    """
    Given a list of item numbers, determine the most significant event class
    and the net impact direction.

    Returns: (event_class: str, impact_direction: float)
    """
    best_class = "company_earnings_guidance"
    best_direction = 0.0
    best_priority = -1.0

    for item_num in items_found:
        clf = ITEM_CLASSIFICATIONS.get(item_num, {})
        if not clf:
            continue
        if not clf.get("material", False):
            continue
        direction = clf.get("direction", 0.0)
        # Priority = materiality magnitude
        priority = abs(direction) + (0.3 if clf.get("material") else 0.0)
        if priority > best_priority:
            best_priority = priority
            best_class = clf["event_class"]
            best_direction = direction

    return best_class, best_direction


def _generate_summary(
    company_name: str,
    ticker: str,
    form_type: str,
    items_found: List[str],
    guidance_signals: dict,
    risk_flags: List[str],
    primary_event_class: str,
    impact_direction: float,
    key_disclosures: List[str],
) -> str:
    """
    Generate a plain-English 2-3 sentence summary of the filing.
    """
    name_part = f"{company_name} ({ticker})" if ticker else company_name

    # Opening line: what was filed
    if form_type in ("8-K", "8K"):
        items_desc = []
        for item_num in items_found:
            clf = ITEM_CLASSIFICATIONS.get(item_num, {})
            if clf.get("description"):
                items_desc.append(clf["description"].lower())
        if items_desc:
            items_str = "; ".join(items_desc[:3])
            opening = f"{name_part} filed an 8-K disclosing: {items_str}."
        else:
            opening = f"{name_part} filed an 8-K with no classified items."
    elif form_type in ("10-K", "10K"):
        opening = f"{name_part} filed its annual report (10-K)."
    elif form_type in ("10-Q", "10Q"):
        opening = f"{name_part} filed its quarterly report (10-Q)."
    else:
        opening = f"{name_part} filed a {form_type} with the SEC."

    # Second line: impact direction
    if impact_direction > 0.3:
        impact_str = "The filing contains potentially bullish signals for the company."
    elif impact_direction < -0.3:
        impact_str = "The filing contains potentially bearish signals for the company."
    else:
        impact_str = "The filing is neutral to mildly informational."

    # Third line: guidance or risk
    if guidance_signals.get("raised"):
        extra = "Management raised or signalled positive guidance."
    elif guidance_signals.get("lowered"):
        extra = "Management lowered guidance or flagged headwinds."
    elif guidance_signals.get("withdrawn"):
        extra = "Management withdrew or suspended guidance."
    elif risk_flags:
        extra = f"Risk flag detected: {risk_flags[0][:100]}"
    elif key_disclosures:
        extra = key_disclosures[0][:120]
    else:
        extra = ""

    parts = [opening, impact_str]
    if extra:
        parts.append(extra)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_filing(rss_entry: dict, session=None) -> Optional[FilingAnalysis]:
    """
    Full pipeline: parse EDGAR RSS entry → fetch filing → extract analysis.

    Parameters
    ----------
    rss_entry : dict
        A feedparser entry dict from an EDGAR RSS feed.
    session : optional
        A DB session (passed through but not used for EDGAR HTTP requests;
        DB writes are handled by the caller in ingestion.py).

    Returns FilingAnalysis or None if the filing could not be fetched/parsed.
    """
    try:
        # Step 1: Parse the RSS entry
        parsed = parse_edgar_rss_entry(rss_entry)
        cik = parsed["cik"]
        accession = parsed["accession"]
        form_type = parsed["form_type"]
        company_name = parsed["company_name"]
        index_url = parsed["index_url"]

        if not cik or not index_url:
            logger.debug("sec_filing_analyzer: could not parse CIK/URL from RSS entry")
            return None

        # Filed date from RSS entry
        filed_date = ""
        pub = rss_entry.get("published", rss_entry.get("updated", ""))
        if not pub:
            pub_parsed = rss_entry.get("published_parsed")
            if pub_parsed:
                try:
                    from datetime import datetime
                    year, month, day = pub_parsed[:3]
                    filed_date = datetime(year, max(1, month), max(1, day)).date().isoformat()
                except Exception:
                    pass
        else:
            filed_date = pub[:10] if len(pub) >= 10 else pub

        # Step 2: Resolve ticker
        ticker = _resolve_ticker_from_cik(cik, company_name)

        # Step 3: Fetch the actual filing document
        filing_text = fetch_filing_document(index_url, cik, accession)
        if not filing_text:
            # Return a minimal analysis using only the RSS headline
            return FilingAnalysis(
                ticker=ticker,
                company_name=company_name,
                cik=cik,
                form_type=form_type,
                filed_date=filed_date,
                accession_number=accession,
                filing_url=index_url,
                items_found=[],
                primary_event_class="company_earnings_guidance",
                impact_direction=0.0,
                material_score=0.1,
                key_disclosures=[],
                revenue_mentions=[],
                guidance_signals={"raised": False, "lowered": False, "withdrawn": False, "keywords": []},
                risk_flags=[],
                full_text_snippet="",
                analysis_summary=f"{company_name} filed a {form_type} with the SEC (document unavailable for deep analysis).",
            )

        # Step 4: Extract items
        items_found = extract_filing_items(filing_text, form_type)

        # Step 5: Determine primary event class and direction
        primary_event_class, impact_direction = _determine_primary_event(items_found)

        # Step 6: Material score
        material_score = _calculate_material_score(items_found, form_type)

        # Step 7: Extract key sentences
        key_disclosures = extract_key_sentences(filing_text, max_sentences=10)

        # Revenue-specific mentions (sentences with dollar/percent figures)
        revenue_mentions = [
            s for s in key_disclosures
            if re.search(
                r"\$[\d,]+|\d+\.?\d*\s*(?:million|billion|percent|%)",
                s,
                re.IGNORECASE,
            )
        ]

        # Step 8: Guidance signals
        guidance_signals = extract_guidance_signals(filing_text)

        # Step 9: Risk flags
        risk_flags = extract_risk_flags(filing_text)

        # Full text snippet (first 2000 chars of body)
        full_text_snippet = filing_text[:2000]

        # Step 10: Generate analysis summary
        analysis_summary = _generate_summary(
            company_name=company_name,
            ticker=ticker,
            form_type=form_type,
            items_found=items_found,
            guidance_signals=guidance_signals,
            risk_flags=risk_flags,
            primary_event_class=primary_event_class,
            impact_direction=impact_direction,
            key_disclosures=key_disclosures,
        )

        return FilingAnalysis(
            ticker=ticker,
            company_name=company_name,
            cik=cik,
            form_type=form_type,
            filed_date=filed_date,
            accession_number=accession,
            filing_url=index_url,
            items_found=items_found,
            primary_event_class=primary_event_class,
            impact_direction=impact_direction,
            material_score=material_score,
            key_disclosures=key_disclosures,
            revenue_mentions=revenue_mentions,
            guidance_signals=guidance_signals,
            risk_flags=risk_flags,
            full_text_snippet=full_text_snippet,
            analysis_summary=analysis_summary,
        )

    except Exception as exc:
        logger.warning("sec_filing_analyzer: analyze_filing failed: %s", exc)
        return None
