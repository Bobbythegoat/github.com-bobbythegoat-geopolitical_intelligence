"""
services/processing.py
----------------------
Normalises raw article text, runs causal event analysis, and updates
EventTickerImpact and StockScore records.

Impact Scoring Model — Causal Event-to-Market (CEM)
----------------------------------------------------
Replaces the previous Credibility-Weighted Composite (CWC) which relied on
keyword density and lexicon sentiment. Per the Event-to-Market Intelligence
Model design:

  Old approach (replaced):
    - Keyword-based ticker tagging: "china" + "chip" → score NVDA, AMD, TSM, etc.
    - Lexicon sentiment scoring: count positive/negative words → raw sentiment
    - CWC formula: sign(sentiment) × √(|z_sentiment| × keyword_exposure × credibility)

  New approach (CEM):
    1. classify_economic_event()  → specific event class (15 types, not 4)
    2. extract_named_entities()   → precisely named companies in the article
    3. compute_causal_impacts()   → causal pathways with direction + strength
       Every ticker connection has an explicit economic mechanism:
         "ASML loses Chinese revenue because equipment export is restricted"
       NOT: "ASML score 0.4 because 'export' + 'semiconductor' keyword hits"

  Impact score for each ticker = direction × pathway_strength × confidence
    direction:        -1.0 (bearish) to +1.0 (bullish)
    pathway_strength: how direct is the causal chain (1.0 = primary subject)
    confidence:       event_class_confidence × source_credibility

  Geographies and sectors still extracted by keyword for event metadata.
  The causal engine handles ticker-level impact scoring exclusively.
"""

import re
import math
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

from sqlalchemy.orm import Session
import database as db
from models import OpportunityFactors
from services.scoring import calculate_opportunity

# ---------------------------------------------------------------------------
# Per-event throttling for expensive yfinance calls
# Events whose prices/tickers have been fetched recently are skipped.
# ---------------------------------------------------------------------------

_PRICE_FETCH_CACHE:   Dict[int, float] = {}   # event_id → last fetch timestamp
_DISCOVERY_CACHE:     Dict[int, float] = {}   # event_id → last discovery timestamp
_YFINANCE_CALL_TTL:   int = 300               # seconds between yfinance calls per event (5 min)
_REGIME_CACHE:        Dict[str, float] = {}   # regime → last update timestamp
_REGIME_TTL:          int = 600               # seconds between regime fetches (10 min)

# ---------------------------------------------------------------------------
# Rolling sentiment statistics (in-process, persists for the server lifetime)
# ---------------------------------------------------------------------------

_SENTIMENT_WINDOW: deque = deque(maxlen=200)   # last 200 raw sentiment values
_SENTIMENT_MEAN: float   = 0.0
_SENTIMENT_STD:  float   = 1.0                  # initialised to 1 to avoid div/0


def _update_sentiment_stats(raw_sentiment: float) -> None:
    """Push a new value into the window and recompute mean/std."""
    global _SENTIMENT_MEAN, _SENTIMENT_STD
    _SENTIMENT_WINDOW.append(raw_sentiment)
    if len(_SENTIMENT_WINDOW) < 2:
        return
    n    = len(_SENTIMENT_WINDOW)
    mean = sum(_SENTIMENT_WINDOW) / n
    variance = sum((x - mean) ** 2 for x in _SENTIMENT_WINDOW) / n
    _SENTIMENT_MEAN = mean
    _SENTIMENT_STD  = max(math.sqrt(variance), 1e-6)   # clamp to avoid /0


def _zscore_sentiment(raw_sentiment: float) -> float:
    """Return z-scored sentiment clamped to [-3, 3]."""
    z = (raw_sentiment - _SENTIMENT_MEAN) / _SENTIMENT_STD
    return max(-3.0, min(3.0, z))


# Threshold below which z-score is treated as statistically insignificant
Z_SIGNIFICANCE = 0.15

# Normaliser: maps z ∈ [0, 3] → [0, 1]
_Z_MAX = 3.0


# ---------------------------------------------------------------------------
# Entity keyword maps
# ---------------------------------------------------------------------------

# Ticker → keywords that suggest exposure to a geopolitical/macro event.
# Include company/brand names so articles that mention the firm directly are captured.
TICKER_KEYWORDS: Dict[str, List[str]] = {
    # ── Primary semiconductor universe (CLAUDE.md focus) ──────────────────
    "NVDA": [
        "nvidia", "nvda", "gpu", "h100", "a100", "blackwell", "hopper",
        "data center", "semiconductor", "chip", "ai", "artificial intelligence",
        "china", "taiwan", "export control", "export ban", "computing", "cuda",
        "inference", "training", "ai accelerator",
    ],
    "AMD": [
        "amd", "advanced micro devices", "mi300", "instinct", "epyc", "ryzen",
        "gpu", "cpu", "data center", "chip", "semiconductor", "ai accelerator",
        "china", "taiwan", "export control",
    ],
    "TSM": [
        "tsmc", "taiwan semiconductor", "tsm", "foundry", "wafer",
        "taiwan", "taipei", "strait", "chip", "semiconductor", "china",
        "advanced node", "3nm", "2nm", "5nm", "packaging",
    ],
    "ASML": [
        "asml", "euv", "extreme ultraviolet", "lithography", "photolithography",
        "semiconductor", "chip", "equipment", "export ban", "netherlands",
        "china", "foundry",
    ],
    "AMAT": [
        "applied materials", "amat", "deposition", "etch", "semiconductor equipment",
        "wafer", "foundry", "china", "export control", "chip manufacturing",
    ],
    "LRCX": [
        "lam research", "lrcx", "etch", "deposition", "semiconductor equipment",
        "wafer", "foundry", "china", "export control", "chip manufacturing",
    ],
    "KLAC": [
        "kla", "klac", "process control", "inspection", "metrology",
        "semiconductor equipment", "wafer", "foundry", "china",
    ],
    "MU": [
        "micron", "mu", "dram", "nand", "hbm", "memory", "flash", "storage",
        "semiconductor", "chip", "china", "taiwan", "ai memory",
        "high bandwidth memory", "data center",
    ],
    "AVGO": [
        "broadcom", "avgo", "networking", "asic", "custom silicon",
        "hyperscaler", "ai chip", "data center", "switch", "ethernet",
        "semiconductor", "vmware",
    ],
    "ARM": [
        "arm", "arm holdings", "cpu", "instruction set", "architecture",
        "mobile chip", "data center", "ai", "softbank", "licensing",
    ],
    "INTC": [
        "intel", "intc", "foundry", "18a", "gaudi", "xeon", "core",
        "cpu", "semiconductor", "chip", "manufacturing", "process node",
        "advanced manufacturing", "fab",
    ],
    "QCOM": [
        "qualcomm", "qcom", "snapdragon", "modem", "5g", "wireless",
        "smartphone", "chip", "semiconductor", "china", "licensing",
        "arm", "mobile",
    ],
    "MRVL": [
        "marvell", "mrvl", "data infrastructure", "custom silicon", "asic",
        "cloud", "networking", "storage", "5g", "semiconductor",
    ],
    "SNPS": [
        "synopsys", "snps", "eda", "electronic design automation",
        "chip design", "semiconductor software", "verification", "ip",
    ],
    "CDNS": [
        "cadence", "cdns", "eda", "electronic design automation",
        "chip design", "semiconductor software", "simulation",
    ],
    "TER": [
        "teradyne", "ter", "semiconductor test", "ate", "test equipment",
        "wafer test", "final test", "robotics",
    ],
    "ON": [
        "onsemi", "on semiconductor", "power semiconductor", "ev", "electric vehicle",
        "silicon carbide", "sic", "automotive chip", "industrial",
    ],
    "NXPI": [
        "nxp", "nxpi", "automotive chip", "ev", "radar",
        "microcontroller", "mcu", "iot", "connected car",
    ],
    "TXN": [
        "texas instruments", "txn", "analog", "ti", "microcontroller", "mcu",
        "industrial", "automotive", "power management", "analog chip",
    ],
    "ADI": [
        "analog devices", "adi", "analog", "dsp", "signal processing",
        "industrial", "healthcare", "5g", "defense", "precision",
    ],
    "SMCI": [
        "super micro", "smci", "supermicro", "server", "ai server",
        "nvidia server", "data center", "rack", "liquid cooling",
    ],
    "GFS": [
        "globalfoundries", "gfs", "foundry", "semiconductor", "fab",
        "advanced manufacturing", "specialty", "defense chip",
    ],
    "UMC": [
        "umc", "united microelectronics", "foundry", "wafer", "taiwan",
        "semiconductor", "mature node",
    ],
    "MCHP": [
        "microchip technology", "mchp", "microcontroller", "mcu",
        "automotive", "industrial", "iot", "connectivity",
    ],
    # ── Legacy sectors (retained for macro/geopolitical events) ───────────
    "XOM": [
        "exxon", "exxonmobil", "oil", "crude", "petroleum", "opec",
        "energy", "refinery", "pipeline", "sanctions", "hormuz",
        "iran", "gulf", "drilling", "brent", "wti",
    ],
    "CVX": [
        "chevron", "oil", "crude", "petroleum", "opec", "energy",
        "refinery", "pipeline", "lng", "brent", "wti",
    ],
    "LMT": [
        "lockheed", "lockheed martin", "f-35", "f35", "hypersonic",
        "defense", "defence", "military", "weapons", "pentagon", "nato",
        "war", "missile", "contract", "arms",
    ],
    "RTX": [
        "raytheon", "rtx", "patriot", "air defense", "missile defense",
        "defense", "military", "weapons", "nato", "war",
        "munition", "drone",
    ],
    "GS": [
        "goldman", "goldman sachs", "investment bank", "wall street",
        "banking", "interest rate", "rate hike", "rate cut",
        "fed", "federal reserve", "treasury", "bond", "yield",
    ],
    "JPM": [
        "jpmorgan", "jp morgan", "chase", "jamie dimon",
        "banking", "interest rate", "rate hike", "rate cut",
        "fed", "federal reserve", "treasury", "bond", "yield",
    ],
    "AAPL": [
        "apple", "iphone", "mac", "ipad", "app store", "foxconn",
        "china", "taiwan", "supply chain", "semiconductor", "trade war",
        "tariff", "export ban", "consumer electronics",
    ],
    "BA": [
        "boeing", "737", "787", "dreamliner", "737 max",
        "defense", "aerospace", "military", "aircraft", "faa", "aviation",
    ],
    "GLD": [
        "gold", "gld", "bullion", "precious metal", "safe haven",
        "inflation", "crisis", "geopolitical", "war", "risk off",
    ],
    "TLT": [
        "treasury", "tlt", "bond", "yield", "rate hike", "rate cut",
        "interest rate", "fed", "federal reserve", "quantitative easing",
        "recession", "long bond", "duration",
    ],
    "UNG": [
        "natural gas", "lng", "ung", "gas prices", "pipeline",
        "energy", "russia", "europe", "gazprom", "nordstream",
    ],
    "DBA": [
        "agriculture", "food", "wheat", "grain", "corn", "soybean",
        "drought", "famine", "fertilizer", "food security", "black sea",
        "ukraine", "russia",
    ],
}

SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "Energy":       ["oil", "gas", "energy", "petroleum", "opec", "pipeline"],
    "Defense":      ["defense", "military", "nato", "war", "weapons", "pentagon"],
    "Technology":   ["semiconductor", "chip", "ai", "tech", "software", "china"],
    "Financials":   ["bank", "rate", "fed", "treasury", "credit", "inflation"],
    "Materials":    ["gold", "copper", "mining", "commodities", "silver"],
    "Agriculture":  ["food", "wheat", "grain", "agriculture", "drought"],
}

GEOGRAPHY_KEYWORDS = {
    "Russia":      ["russia", "russian", "kremlin", "moscow", "putin", "gazprom"],
    "China":       ["china", "chinese", "beijing", "xi jinping", "pla", "ccp", "brics"],
    "Middle East": ["iran", "saudi", "opec", "israel", "hamas", "hezbollah", "gulf",
                    "hormuz", "yemen", "houthi", "riyadh", "tehran"],
    "Europe":      ["europe", "european", "eu", "nato", "ukraine", "germany", "france",
                    "ecb", "brussels", "berlin", "paris", "kyiv"],
    "US":          ["fed", "federal reserve", "white house", "pentagon", "congress",
                    "treasury", "washington", "biden", "trump", "us sanctions",
                    "state department"],
    "Taiwan":      ["taiwan", "taipei", "tsmc", "strait", "pla navy"],
    "North Korea": ["north korea", "dprk", "kim jong", "pyongyang", "nuclear",
                    "ballistic missile", "icbm"],
    "India":       ["india", "indian", "modi", "new delhi", "mumbai", "bse", "sensex"],
    "Japan":       ["japan", "japanese", "tokyo", "boj", "bank of japan", "yen"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, remove special characters."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Company-specific anchor keywords — a single match is sufficient to qualify
# a ticker. Generic terms (china, chip, ai) are NOT anchors; they require 2+
# total hits to prevent unrelated news from polluting sector stock scores.
TICKER_ANCHORS: Dict[str, List[str]] = {
    "NVDA": ["nvidia", "nvda", "h100", "a100", "blackwell", "hopper", "cuda"],
    "AMD":  ["amd", "advanced micro devices", "mi300", "instinct", "epyc", "ryzen"],
    "TSM":  ["tsmc", "taiwan semiconductor"],
    "ASML": ["asml", "euv", "extreme ultraviolet", "lithography"],
    "AMAT": ["applied materials", "amat", "deposition", "etch"],
    "LRCX": ["lam research", "lrcx"],
    "KLAC": ["kla", "klac", "metrology", "inspection"],
    "MU":   ["micron", "dram", "nand", "hbm", "high bandwidth memory"],
    "AVGO": ["broadcom", "avgo", "vmware"],
    "ARM":  ["arm holdings", "softbank arm"],
    "INTC": ["intel", "intc", "gaudi", "xeon", "18a"],
    "QCOM": ["qualcomm", "qcom", "snapdragon"],
    "MRVL": ["marvell", "mrvl"],
    "SNPS": ["synopsys", "snps", "electronic design automation"],
    "CDNS": ["cadence", "cdns"],
    "TER":  ["teradyne", "ter"],
    "ON":   ["onsemi", "on semiconductor", "silicon carbide"],
    "NXPI": ["nxp", "nxpi"],
    "TXN":  ["texas instruments", "txn"],
    "ADI":  ["analog devices", "adi"],
    "SMCI": ["super micro", "smci", "supermicro"],
    "GFS":  ["globalfoundries", "gfs"],
    "UMC":  ["umc", "united microelectronics"],
    "MCHP": ["microchip technology", "mchp"],
    "XOM":  ["exxon", "exxonmobil"],
    "CVX":  ["chevron"],
    "LMT":  ["lockheed", "lockheed martin", "f-35"],
    "RTX":  ["raytheon", "rtx", "patriot missile"],
    "GS":   ["goldman sachs", "goldman"],
    "JPM":  ["jpmorgan", "jp morgan", "jamie dimon"],
    "AAPL": ["apple", "iphone", "ipad", "foxconn"],
    "BA":   ["boeing", "737 max", "dreamliner"],
    "GLD":  ["gold etf", "spdr gold", "bullion"],
    "TLT":  ["tlt", "treasury etf", "long bond"],
    "UNG":  ["natural gas etf", "ung"],
    "DBA":  ["dba", "agriculture etf", "invesco db"],
}


def extract_tickers(text: str) -> List[Tuple[str, float]]:
    """
    Return (ticker, raw_exposure_score) pairs found in text.
    Exposure = fraction of keyword matches relative to keyword list length.

    Qualification rule (prevents generic-term false positives):
      A ticker qualifies if it has ≥1 company-specific anchor keyword match
      OR ≥2 total keyword matches.

    This stops articles about unrelated topics (sports, politics) from
    polluting stock scores simply because they contain a generic word like
    "china" or "chip" that appears in many ticker keyword lists.
    """
    norm = normalise_text(text)
    results = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in norm)
        if hits == 0:
            continue
        # Check for a company-specific anchor hit
        anchors = TICKER_ANCHORS.get(ticker, [])
        has_anchor = any(a in norm for a in anchors)
        # Qualify: anchor hit OR at least 2 generic keyword hits
        if not has_anchor and hits < 2:
            continue
        exposure = min(hits / len(keywords), 1.0)
        results.append((ticker, round(exposure, 3)))
    return sorted(results, key=lambda x: x[1], reverse=True)


def extract_sectors(text: str) -> List[str]:
    norm = normalise_text(text)
    return [
        sector for sector, keywords in SECTOR_KEYWORDS.items()
        if any(kw in norm for kw in keywords)
    ]


def extract_geographies(text: str) -> List[str]:
    norm = normalise_text(text)
    return [
        geo for geo, keywords in GEOGRAPHY_KEYWORDS.items()
        if any(kw in norm for kw in keywords)
    ]


def _has_negation_before(text: str, match_start: int, window: int = 35) -> bool:
    """Check if a negation word appears within `window` characters before match_start."""
    NEGATION_WORDS = {"no", "not", "never", "without", "halt", "stop", "avoid",
                      "prevent", "ban", "end", "cancel", "reverse", "oppose"}
    preceding = text[max(0, match_start - window):match_start].lower()
    tokens = preceding.split()[-4:]  # last 4 tokens
    return bool(NEGATION_WORDS.intersection(tokens))


def infer_sentiment(text: str) -> float:
    """
    Expanded lexicon sentiment with domain-specific finance/geopolitical terms.
    Positive words → bullish (+), negative → bearish (−).

    The lexicon is intentionally asymmetric: geopolitical/macro negative terms
    carry more weight because adverse macro events have historically had larger
    and faster price impacts than positive ones (loss-aversion asymmetry).

    Returns raw float in [-1.0, 1.0].
    """
    norm = normalise_text(text)
    text_lower = text.lower()

    # Weighted positive signals (weight, keyword)
    positive_signals = [
        (1.0, "growth"), (1.0, "rally"), (1.0, "surge"), (1.2, "record high"),
        (1.0, "boost"), (1.0, "gain"), (1.0, "gains"), (1.2, "agreement"),
        (1.2, "deal"), (1.0, "recovery"), (1.0, "stable"), (1.0, "expand"),
        (1.0, "increase"), (1.2, "ceasefire"), (1.2, "peace"), (1.0, "upgrade"),
        (0.8, "beat expectations"), (0.8, "guidance raise"), (0.8, "buyback"),
        (1.0, "contract awarded"), (1.0, "approval"), (1.0, "approved"),
    ]

    # Weighted negative signals — intentionally heavier
    negative_signals = [
        (1.5, "war"), (1.5, "invasion"), (1.5, "nuclear"), (1.5, "sanctions"),
        (1.3, "sanction"), (1.3, "conflict"), (1.3, "crisis"), (1.3, "collapse"),
        (1.0, "cut"), (1.2, "ban"), (1.3, "attack"), (1.2, "threat"),
        (1.0, "decline"), (1.3, "recession"), (1.2, "tariff"), (1.2, "tariffs"),
        (1.2, "embargo"), (1.0, "risk"), (1.2, "shortage"), (1.3, "supply chain"),
        (1.0, "downgrade"), (1.0, "miss"), (1.2, "guidance cut"), (1.0, "layoffs"),
        (1.3, "default"), (1.3, "bank failure"), (1.2, "fraud"),
        (1.5, "missile"), (1.5, "strike"), (1.3, "shutdown"),
    ]

    pos_score = 0.0
    for w, kw in positive_signals:
        match_pos = text_lower.find(kw)
        if match_pos != -1:
            neg = _has_negation_before(text_lower, match_pos)
            pos_score += w * (-1 if neg else 1)

    neg_score = 0.0
    for w, kw in negative_signals:
        match_pos = text_lower.find(kw)
        if match_pos != -1:
            neg = _has_negation_before(text_lower, match_pos)
            neg_score += w * (-1 if neg else 1)

    total = pos_score + neg_score
    if total == 0.0:
        return 0.0
    return round((pos_score - neg_score) / total, 4)


# ---------------------------------------------------------------------------
# Credibility-Weighted Composite (CWC) impact calculation
# ---------------------------------------------------------------------------

def _cwc_impact(raw_sentiment: float, exposure: float,
                credibility: float) -> float:
    """
    Credibility-Weighted Composite impact score.

    Formula
    -------
        z  = z-score of raw_sentiment against recent article window
        If |z| < Z_SIGNIFICANCE → impact = 0  (statistically negligible)

        z_norm = |z| / Z_MAX       (maps to [0, 1])
        mag    = √(z_norm × E × C) (geometric mean keeps dims balanced)
        impact = sign(z) × mag     (restore directional sign)

    The geometric mean has the property that if any single factor is zero,
    the whole impact collapses to zero — preventing phantom signals from
    low-exposure or low-credibility articles.

    Returns float in [-1.0, 1.0].
    """
    _update_sentiment_stats(raw_sentiment)

    z = _zscore_sentiment(raw_sentiment)
    if abs(z) < Z_SIGNIFICANCE:
        return 0.0

    z_norm = abs(z) / _Z_MAX                          # [0, 1]
    mag    = math.sqrt(z_norm * exposure * credibility)  # geometric mean
    return round(math.copysign(mag, z), 4)


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Event type classification
# ---------------------------------------------------------------------------

_GEOPOLITICAL_KEYWORDS = {
    "war", "invasion", "conflict", "military", "attack", "missile", "nuclear",
    "nato", "sanctions", "embargo", "ceasefire", "troops", "occupation",
    "coup", "rebellion", "insurgency", "terrorism", "drone strike",
    "air strike", "naval", "blockade", "pentagon", "defence", "defense",
    "strait", "hormuz", "red sea", "houthi", "regime", "treaty",
}

_MACRO_KEYWORDS = {
    "federal reserve", "fed rate", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp", "recession", "yield curve",
    "quantitative easing", "central bank", "treasury", "bond yield",
    "debt ceiling", "bank failure", "banking crisis", "money supply",
    "monetary policy", "fiscal policy", "imf", "world bank",
}

_COMPANY_KEYWORDS = {
    "earnings", "guidance", "quarterly", "revenue", "ipo", "merger",
    "acquisition", "buyback", "dividend", "ceo", "cfo", "management",
    "layoff", "layoffs", "product launch", "filing", "sec filing",
    "shareholder", "restatement", "fraud", "settlement",
}


def infer_event_type(text: str) -> str:
    """
    Classify event type using the causal engine's 15-class system,
    mapped back to the legacy 4-category format for backward compatibility.

    The causal engine's specific classification is used in the main pipeline;
    this function is kept for any remaining callers that expect the 4-bucket format.
    """
    from services.causal_engine import classify_economic_event, CausalAnalysis
    event_class, _ = classify_economic_event(text)
    # Build a minimal CausalAnalysis to access the legacy_event_type property
    legacy_map = {
        "export_restriction_chips":       "geopolitical",
        "export_restriction_equipment":   "geopolitical",
        "supply_disruption_foundry":      "sector",
        "supply_disruption_memory":       "sector",
        "geopolitical_taiwan_risk":       "geopolitical",
        "geopolitical_energy_risk":       "geopolitical",
        "geopolitical_sanctions":         "geopolitical",
        "ai_demand_expansion":            "sector",
        "ai_demand_contraction":          "sector",
        "trade_policy_tariff":            "macro",
        "domestic_manufacturing_subsidy": "macro",
        "company_earnings_guidance":      "company",
        "macro_monetary_hawkish":         "macro",
        "macro_monetary_dovish":          "macro",
        "regulatory_antitrust":           "company",
        "merger_acquisition_tech":        "company",
        "unknown":                        "sector",
    }
    return legacy_map.get(event_class, "sector")


# ---------------------------------------------------------------------------
# Decision bucket computation
# ---------------------------------------------------------------------------

def compute_decision_bucket(
    opportunity_score: float,
    crowding_score: float,
    risk_score: float,
    lag_score: float,
    narrative_stage: str,
    credibility: float,
) -> str:
    """
    Maps score dimensions to a constrained decision bucket.
    Buckets: Early Opportunity | Research Deeper | Watch | Wait for Confirmation |
             Avoid - Crowded | Reduce Exposure | Exit Window Approaching
    """
    if credibility < 0.35:
        return "Wait for Confirmation"
    # Guard: insufficient signal — don't assign actionable bucket regardless of stage
    if opportunity_score < 0.10:
        return "Watch"
    if narrative_stage == "declining" and opportunity_score >= 0.10 and opportunity_score < 0.3:
        return "Exit Window Approaching"
    if crowding_score >= 0.60 and narrative_stage in ("peak", "declining"):
        return "Avoid - Crowded"
    if risk_score >= 0.65 and opportunity_score < 0.4:
        return "Reduce Exposure"
    if opportunity_score >= 0.65 and credibility >= 0.60 and narrative_stage in ("emerging", "developing") and lag_score >= 0.50:
        return "Early Opportunity"
    if opportunity_score >= 0.45 and credibility >= 0.50:
        return "Research Deeper"
    return "Watch"


def process_article(article_id: int, session: Session):
    """
    Run the full causal event analysis pipeline on a newly stored article and
    update EventTickerImpact and StockScore records accordingly.

    CEM Pipeline:
      1. Causal event analysis  — classify event type, extract named entities,
                                  compute causal impacts with explicit pathways
      2. Expectation Gap        — surprise vs. market consensus
      3. Narrative Inflection   — attention velocity and turning-point detection
      4. Second-Order Impact    — supply-chain transmission graph
      5. Stock score update     — all sub-scores recalculated

    Key difference from previous CWC approach:
      - Every ticker-event connection has an explicit economic mechanism
      - Direction is determined by causal logic, not sentiment keywords
      - Tickers not causally connected to the event are excluded (not tagged)
      - Named entities (specifically mentioned companies) get full strength;
        inferred downstream tickers get reduced strength automatically
    """
    from services.causal_engine import analyze_article as causal_analyze_article

    article = session.get(db.Article, article_id)
    if article is None or article.event_id is None:
        return

    event     = session.get(db.Event, article.event_id)
    full_text = f"{article.headline} {article.content}"

    credibility  = event.credibility_score if event else 0.5
    geographies  = extract_geographies(full_text)

    # ── Causal event analysis ─────────────────────────────────────────────────
    # This replaces extract_tickers() + infer_sentiment() + _cwc_impact()
    causal = causal_analyze_article(full_text, credibility=credibility)

    # ── Update event metadata ─────────────────────────────────────────────────
    if event:
        # Use specific causal event class; fall back to legacy type for
        # backward compatibility with parts of the system that still use event_type
        event.event_type   = causal.legacy_event_type
        event.source_count = len(event.articles)
        distinct_sources   = {a.source for a in event.articles}
        event.source_breadth = len(distinct_sources)
        if geographies:
            existing = set((event.geography_tags or "").split(",")) - {""}
            existing.update(geographies)
            event.geography_tags = ",".join(sorted(existing))

    # ── Write causal impacts to EventTickerImpact ─────────────────────────────
    # Minimum net impact below which we still exclude (near-zero causal signals
    # indicate the pathway is too weak to register in the scoring system)
    MIN_CAUSAL_NET_IMPACT = 0.12

    for causal_impact in causal.causal_impacts:
        ticker_sym = causal_impact.ticker
        net_score  = causal_impact.net_impact   # direction × strength × confidence

        if abs(net_score) < MIN_CAUSAL_NET_IMPACT:
            continue

        ticker_row = session.get(db.Ticker, ticker_sym)
        if ticker_row is None:
            continue

        impact_row = (
            session.query(db.EventTickerImpact)
            .filter_by(event_id=event.event_id, ticker=ticker_sym)
            .first()
        )
        if impact_row:
            # Blend new causal score with existing (70/30 recency-weighted)
            # This handles multiple articles on the same event updating the same impact
            blended = round(0.70 * net_score + 0.30 * impact_row.impact_score, 4)
            impact_row.impact_score = blended
        else:
            session.add(db.EventTickerImpact(
                event_id=event.event_id,
                ticker=ticker_sym,
                impact_score=net_score,
            ))

    session.commit()

    # ── Phase 2+: Expectation Gap ─────────────────────────────────────────
    try:
        from services.expectation_gap import compute_event_expectation_gaps
        compute_event_expectation_gaps(event.event_id, session)
    except Exception as e:
        print(f"[expectation_gap] Warning: {e}")

    # ── Phase 2+: Narrative Inflection ────────────────────────────────────
    try:
        from services.narrative_inflection import compute_narrative_inflection
        compute_narrative_inflection(event.event_id, session)
    except Exception as e:
        print(f"[narrative_inflection] Warning: {e}")

    # ── Commodity & strategic-input tracking ──────────────────────────────
    # Scans this article for tier-1/2/3 commodity signals (copper, uranium,
    # gallium, neon, photoresist, etc.) and writes both CommoditySignal rows
    # and synthetic EventTickerImpact entries against the commodity symbol.
    # The second_order engine below then propagates the shock to every
    # exposed downstream ticker via the seeded commodity_input edges.
    try:
        from services.commodity_tracker import link_event_to_commodities
        link_event_to_commodities(article_id, session)
    except Exception as e:
        print(f"[commodity_tracker] Warning: {e}")

    # ── Phase 2+: Second-Order Impact ─────────────────────────────────────
    try:
        from services.second_order import update_stock_indirect_scores
        update_stock_indirect_scores(event.event_id, session)
    except Exception as e:
        print(f"[second_order] Warning: {e}")

    # Fetch real price data — at most once every 5 minutes per event
    # (yfinance calls are slow; skipping per-article deduplication prevents
    #  seconds of network I/O for every article in a clustered event)
    if event and event.event_id:
        now = time.monotonic()
        last_price_fetch = _PRICE_FETCH_CACHE.get(event.event_id, 0.0)
        if (now - last_price_fetch) >= _YFINANCE_CALL_TTL:
            try:
                from services.price_fetcher import fetch_and_update_event_prices
                fetch_and_update_event_prices(event.event_id, session)
                _PRICE_FETCH_CACHE[event.event_id] = now
            except Exception:
                pass

    # Discover new tickers — at most once every 5 minutes per event
    if event and event.event_id:
        now = time.monotonic()
        last_discovery = _DISCOVERY_CACHE.get(event.event_id, 0.0)
        if (now - last_discovery) >= _YFINANCE_CALL_TTL:
            try:
                from services.stock_discovery import discover_tickers_from_event
                discover_tickers_from_event(event.event_id, session)
                _DISCOVERY_CACHE[event.event_id] = now
            except Exception:
                pass

    _update_stock_scores(session)


def _event_recency_decay(event: db.Event) -> float:
    """
    Time-decay multiplier for an event's contribution to stock scores.
    Events older than EVENT_SCORE_MAX_AGE_DAYS are excluded entirely.

    Decay schedule:
      0–1 days  → 1.00 (full weight)
      1–3 days  → 0.80 (slightly stale)
      3–7 days  → 0.40 (significantly stale)
      7+ days   → 0.00 (excluded)
    """
    EVENT_SCORE_MAX_AGE_DAYS = 7
    if event is None:
        return 1.0
    # Event has a single `timestamp` column (set on creation, advanced on
    # cluster updates). Older drafts referenced last_updated_at/first_seen_at,
    # which never existed on the ORM model.
    ref_ts = getattr(event, "timestamp", None)
    if not ref_ts:
        return 1.0
    # SQLAlchemy returns naive UTC datetimes here; compare against utcnow().
    if ref_ts.tzinfo is not None:
        ref_ts = ref_ts.replace(tzinfo=None)
    age_days = (datetime.utcnow() - ref_ts).total_seconds() / 86400.0
    if age_days >= EVENT_SCORE_MAX_AGE_DAYS:
        return 0.0
    if age_days <= 1.0:
        return 1.0
    if age_days <= 3.0:
        return 1.0 - 0.20 * ((age_days - 1.0) / 2.0)   # 1.0 → 0.80
    # 3–7 days: 0.80 → 0.05
    return max(0.05, 0.80 - 0.75 * ((age_days - 3.0) / 4.0))


def _update_stock_scores(session: Session):
    """
    Recompute StockScore for every ticker that has causal impact data.

    Key improvements over previous CWC version:
    • exposure_score = mean causal pathway strength (not keyword fraction)
      Pathway strength reflects how directly the ticker is causally connected
      to events — primary subjects score high, distant third-order effects low
    • impact_score   = credibility-weighted directional causal impact
    • risk_score     = credibility-weighted RMS of causal impact scores
    • All scores now driven by explicit economic mechanisms, not keyword density
    • Recency decay: events older than 7 days are excluded; 3–7 days are downweighted
    """
    all_impacts = session.query(db.EventTickerImpact).all()
    if not all_impacts:
        return

    impacts_by_ticker: Dict[str, list] = defaultdict(list)
    for imp in all_impacts:
        impacts_by_ticker[imp.ticker].append(imp)

    event_ids  = {imp.event_id for imp in all_impacts}
    events     = session.query(db.Event).filter(db.Event.event_id.in_(event_ids)).all()
    ev_cred    = {e.event_id: e.credibility_score  for e in events}
    ev_stage   = {e.event_id: e.narrative_stage    for e in events}
    events_map = {e.event_id: e                    for e in events}  # Phase 2+
    # Recency decay per event — events older than 7 days are excluded/downweighted
    ev_decay   = {e.event_id: _event_recency_decay(e) for e in events}

    # Narrative-stage → estimated price-reaction lag
    # Emerging events have high lag (market hasn't priced in yet)
    STAGE_LAG = {
        "emerging":   0.80,
        "developing": 0.55,
        "peak":       0.25,
        "declining":  0.10,
    }

    existing_scores = {
        row.ticker: row
        for row in session.query(db.StockScore).all()
    }

    # Retrieve adaptive weights once for the entire batch (not per ticker)
    adaptive_weights = None
    try:
        from services.learning import get_weights, get_current_regime
        _regime          = get_current_regime(session)
        adaptive_weights = get_weights(session, _regime)
    except Exception:
        pass

    for ticker_sym, impacts in impacts_by_ticker.items():
        # Filter out fully-decayed (stale) impacts before aggregation
        active_impacts = [
            i for i in impacts
            if ev_decay.get(i.event_id, 1.0) > 0.0
        ]
        if not active_impacts:
            # All events for this ticker are stale — reset to neutral
            score_row = existing_scores.get(ticker_sym)
            if score_row:
                score_row.opportunity_score = 0.0
                score_row.crowding_score    = 0.0
                score_row.risk_score        = 0.0
                score_row.exposure_score    = 0.0
                score_row.impact_score      = 0.0
                score_row.decision_bucket   = "Watch"
            continue

        impacts = active_impacts

        ticker_event_ids = [i.event_id for i in impacts]
        ticker_stages    = [ev_stage.get(eid, "emerging") for eid in ticker_event_ids]

        # Apply decay to raw impact scores before aggregation
        decayed_scores = [
            i.impact_score * ev_decay.get(i.event_id, 1.0)
            for i in impacts
        ]
        total_abs = sum(abs(s) for s in decayed_scores) or 1e-9

        # Credibility-weighted mean credibility
        avg_credibility = sum(
            ev_cred.get(i.event_id, 0.5) * abs(decayed_scores[j])
            for j, i in enumerate(impacts)
        ) / total_abs

        # Credibility-weighted RMS impact (risk proxy)
        rms_impact = math.sqrt(
            sum(
                ev_cred.get(i.event_id, 0.5) * (decayed_scores[j] ** 2)
                for j, i in enumerate(impacts)
            ) / max(len(impacts), 1)
        )

        crowding = (
            sum(1 for s in ticker_stages if s == "peak")
            / max(len(ticker_stages), 1)
        )

        # Best event = highest |CWC impact| × credibility
        best_event_id = max(
            impacts,
            key=lambda i: abs(i.impact_score) * ev_cred.get(i.event_id, 0.5)
        ).event_id
        narrative = ev_stage.get(best_event_id, "emerging")
        lag       = STAGE_LAG.get(narrative, 0.5)

        exposure_val  = round(min(total_abs / max(len(impacts), 1), 1.0), 4)
        impact_val    = round(avg_credibility * exposure_val, 4)
        narrative_val = round({"emerging": 1.0, "developing": 0.75, "peak": 0.40, "declining": 0.15}.get(narrative, 0.5), 4)
        lag_val       = round(lag, 4)
        # Asymmetry: positive impacts vs negative (bullish bias = positive asymmetry)
        # Uses decay-adjusted scores so stale events don't skew direction
        pos_impact    = sum(abs(s) for s in decayed_scores if s > 0)
        neg_impact    = sum(abs(s) for s in decayed_scores if s < 0)
        total_impact  = pos_impact + neg_impact or 1e-9
        asymmetry_val = round((pos_impact - neg_impact) / total_impact, 4)

        # Phase 2+: Expectation gap from best event; indirect impact from existing score row
        best_event_obj  = events_map.get(best_event_id)
        gap_score       = float(getattr(best_event_obj, 'expectation_proxy', None) or 0.0)
        indirect_score  = float(getattr(existing_scores.get(ticker_sym), 'indirect_impact_score', 0.0) or 0.0)

        opp_factors = OpportunityFactors(
            exposure=exposure_val,
            credibility=avg_credibility,
            narrative_stage=narrative,
            crowding=crowding,
            price_reaction_lag=lag,
            risk=rms_impact,
            expectation_gap=gap_score,
            indirect_impact=indirect_score,
            asymmetry=asymmetry_val,
        )
        opp_score = calculate_opportunity(opp_factors, weights=adaptive_weights)

        bucket = compute_decision_bucket(
            opportunity_score=opp_score,
            crowding_score=crowding,
            risk_score=rms_impact,
            lag_score=lag,
            narrative_stage=narrative,
            credibility=avg_credibility,
        )

        score_row = existing_scores.get(ticker_sym)
        if score_row:
            score_row.opportunity_score     = round(opp_score, 4)
            score_row.crowding_score        = round(crowding, 4)
            score_row.risk_score            = round(rms_impact, 4)
            score_row.exposure_score        = exposure_val
            score_row.impact_score          = impact_val
            score_row.narrative_score       = narrative_val
            score_row.lag_score             = lag_val
            score_row.asymmetry_score       = asymmetry_val
            score_row.expectation_gap_score = round(gap_score, 4)   # Phase 2+
            # indirect_impact_score is updated by second_order engine; only set if not already set
            if not score_row.indirect_impact_score:
                score_row.indirect_impact_score = round(indirect_score, 4)
            score_row.decision_bucket       = bucket
        else:
            new_row = db.StockScore(
                ticker=ticker_sym,
                opportunity_score=round(opp_score, 4),
                crowding_score=round(crowding, 4),
                risk_score=round(rms_impact, 4),
                exposure_score=exposure_val,
                impact_score=impact_val,
                narrative_score=narrative_val,
                lag_score=lag_val,
                asymmetry_score=asymmetry_val,
                expectation_gap_score=round(gap_score, 4),    # Phase 2+
                indirect_impact_score=round(indirect_score, 4), # Phase 2+
                decision_bucket=bucket,
            )
            session.add(new_row)
            existing_scores[ticker_sym] = new_row

    session.commit()
