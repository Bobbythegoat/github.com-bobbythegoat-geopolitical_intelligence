"""
services/causal_engine.py
--------------------------
Causal Event-to-Market Intelligence Engine

Implements the Event-to-Market Intelligence Model design:

  1. Extract the real-world event (ignore commentary)
  2. Classify the type of economic impact (supply shock, export restriction, etc.)
  3. Identify directly named entities (companies explicitly mentioned as subjects)
  4. Trace causal economic pathways to affected downstream/upstream tickers
  5. Map effects onto specific securities with explicit causal justification
  6. Evaluate strength and timing of impact
  7. Filter out weak or speculative connections

This replaces:
  - Keyword-based ticker tagging (extract_tickers in processing.py)
  - Lexicon sentiment analysis (infer_sentiment in processing.py)
  - CWC keyword-density scoring (_cwc_impact in processing.py)

Design principles:
  - Every ticker connection requires an explicit economic mechanism
  - Direction is determined by causal logic, not sentiment words
  - Strength reflects proximity in the causal chain, not keyword density
  - A ticker with no causal pathway to the event is excluded entirely
  - Weak/indirect connections below MIN_PATHWAY_STRENGTH are filtered out
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Minimum pathway strength to include a ticker in the impact set ──────────
# Connections weaker than this are speculative and add noise without signal.
MIN_PATHWAY_STRENGTH = 0.18

# ── When a ticker is specifically NAMED in the article, boost its strength ──
# If the article explicitly names ASML in an equipment restriction story,
# that's far more certain than inferring ASML is affected from context alone.
NAMED_ENTITY_BOOST = 1.0   # strength multiplier for named entities (no reduction)
UNNAMED_ENTITY_MULTIPLIER = 0.65  # reduce strength for entities not named in article


# ─────────────────────────────────────────────────────────────────────────────
# CausalImpact dataclass — one per affected ticker per event analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CausalImpact:
    ticker: str
    direction: float         # -1.0 (very negative) → +1.0 (very positive)
    pathway_strength: float  # 0.0 → 1.0 (how strong is the causal link)
    pathway_type: str        # e.g. "direct_revenue_exclusion", "supply_chain_disruption"
    reasoning: str           # human-readable causal explanation
    confidence: float        # 0.0 → 1.0 (confidence in this specific impact)
    impact_horizon: str      # "intraday" | "short_term" | "medium_term" | "structural"
    is_primary: bool         # True if this ticker is the direct subject of the event

    @property
    def net_impact(self) -> float:
        """Directional causal impact: direction × strength × confidence ∈ [-1, 1]."""
        return round(self.direction * self.pathway_strength * self.confidence, 4)


@dataclass
class CausalAnalysis:
    event_class: str                      # specific economic event type
    event_class_confidence: float         # 0.0 → 1.0
    named_entities: Dict[str, float]      # {ticker: presence_confidence}
    causal_impacts: List[CausalImpact]    # filtered list of impacted tickers
    impact_horizon: str
    overall_confidence: float

    # Human-readable summary of the causal chain
    mechanism_summary: str = ""

    # Legacy compatibility: maps to old event_type field
    @property
    def legacy_event_type(self) -> str:
        CLASS_TO_LEGACY = {
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
        return CLASS_TO_LEGACY.get(self.event_class, "sector")


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY_REGISTRY — precise company/product name → ticker mapping
# Used for named entity detection (much more precise than keyword lists)
# ─────────────────────────────────────────────────────────────────────────────

# Format: (text_pattern, ticker, confidence_weight)
# confidence_weight: 1.0 = company name directly, 0.8 = product name implies ticker
ENTITY_PATTERNS: List[Tuple[str, str, float]] = [
    # ── NVDA ──────────────────────────────────────────────────────────────────
    ("nvidia",                   "NVDA", 1.0),
    (r"\bnvda\b",                "NVDA", 1.0),
    ("h100",                     "NVDA", 0.90),
    ("h200",                     "NVDA", 0.90),
    ("a100",                     "NVDA", 0.90),
    ("b200",                     "NVDA", 0.90),
    ("blackwell",                "NVDA", 0.90),
    ("hopper gpu",               "NVDA", 0.90),
    ("cuda platform",            "NVDA", 0.85),
    ("dgx",                      "NVDA", 0.85),
    # ── AMD ───────────────────────────────────────────────────────────────────
    (r"\bamd\b",                 "AMD",  1.0),
    ("advanced micro devices",   "AMD",  1.0),
    ("mi300",                    "AMD",  0.90),
    ("mi250",                    "AMD",  0.90),
    ("instinct gpu",             "AMD",  0.90),
    (r"\bepyc\b",                "AMD",  0.85),
    (r"\bryzen\b",               "AMD",  0.80),
    # ── TSM ───────────────────────────────────────────────────────────────────
    ("tsmc",                     "TSM",  1.0),
    ("taiwan semiconductor",     "TSM",  1.0),
    (r"\btsm\b",                 "TSM",  0.95),
    ("taiwan foundry",           "TSM",  0.90),
    ("hsinchu fab",              "TSM",  0.85),
    # ── ASML ──────────────────────────────────────────────────────────────────
    (r"\basml\b",                "ASML", 1.0),
    ("euv lithography",          "ASML", 0.95),
    (r"\beuv\b",                 "ASML", 0.85),
    ("extreme ultraviolet",      "ASML", 0.85),
    # ── AMAT ──────────────────────────────────────────────────────────────────
    ("applied materials",        "AMAT", 1.0),
    (r"\bamat\b",                "AMAT", 1.0),
    # ── LRCX ──────────────────────────────────────────────────────────────────
    ("lam research",             "LRCX", 1.0),
    (r"\blrcx\b",                "LRCX", 1.0),
    # ── KLAC ──────────────────────────────────────────────────────────────────
    (r"\bkla\b",                 "KLAC", 1.0),
    (r"\bklac\b",                "KLAC", 1.0),
    # ── MU ────────────────────────────────────────────────────────────────────
    ("micron",                   "MU",   1.0),
    (r"\b(dram|nand|hbm)\b",     "MU",   0.80),
    ("high bandwidth memory",    "MU",   0.80),
    # ── AVGO ──────────────────────────────────────────────────────────────────
    ("broadcom",                 "AVGO", 1.0),
    (r"\bavgo\b",                "AVGO", 1.0),
    # ── ARM ───────────────────────────────────────────────────────────────────
    ("arm holdings",             "ARM",  1.0),
    (r"\barm\b(?! forces| army)", "ARM", 0.80),
    # ── INTC ──────────────────────────────────────────────────────────────────
    (r"\bintel\b",               "INTC", 1.0),
    (r"\bintc\b",                "INTC", 1.0),
    ("intel foundry",            "INTC", 0.95),
    (r"\b18a\b",                 "INTC", 0.85),
    ("gaudi",                    "INTC", 0.85),
    # ── QCOM ──────────────────────────────────────────────────────────────────
    ("qualcomm",                 "QCOM", 1.0),
    (r"\bqcom\b",                "QCOM", 1.0),
    ("snapdragon",               "QCOM", 0.90),
    # ── MRVL ──────────────────────────────────────────────────────────────────
    ("marvell",                  "MRVL", 1.0),
    (r"\bmrvl\b",                "MRVL", 1.0),
    # ── SNPS ──────────────────────────────────────────────────────────────────
    ("synopsys",                 "SNPS", 1.0),
    (r"\bsnps\b",                "SNPS", 1.0),
    # ── CDNS ──────────────────────────────────────────────────────────────────
    ("cadence",                  "CDNS", 1.0),
    (r"\bcdns\b",                "CDNS", 1.0),
    # ── TER ───────────────────────────────────────────────────────────────────
    ("teradyne",                 "TER",  1.0),
    (r"\bter\b",                 "TER",  0.85),
    # ── ON ────────────────────────────────────────────────────────────────────
    ("onsemi",                   "ON",   1.0),
    ("on semiconductor",         "ON",   1.0),
    # ── NXPI ──────────────────────────────────────────────────────────────────
    (r"\bnxp\b",                 "NXPI", 1.0),
    (r"\bnxpi\b",                "NXPI", 1.0),
    # ── TXN ───────────────────────────────────────────────────────────────────
    ("texas instruments",        "TXN",  1.0),
    (r"\btxn\b",                 "TXN",  1.0),
    # ── ADI ───────────────────────────────────────────────────────────────────
    ("analog devices",           "ADI",  1.0),
    (r"\badi\b",                 "ADI",  0.85),
    # ── SMCI ──────────────────────────────────────────────────────────────────
    ("super micro",              "SMCI", 1.0),
    ("supermicro",               "SMCI", 1.0),
    (r"\bsmci\b",                "SMCI", 1.0),
    # ── GFS ───────────────────────────────────────────────────────────────────
    ("globalfoundries",          "GFS",  1.0),
    (r"\bgfs\b",                 "GFS",  0.85),
    # ── UMC ───────────────────────────────────────────────────────────────────
    ("united microelectronics",  "UMC",  1.0),
    (r"\bumc\b",                 "UMC",  0.85),
    # ── MCHP ──────────────────────────────────────────────────────────────────
    ("microchip technology",     "MCHP", 1.0),
    (r"\bmchp\b",                "MCHP", 1.0),
    # ── Non-semiconductor tickers ─────────────────────────────────────────────
    ("exxonmobil",               "XOM",  1.0),
    (r"\bexxon\b",               "XOM",  0.95),
    (r"\bxom\b",                 "XOM",  1.0),
    ("chevron",                  "CVX",  1.0),
    (r"\bcvx\b",                 "CVX",  1.0),
    ("lockheed martin",          "LMT",  1.0),
    (r"\blockheed\b",            "LMT",  0.95),
    (r"\blmt\b",                 "LMT",  1.0),
    (r"\bf-35\b",                "LMT",  0.85),
    ("raytheon",                 "RTX",  1.0),
    (r"\brtx\b",                 "RTX",  1.0),
    ("goldman sachs",            "GS",   1.0),
    ("goldman",                  "GS",   0.90),
    (r"\bjpmorgan\b",            "JPM",  1.0),
    ("jp morgan",                "JPM",  1.0),
    (r"\bapple\b(?! tv| watch| music| podcast| pay)",  "AAPL", 0.90),
    ("iphone",                   "AAPL", 0.85),
    (r"\bboeing\b",              "BA",   1.0),
    (r"\b737 max\b",             "BA",   0.90),
]

# Pre-compile all patterns
_COMPILED_ENTITY_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), ticker, weight)
    for pat, ticker, weight in ENTITY_PATTERNS
]


# ─────────────────────────────────────────────────────────────────────────────
# EVENT_CLASS_PATTERNS — scores article text against 15 economic event classes
#
# Structure: {event_class: [pattern_group, ...]}
# A pattern_group is (primary_terms, context_terms):
#   - At least 1 primary_term must match
#   - At least 1 context_term must match (or context_terms=None → any primary match qualifies)
# Each matched group adds 1 point. Highest total wins.
# ─────────────────────────────────────────────────────────────────────────────

_PRIMARY_RESTRICTION_TERMS = [
    "export control", "export ban", "export restriction", "chip ban",
    "export license", "denied license", "license required", "trade restriction",
    "blocked from", "prohibit", "restrict sale", "restrict export",
    "entity list", "commerce department", "bis rule", "export administration",
    "department of commerce",
]

_CHIP_CONTEXT_TERMS = [
    "semiconductor", "chip", "gpu", "ai chip", "advanced computing",
    "nvidia", "amd", "qualcomm", "broadcom", "integrated circuit",
    "processor", "logic chip", "high performance", "data center chip",
]

_EQUIPMENT_CONTEXT_TERMS = [
    "asml", "applied materials", "lam research", "kla", "klac",
    "semiconductor equipment", "fab equipment", "lithography", "wafer",
    "deposition", "etch", "metrology", "process control", "euv",
    "semiconductor tool", "chip manufacturing equipment",
]

_SUPPLY_DISRUPTION_TERMS = [
    "disruption", "disrupted", "outage", "shutdown", "halt", "halted",
    "earthquake", "fire", "flood", "typhoon", "power outage", "blackout",
    "production stop", "capacity reduction", "fab offline", "forced closure",
    "supply shortage", "supply crunch", "capacity crunch",
]

_FOUNDRY_CONTEXT = [
    "tsmc", "taiwan semiconductor", "foundry", "wafer fab", "chip factory",
    "semiconductor plant", "intel foundry", "samsung foundry", "globalfoundries",
]

_TAIWAN_RISK_TERMS = [
    "taiwan strait", "strait of taiwan", "pla", "chinese military", "invasion",
    "taiwan conflict", "taiwan crisis", "taiwan war", "china taiwan",
    "taiwan independence", "taiwan sovereignty", "cross-strait",
    "taiwan blockade", "pla navy", "military exercises taiwan",
]

_ENERGY_RISK_TERMS = [
    "oil supply", "oil shock", "opec cut", "oil embargo", "gas supply",
    "energy supply", "oil pipeline", "hormuz", "strait of hormuz",
    "oil sanction", "iran oil", "russia oil", "russia gas",
    "nordstream", "energy crisis", "oil price spike",
]

_AI_DEMAND_POSITIVE = [
    "ai spending", "ai investment", "ai capex", "ai infrastructure",
    "data center expansion", "hyperscaler capex", "ai accelerator demand",
    "gpu demand", "ai chip demand", "training demand", "inference demand",
    "ai deployment", "ai buildout", "ai arms race",
    "ai revenue", "ai growth", "generative ai demand",
]

_AI_DEMAND_NEGATIVE = [
    "ai slowdown", "ai spending cut", "ai capex reduction", "ai budget cut",
    "ai efficiency", "ai cost reduction", "cheaper ai", "ai substitute",
    "model efficiency", "ai demand falling", "ai overspend", "ai bubble",
    "ai revenue miss", "ai capex pause", "data center pause",
]

_TARIFF_TERMS = [
    "tariff", "tariffs", "import duty", "import tax", "trade war",
    "trade barriers", "customs duty", "levy on imports",
]

_SUBSIDY_TERMS = [
    "chips act", "subsidy", "grant", "government funding", "federal funding",
    "domestic manufacturing", "reshoring", "onshoring", "fab subsidy",
    "manufacturing incentive", "industrial policy", "advanced manufacturing",
    "semiconductor subsidy", "national security chip",
]

_EARNINGS_TERMS = [
    "earnings", "quarterly results", "guidance", "revenue beat", "revenue miss",
    "eps beat", "eps miss", "raised guidance", "lowered guidance", "cut guidance",
    "q1 results", "q2 results", "q3 results", "q4 results",
    "annual results", "fiscal year", "reported earnings", "posted earnings",
]

_HAWKISH_MONETARY_TERMS = [
    "rate hike", "rate increase", "federal reserve hike", "fed hike", "fomc hike",
    "tightening monetary", "quantitative tightening", "qt", "higher rates",
    "inflation fight", "restrictive policy", "rate holds", "holds rates",
    "above neutral", "hawkish fed", "hawkish signal",
]

_DOVISH_MONETARY_TERMS = [
    "rate cut", "rate decrease", "fed cut", "fomc cut", "lower rates",
    "easing monetary", "quantitative easing", "qe", "rate pause",
    "accommodative policy", "dovish fed", "dovish signal", "pivot",
    "inflation cooling", "disinflationary",
]

_ANTITRUST_TERMS = [
    "antitrust", "doj investigation", "ftc investigation", "competition probe",
    "monopoly investigation", "market dominance probe", "competition authority",
    "eu investigation", "merger blocked", "acquisition blocked",
    "competition violation", "abuse of dominance",
]

_MA_TERMS = [
    "acquisition", "merger", "takeover", "buyout", "deal agreement",
    "agreed to acquire", "agreed to buy", "tender offer",
    "definitive agreement", "strategic combination",
]

_SANCTIONS_TERMS = [
    "sanction", "sanctions", "sanctioned", "ofac", "treasury sanction",
    "us sanction", "eu sanction", "asset freeze", "blacklisted",
    "blocked entity", "designated entity", "restricted entity",
]

EVENT_CLASS_PATTERNS: Dict[str, List[Tuple]] = {
    "export_restriction_chips": [
        (_PRIMARY_RESTRICTION_TERMS, _CHIP_CONTEXT_TERMS),
        (["restrict", "ban", "prohibit", "block", "deny", "denied"],
         ["nvidia", "amd", "qualcomm", "ai chip", "advanced chip", "h100", "mi300"]),
        (["entity list", "export administration", "bis rule"],
         ["semiconductor", "chip", "artificial intelligence", "computing"]),
    ],
    "export_restriction_equipment": [
        (_PRIMARY_RESTRICTION_TERMS, _EQUIPMENT_CONTEXT_TERMS),
        (["restrict", "ban", "prohibit", "block", "deny", "denied"],
         ["asml", "applied materials", "lam research", "kla", "semiconductor equipment"]),
        (["equipment export", "tool export", "fab equipment ban"],
         None),
    ],
    "supply_disruption_foundry": [
        (_SUPPLY_DISRUPTION_TERMS, _FOUNDRY_CONTEXT),
        (["production halt", "fab fire", "fab earthquake", "foundry shutdown",
          "wafer shortage", "chip shortage", "production delay"],
         None),
        (["tsmc", "taiwan semiconductor", "samsung foundry"],
         _SUPPLY_DISRUPTION_TERMS),
    ],
    "supply_disruption_memory": [
        (_SUPPLY_DISRUPTION_TERMS, ["dram", "nand", "hbm", "memory", "micron", "sk hynix", "samsung memory"]),
        (["memory shortage", "dram shortage", "nand shortage", "hbm shortage", "memory supply"],
         None),
    ],
    "geopolitical_taiwan_risk": [
        (_TAIWAN_RISK_TERMS, None),
        (["taiwan", "strait"],
         ["military", "war", "conflict", "invasion", "blockade", "exercises", "tension", "crisis"]),
        (["china", "pla"],
         ["taiwan", "strait", "invasion", "amphibious", "blockade"]),
    ],
    "geopolitical_energy_risk": [
        (_ENERGY_RISK_TERMS, None),
        (["iran", "saudi arabia", "opec", "russia", "houthi", "yemen"],
         ["oil", "gas", "energy", "pipeline", "tanker", "hormuz", "supply"]),
        (["oil price", "energy price", "gas price"],
         ["sanction", "attack", "disruption", "supply", "conflict"]),
    ],
    "geopolitical_sanctions": [
        (_SANCTIONS_TERMS,
         ["russia", "china", "iran", "north korea", "company", "entity", "bank"]),
        (["asset freeze", "financial sanction", "ofac"],
         None),
    ],
    "ai_demand_expansion": [
        (_AI_DEMAND_POSITIVE, None),
        (["artificial intelligence", "ai", "machine learning"],
         ["capex", "investment", "spending", "demand", "infrastructure", "build",
          "hyperscaler", "data center", "accelerator"]),
        (["gpu", "accelerator", "ai chip"],
         ["demand", "order", "record", "surge", "increase", "growth"]),
    ],
    "ai_demand_contraction": [
        (_AI_DEMAND_NEGATIVE, None),
        (["artificial intelligence", "ai"],
         ["slowdown", "cut", "reduce", "pause", "efficient", "cheaper", "lower",
          "disappointment", "miss", "bubble", "overspend"]),
    ],
    "trade_policy_tariff": [
        (_TARIFF_TERMS,
         ["technology", "semiconductor", "chip", "electronic", "computer", "trade",
          "import", "china", "manufacturing"]),
        (["import duty", "trade war", "customs levy"],
         None),
    ],
    "domestic_manufacturing_subsidy": [
        (_SUBSIDY_TERMS,
         ["semiconductor", "chip", "manufacturing", "foundry", "fab", "technology"]),
        (["chips act", "chips and science act"],
         None),
        (["government fund", "federal grant"],
         ["semiconductor", "chip", "manufacturing", "foundry"]),
    ],
    "company_earnings_guidance": [
        (_EARNINGS_TERMS,
         ["nvidia", "tsmc", "amd", "asml", "intel", "qualcomm", "micron",
          "broadcom", "marvell", "applied materials", "lam research", "kla",
          "supermicro", "globalfoundries"]),
        (["earnings beat", "earnings miss", "guidance raised", "guidance cut",
          "guidance lowered", "revenue surprised"],
         None),
    ],
    "macro_monetary_hawkish": [
        (_HAWKISH_MONETARY_TERMS, None),
        (["federal reserve", "fed", "fomc", "central bank"],
         ["rate", "hike", "increase", "tighten", "inflation", "hawkish"]),
    ],
    "macro_monetary_dovish": [
        (_DOVISH_MONETARY_TERMS, None),
        (["federal reserve", "fed", "fomc", "central bank"],
         ["cut", "reduce", "lower", "ease", "dovish", "pivot", "accommodative"]),
    ],
    "regulatory_antitrust": [
        (_ANTITRUST_TERMS, None),
        (["investigation", "probe", "inquiry"],
         ["competition", "antitrust", "monopoly", "market power", "dominant"]),
    ],
    "merger_acquisition_tech": [
        (_MA_TERMS,
         ["semiconductor", "chip", "technology", "nvidia", "amd", "intel",
          "qualcomm", "broadcom", "marvell", "synopsys", "cadence"]),
        (["acquisition", "merger", "takeover", "buyout"],
         ["billion", "deal", "company", "firm", "agreed"]),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL_GRAPH — for each event_class, maps tickers to their causal impact spec
#
# Each entry: (direction, base_strength, pathway_type, reasoning, horizon)
#   direction: -1.0 to +1.0
#   base_strength: 0.0 to 1.0 (BEFORE named-entity boost or unnamed-entity reduction)
#   pathway_type: short descriptor of causal mechanism
#   reasoning: plain-English justification required by the model design
#   horizon: impact time horizon
# ─────────────────────────────────────────────────────────────────────────────

_H_SHORT    = "short_term"
_H_MEDIUM   = "medium_term"
_H_INTRADAY = "intraday"
_H_STRUCT   = "structural"

CAUSAL_GRAPH: Dict[str, Dict[str, Tuple]] = {

    # ── Export restriction: AI/advanced chips ────────────────────────────────
    "export_restriction_chips": {
        "NVDA": (-1.0, 0.95, "direct_revenue_exclusion",
                 "NVIDIA has highest China data-center revenue exposure (~20-25%); export restrictions directly remove that market",
                 _H_MEDIUM),
        "AMD":  (-1.0, 0.82, "direct_revenue_exclusion",
                 "AMD Instinct MI-series GPUs face the same export restrictions as NVIDIA for Chinese data centers",
                 _H_MEDIUM),
        "QCOM": (-0.70, 0.65, "direct_revenue_exclusion",
                 "Qualcomm derives significant revenue from Chinese smartphone and IoT chip sales",
                 _H_MEDIUM),
        "MRVL": (-0.60, 0.55, "direct_revenue_exclusion",
                 "Marvell custom AI chips for hyperscalers may face restriction if China-targeted",
                 _H_MEDIUM),
        "SMCI": (-0.65, 0.70, "downstream_demand_loss",
                 "SuperMicro AI servers require NVIDIA GPUs; restricted GPU supply eliminates China server sales",
                 _H_SHORT),
        "MU":   (-0.40, 0.50, "downstream_demand_contraction",
                 "Fewer GPU shipments reduce demand for HBM memory co-packaged with restricted chips",
                 _H_MEDIUM),
        "TSM":  (-0.25, 0.35, "foundry_utilization_decline",
                 "Lower wafer starts from restricted fabless customers reduce TSMC utilization",
                 _H_MEDIUM),
        "ASML": (-0.18, 0.25, "capital_expenditure_delay",
                 "Slowing Chinese fab expansion reduces near-term demand for ASML EUV tools",
                 _H_MEDIUM),
        "AMAT": (-0.18, 0.25, "capital_expenditure_delay",
                 "Chinese customers delay fab tool purchases under export restriction regime",
                 _H_MEDIUM),
        "INTC": ( 0.20, 0.30, "competitor_narrative_benefit",
                 "Intel positioned as domestically compliant alternative; gains narrative tailwind",
                 _H_MEDIUM),
        "GFS":  ( 0.15, 0.25, "competitor_narrative_benefit",
                 "GlobalFoundries benefits from reshoring and domestic-chip policy narrative",
                 _H_STRUCT),
    },

    # ── Export restriction: semiconductor manufacturing equipment ────────────
    "export_restriction_equipment": {
        "ASML": (-1.0, 0.95, "direct_revenue_exclusion",
                 "ASML cannot legally ship EUV/DUV tools to restricted Chinese fabs; direct revenue loss",
                 _H_MEDIUM),
        "AMAT": (-0.90, 0.88, "direct_revenue_exclusion",
                 "Applied Materials loses Chinese foundry equipment orders (China ~27% of revenue historically)",
                 _H_MEDIUM),
        "LRCX": (-0.90, 0.88, "direct_revenue_exclusion",
                 "Lam Research loses etch/deposition tool orders from Chinese customers",
                 _H_MEDIUM),
        "KLAC": (-0.85, 0.82, "direct_revenue_exclusion",
                 "KLA loses process-control and inspection equipment sales in China",
                 _H_MEDIUM),
        "TER":  (-0.40, 0.45, "direct_revenue_exclusion",
                 "Teradyne loses semiconductor test equipment orders from Chinese customers",
                 _H_MEDIUM),
        "TSM":  ( 0.15, 0.25, "competitor_capacity_constraint",
                 "Chinese foundry capacity constrained by equipment denial; TSMC gains relative advantage",
                 _H_STRUCT),
        "NVDA": ( 0.10, 0.18, "competitive_moat_narrative",
                 "Chinese foundry capacity constrained, reducing competitive chip production pressure on NVIDIA",
                 _H_STRUCT),
    },

    # ── Physical/operational disruption at foundry ───────────────────────────
    "supply_disruption_foundry": {
        "TSM":  (-1.0, 0.95, "direct_operational_disruption",
                 "TSMC is the disrupted entity; production halts propagate to all advanced-node customers",
                 _H_SHORT),
        "NVDA": (-0.80, 0.85, "critical_supply_constraint",
                 "NVIDIA sources 100% of advanced AI GPUs from TSMC; no alternative foundry exists at 4nm/3nm",
                 _H_SHORT),
        "AMD":  (-0.75, 0.82, "critical_supply_constraint",
                 "AMD Instinct and EPYC lines manufactured exclusively at TSMC advanced nodes",
                 _H_SHORT),
        "AAPL": (-0.70, 0.78, "critical_supply_constraint",
                 "Apple A-series and M-series chips are exclusively TSMC-manufactured",
                 _H_SHORT),
        "QCOM": (-0.65, 0.72, "supply_constraint",
                 "Qualcomm Snapdragon and modem chips rely heavily on TSMC advanced nodes",
                 _H_SHORT),
        "AVGO": (-0.55, 0.60, "supply_constraint",
                 "Broadcom custom silicon and networking ASICs use TSMC advanced nodes",
                 _H_SHORT),
        "ARM":  (-0.35, 0.40, "indirect_ecosystem_impact",
                 "ARM licensees constrained in production; impacts IP licensing revenue trajectory",
                 _H_MEDIUM),
        "MU":   (-0.30, 0.38, "downstream_demand_effect",
                 "GPU production constraint reduces demand for HBM memory co-packaged with GPU chips",
                 _H_SHORT),
        "ASML": (-0.22, 0.30, "equipment_idle_asset",
                 "ASML EUV tools installed at disrupted TSMC generate lower utilization fees",
                 _H_SHORT),
        "AMAT": (-0.18, 0.25, "equipment_idle_asset",
                 "Applied Materials equipment utilization at disrupted TSMC facilities declines",
                 _H_SHORT),
        "INTC": ( 0.25, 0.35, "competitor_market_share",
                 "Intel Foundry Services positioned to capture emergency orders from TSMC customers",
                 _H_SHORT),
    },

    # ── Memory supply shock ──────────────────────────────────────────────────
    "supply_disruption_memory": {
        "MU":   (-0.75, 0.78, "direct_supply_disruption",
                 "Micron is primary DRAM/NAND supplier; disruption directly impacts available capacity",
                 _H_SHORT),
        "NVDA": (-0.40, 0.45, "hbm_supply_constraint",
                 "HBM memory supply constraint limits H100/H200/B200 GPU final assembly volumes",
                 _H_SHORT),
        "AMD":  (-0.35, 0.40, "hbm_supply_constraint",
                 "AMD Instinct GPUs require HBM; memory supply shock constrains final assembly",
                 _H_SHORT),
        "SMCI": (-0.30, 0.35, "system_memory_constraint",
                 "SuperMicro AI servers require large DRAM configurations; memory shortage limits builds",
                 _H_SHORT),
    },

    # ── Taiwan geopolitical risk ─────────────────────────────────────────────
    "geopolitical_taiwan_risk": {
        "TSM":  (-1.0, 0.90, "direct_geopolitical_exposure",
                 "TSMC is physically in Taiwan; military conflict would halt all production permanently",
                 _H_INTRADAY),
        "NVDA": (-0.80, 0.82, "supply_chain_concentration_risk",
                 "NVIDIA has no alternative foundry for 4nm/3nm AI GPUs; Taiwan risk is existential supply risk",
                 _H_INTRADAY),
        "AMD":  (-0.75, 0.78, "supply_chain_concentration_risk",
                 "AMD similarly concentrated at TSMC with no near-term advanced-node alternative",
                 _H_INTRADAY),
        "AAPL": (-0.70, 0.75, "supply_chain_concentration_risk",
                 "Apple's entire chip supply is dependent on TSMC Taiwan operations",
                 _H_INTRADAY),
        "QCOM": (-0.60, 0.68, "supply_chain_concentration_risk",
                 "Qualcomm Snapdragon and modem production concentrated at TSMC Taiwan",
                 _H_INTRADAY),
        "AVGO": (-0.55, 0.62, "supply_chain_concentration_risk",
                 "Broadcom networking and custom AI chips use TSMC advanced nodes",
                 _H_INTRADAY),
        "ASML": (-0.65, 0.68, "equipment_stranded_asset",
                 "ASML EUV tools installed at TSMC Taiwan would be inaccessible in a conflict scenario",
                 _H_INTRADAY),
        "AMAT": (-0.52, 0.58, "equipment_stranded_asset",
                 "Applied Materials equipment at TSMC Taiwan represents a stranded asset risk",
                 _H_INTRADAY),
        "LRCX": (-0.52, 0.58, "equipment_stranded_asset",
                 "Lam Research equipment at TSMC Taiwan would be stranded in a conflict",
                 _H_INTRADAY),
        "MRVL": (-0.55, 0.62, "supply_chain_concentration_risk",
                 "Marvell custom silicon for hyperscalers uses TSMC advanced nodes",
                 _H_INTRADAY),
        "INTC": ( 0.30, 0.40, "geopolitical_hedge_narrative",
                 "Intel US-based manufacturing is the primary beneficiary of Taiwan diversification narrative",
                 _H_MEDIUM),
        "GFS":  ( 0.22, 0.32, "geopolitical_hedge_narrative",
                 "GlobalFoundries US/EU manufacturing benefits from Taiwan supply-chain diversification push",
                 _H_MEDIUM),
    },

    # ── Energy/oil supply disruption ─────────────────────────────────────────
    "geopolitical_energy_risk": {
        "XOM":  ( 0.80, 0.85, "direct_commodity_price_benefit",
                 "Exxon's upstream production directly benefits from higher crude oil prices",
                 _H_SHORT),
        "CVX":  ( 0.75, 0.82, "direct_commodity_price_benefit",
                 "Chevron benefits from higher crude prices across its production portfolio",
                 _H_SHORT),
        "UNG":  ( 0.70, 0.75, "direct_commodity_price_benefit",
                 "Natural gas ETF benefits from energy supply disruption",
                 _H_SHORT),
        "GLD":  ( 0.60, 0.65, "safe_haven_demand",
                 "Gold demand increases in geopolitical risk-off environment",
                 _H_INTRADAY),
        "TLT":  (-0.30, 0.35, "inflation_rate_pressure",
                 "Energy-driven inflation increases rate expectations, pressuring long-duration bonds",
                 _H_SHORT),
        "LMT":  ( 0.50, 0.58, "defense_demand_benefit",
                 "Lockheed Martin benefits from elevated defense spending in conflict scenarios",
                 _H_MEDIUM),
        "RTX":  ( 0.50, 0.58, "defense_demand_benefit",
                 "Raytheon benefits from increased missile and air defense system procurement",
                 _H_MEDIUM),
        "TSM":  (-0.22, 0.28, "input_cost_pressure",
                 "Semiconductor fabrication is energy-intensive; higher energy costs pressure margins",
                 _H_MEDIUM),
        "INTC": (-0.22, 0.28, "input_cost_pressure",
                 "Intel's fab-heavy model has higher energy cost exposure than fabless peers",
                 _H_MEDIUM),
    },

    # ── Entity-level sanctions ───────────────────────────────────────────────
    "geopolitical_sanctions": {
        "GLD":  ( 0.42, 0.45, "safe_haven_demand",
                 "Sanctions typically trigger financial market stress; gold benefits as safe-haven",
                 _H_INTRADAY),
        "TLT":  ( 0.22, 0.28, "safe_haven_demand",
                 "Geopolitical sanctions create risk-off flows into US Treasuries",
                 _H_INTRADAY),
        # When a specific company is sanctioned, entity-specific overrides dominate
        # These defaults apply to macro/country-level sanctions only
    },

    # ── AI demand expansion ──────────────────────────────────────────────────
    "ai_demand_expansion": {
        "NVDA": ( 1.0,  0.95, "primary_demand_beneficiary",
                 "NVIDIA GPUs are the dominant compute infrastructure for AI training and inference",
                 _H_SHORT),
        "AMD":  ( 0.70, 0.75, "direct_demand_alternative",
                 "AMD Instinct GPU series benefits as second-source AI accelerator",
                 _H_SHORT),
        "SMCI": ( 0.82, 0.82, "infrastructure_systems_demand",
                 "SuperMicro AI server shipments are directly tied to GPU deployment volumes",
                 _H_SHORT),
        "MU":   ( 0.70, 0.75, "memory_demand",
                 "HBM and DRAM demand grows with AI compute cluster deployments",
                 _H_SHORT),
        "AVGO": ( 0.65, 0.70, "networking_and_custom_asic",
                 "Broadcom custom AI ASICs and networking chips benefit from hyperscaler AI capex",
                 _H_SHORT),
        "TSM":  ( 0.60, 0.65, "foundry_capacity_demand",
                 "TSMC benefits from increased wafer starts for AI chips across NVDA, AMD, custom ASICs",
                 _H_SHORT),
        "ASML": ( 0.50, 0.55, "equipment_capacity_expansion",
                 "AI-driven foundry expansion requires additional EUV lithography tools",
                 _H_MEDIUM),
        "AMAT": ( 0.50, 0.55, "equipment_capacity_expansion",
                 "Applied Materials benefits from new fab construction for AI chip demand",
                 _H_MEDIUM),
        "LRCX": ( 0.45, 0.50, "equipment_capacity_expansion",
                 "Lam Research benefits from deposition and etch tool orders for AI fabs",
                 _H_MEDIUM),
        "KLAC": ( 0.45, 0.50, "equipment_capacity_expansion",
                 "KLA process control tools are needed for advanced AI chip manufacturing nodes",
                 _H_MEDIUM),
        "MRVL": ( 0.62, 0.68, "custom_silicon_demand",
                 "Marvell custom AI accelerators for hyperscalers benefit from AI capex expansion",
                 _H_SHORT),
        "SNPS": ( 0.40, 0.45, "eda_design_activity",
                 "Synopsys EDA tools are essential for complex AI chip design; more designs = more revenue",
                 _H_MEDIUM),
        "CDNS": ( 0.40, 0.45, "eda_design_activity",
                 "Cadence EDA tools benefit from increased AI chip design complexity and volume",
                 _H_MEDIUM),
        "ARM":  ( 0.45, 0.50, "ip_licensing_growth",
                 "ARM CPU IP is increasingly used in custom AI accelerator designs; licensing revenue grows",
                 _H_MEDIUM),
    },

    # ── AI demand contraction ────────────────────────────────────────────────
    "ai_demand_contraction": {
        "NVDA": (-0.85, 0.88, "primary_demand_decline",
                 "Reduced AI capex directly reduces NVIDIA GPU order volumes and revenue guidance",
                 _H_SHORT),
        "AMD":  (-0.70, 0.72, "demand_decline",
                 "AMD Instinct GPU demand falls with overall AI infrastructure spending",
                 _H_SHORT),
        "SMCI": (-0.78, 0.78, "infrastructure_demand_decline",
                 "SuperMicro server demand falls with reduced GPU deployment volumes",
                 _H_SHORT),
        "MU":   (-0.60, 0.65, "memory_demand_decline",
                 "HBM demand declines as AI cluster growth and new GPU builds slow",
                 _H_SHORT),
        "AVGO": (-0.55, 0.60, "networking_demand_decline",
                 "Broadcom networking and custom ASIC demand declines with AI capex slowdown",
                 _H_SHORT),
        "TSM":  (-0.45, 0.50, "foundry_utilization_decline",
                 "Reduced chip volumes from AI slowdown lower TSMC utilization and margins",
                 _H_MEDIUM),
        "ASML": (-0.35, 0.40, "equipment_order_decline",
                 "Foundry expansion plans deferred; fewer EUV tool orders result",
                 _H_MEDIUM),
        "MRVL": (-0.55, 0.60, "custom_silicon_program_cutback",
                 "Hyperscaler AI ASIC programs deprioritized in a spending pullback",
                 _H_SHORT),
        "AMAT": (-0.30, 0.35, "equipment_order_decline",
                 "Applied Materials sees deferred tool orders as fabs slow capacity expansion",
                 _H_MEDIUM),
    },

    # ── Technology tariffs ───────────────────────────────────────────────────
    "trade_policy_tariff": {
        "AAPL": (-0.70, 0.72, "assembly_cost_increase",
                 "Apple's China-assembled iPhones face tariff-driven cost increases; margin or price pressure",
                 _H_SHORT),
        "SMCI": (-0.52, 0.56, "direct_tariff_exposure",
                 "SuperMicro assembles servers in Asia; tariffs directly increase product costs",
                 _H_SHORT),
        "NVDA": (-0.38, 0.42, "supply_chain_cost",
                 "NVIDIA servers and cards assembled in Asia face tariff-driven cost increases",
                 _H_SHORT),
        "TSM":  (-0.30, 0.35, "operational_cost",
                 "TSMC US fab construction costs increase if semiconductor equipment is tariffed",
                 _H_MEDIUM),
        "INTC": (-0.22, 0.28, "supply_chain_disruption",
                 "Intel faces tariff pressure on components sourced from Asian supply chain",
                 _H_SHORT),
        "GLD":  ( 0.30, 0.35, "inflation_hedge",
                 "Tariffs are inflationary; gold benefits as inflation hedge",
                 _H_SHORT),
        "TLT":  (-0.28, 0.32, "inflation_rate_pressure",
                 "Tariff-driven inflation increases rate expectations, pressuring long-duration bonds",
                 _H_SHORT),
    },

    # ── Domestic manufacturing subsidies (CHIPS Act, etc.) ───────────────────
    "domestic_manufacturing_subsidy": {
        "INTC": ( 0.80, 0.82, "direct_subsidy_recipient",
                 "Intel is the primary beneficiary of CHIPS Act with the largest US fab expansion plans",
                 _H_STRUCT),
        "GFS":  ( 0.65, 0.68, "direct_subsidy_recipient",
                 "GlobalFoundries receives CHIPS Act funding for US fab expansion",
                 _H_STRUCT),
        "TSM":  ( 0.55, 0.60, "direct_subsidy_recipient",
                 "TSMC Arizona receives CHIPS Act grants for US advanced-node manufacturing",
                 _H_STRUCT),
        "MU":   ( 0.60, 0.65, "direct_subsidy_recipient",
                 "Micron receives CHIPS Act support for US memory fab expansion",
                 _H_STRUCT),
        "AMAT": ( 0.50, 0.55, "equipment_demand_expansion",
                 "New domestic fabs require large equipment orders from Applied Materials",
                 _H_STRUCT),
        "LRCX": ( 0.45, 0.50, "equipment_demand_expansion",
                 "New domestic fab construction drives Lam Research etch/dep tool orders",
                 _H_STRUCT),
        "KLAC": ( 0.45, 0.50, "equipment_demand_expansion",
                 "KLA process control tools are required in every new domestic fab",
                 _H_STRUCT),
        "ASML": ( 0.38, 0.42, "equipment_demand_expansion",
                 "New US advanced fabs require ASML EUV tools; subsidy makes them viable",
                 _H_STRUCT),
        "SNPS": ( 0.40, 0.45, "eda_demand_expansion",
                 "Domestic chip design activity increases Synopsys EDA tool demand",
                 _H_STRUCT),
        "CDNS": ( 0.40, 0.45, "eda_demand_expansion",
                 "Cadence benefits from expanded US chip design and verification activity",
                 _H_STRUCT),
    },

    # ── Hawkish monetary policy ──────────────────────────────────────────────
    "macro_monetary_hawkish": {
        "NVDA": (-0.45, 0.45, "valuation_compression",
                 "High-PE growth stock; rising discount rate compresses terminal value multiple",
                 _H_SHORT),
        "AMD":  (-0.42, 0.42, "valuation_compression",
                 "Growth stock valuation sensitive to interest rate increases",
                 _H_SHORT),
        "SMCI": (-0.38, 0.38, "valuation_compression",
                 "High-growth server company faces multiple compression in rising-rate environment",
                 _H_SHORT),
        "ASML": (-0.32, 0.35, "valuation_compression",
                 "ASML's long-duration earnings stream is sensitive to discount rate",
                 _H_SHORT),
        "TSM":  (-0.28, 0.32, "capital_cost_increase",
                 "Capital-intensive foundry faces higher cost of capital for ongoing CapEx",
                 _H_MEDIUM),
        "TLT":  (-0.72, 0.78, "direct_price_inverse",
                 "Long bond ETF price falls inversely as yields rise with hawkish Fed",
                 _H_INTRADAY),
        "GLD":  (-0.38, 0.42, "opportunity_cost",
                 "Rising real rates increase the opportunity cost of holding gold",
                 _H_SHORT),
        "JPM":  ( 0.22, 0.28, "net_interest_margin_benefit",
                 "JPMorgan net interest income improves with higher rates on its loan book",
                 _H_MEDIUM),
    },

    # ── Dovish monetary policy ───────────────────────────────────────────────
    "macro_monetary_dovish": {
        "NVDA": ( 0.40, 0.40, "valuation_expansion",
                 "Lower discount rate expands terminal value multiple for high-growth tech",
                 _H_SHORT),
        "AMD":  ( 0.36, 0.36, "valuation_expansion",
                 "Growth stock multiple expands with lower discount rate",
                 _H_SHORT),
        "ASML": ( 0.30, 0.33, "capital_cost_reduction",
                 "Lower cost of capital benefits capital-intensive businesses with long investment cycles",
                 _H_SHORT),
        "TSM":  ( 0.26, 0.30, "capital_cost_reduction",
                 "TSMC capital-intensive foundry operations benefit from lower borrowing costs",
                 _H_SHORT),
        "TLT":  ( 0.72, 0.78, "direct_price_inverse",
                 "Long bond ETF price rises inversely as yields fall with dovish Fed",
                 _H_INTRADAY),
        "GLD":  ( 0.40, 0.45, "opportunity_cost_reduction",
                 "Lower real rates reduce the opportunity cost of holding non-yielding gold",
                 _H_SHORT),
    },

    # ── Antitrust / regulatory ───────────────────────────────────────────────
    "regulatory_antitrust": {
        # Minimal defaults; entity-specific overrides dominate for named companies
        "GS":  (-0.28, 0.30, "regulatory_legal_cost",
                "Financial firms face compliance costs and constrained activities from regulatory action",
                _H_MEDIUM),
        "JPM": (-0.22, 0.25, "regulatory_legal_cost",
                "Banking regulation increases compliance burden and constrains capital deployment",
                _H_MEDIUM),
    },

    # ── M&A activity in tech ─────────────────────────────────────────────────
    "merger_acquisition_tech": {
        # Default sector sentiment effect; entity-specific overrides dominate
        "SNPS": ( 0.20, 0.25, "sector_consolidation_premium",
                 "EDA sector consolidation drives premium expectations for remaining independent players",
                 _H_SHORT),
        "CDNS": ( 0.20, 0.25, "sector_consolidation_premium",
                 "Cadence benefits from semiconductor sector M&A consolidation narrative",
                 _H_SHORT),
    },

    # ── Company earnings/guidance ────────────────────────────────────────────
    "company_earnings_guidance": {
        # Entity-specific pathways dominate entirely here
        # These defaults are minimal sector-sympathy effects only
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Event classification engine
# ─────────────────────────────────────────────────────────────────────────────

def _score_text_against_group(text_lower: str,
                               primary_terms: List[str],
                               context_terms: Optional[List[str]]) -> bool:
    """
    Return True if the pattern group matches:
    at least 1 primary_term is found AND (context_terms is None OR at least 1 context_term is found).
    """
    primary_hit = any(t in text_lower for t in primary_terms)
    if not primary_hit:
        return False
    if context_terms is None:
        return True
    return any(t in text_lower for t in context_terms)


def classify_economic_event(text: str) -> Tuple[str, float]:
    """
    Classify article text into one of 15 specific economic event classes.

    Returns (event_class, confidence) where:
      event_class: string key from CAUSAL_GRAPH
      confidence: 0.0 → 1.0 (fraction of matched pattern groups vs total)

    Replaces infer_event_type() which used only 4 vague categories with
    keyword counting. This version uses structured pattern groups that
    require both primary and context terms to match.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {}

    for event_class, pattern_groups in EVENT_CLASS_PATTERNS.items():
        score = 0
        for primary_terms, context_terms in pattern_groups:
            if _score_text_against_group(text_lower, primary_terms, context_terms):
                score += 1
        if score > 0:
            scores[event_class] = score

    if not scores:
        return "unknown", 0.0

    # Best class = highest score; break ties by specificity (more pattern groups = more specific)
    best_class = max(scores, key=lambda k: (scores[k], len(EVENT_CLASS_PATTERNS.get(k, []))))
    best_score = scores[best_class]
    total_groups = len(EVENT_CLASS_PATTERNS.get(best_class, [1]))
    confidence = min(best_score / max(total_groups, 1), 1.0)

    return best_class, round(confidence, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Named entity extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_named_entities(text: str) -> Dict[str, float]:
    """
    Find precisely named companies and products in text and map them to tickers.

    Returns {ticker: entity_confidence} where entity_confidence ∈ [0.0, 1.0].

    Key improvement over extract_tickers():
    - Uses exact company/product name matching, not generic keywords
    - "china" appearing in a NVDA article does NOT count as an AMD mention
    - Only named entities are marked as primary; others get reduced strength
    """
    text_lower = text.lower()
    entity_scores: Dict[str, float] = {}

    for pattern, ticker, weight in _COMPILED_ENTITY_PATTERNS:
        if pattern.search(text_lower):
            # Take the highest weight if multiple patterns match same ticker
            current = entity_scores.get(ticker, 0.0)
            entity_scores[ticker] = max(current, weight)

    return entity_scores


# ─────────────────────────────────────────────────────────────────────────────
# Impact direction override for named entities in earnings events
# ─────────────────────────────────────────────────────────────────────────────

def _infer_earnings_direction(text: str) -> float:
    """
    For company_earnings_guidance events, infer direction from the article.
    Returns -1.0 (miss/cut), 0.0 (neutral), +1.0 (beat/raise).
    """
    text_lower = text.lower()
    positive_earnings = [
        "beat", "beats", "exceeded", "surpassed", "raised guidance",
        "raises guidance", "raised forecast", "stronger than expected",
        "above expectations", "record revenue", "record earnings",
        "revenue beat", "eps beat", "guidance raise", "bullish outlook",
    ]
    negative_earnings = [
        "miss", "missed", "fell short", "below expectations", "cut guidance",
        "cuts guidance", "lowered guidance", "lowered forecast", "warned",
        "revenue miss", "eps miss", "guidance cut", "weak outlook",
        "disappointing", "missed estimates",
    ]
    pos_hits = sum(1 for t in positive_earnings if t in text_lower)
    neg_hits = sum(1 for t in negative_earnings if t in text_lower)
    if pos_hits > neg_hits:
        return 1.0
    if neg_hits > pos_hits:
        return -1.0
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Causal impact computation — the core engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_causal_impacts(
    event_class: str,
    event_class_confidence: float,
    named_entities: Dict[str, float],
    credibility: float,
    text: str = "",
) -> List[CausalImpact]:
    """
    Compute causal impacts for all affected tickers given the event class
    and named entities.

    Algorithm:
      1. Retrieve the causal pathway graph for this event_class
      2. For each ticker in the graph:
         a. Look up direction, base_strength, pathway_type, reasoning, horizon
         b. If ticker is named in the article → full base_strength (NAMED_ENTITY_BOOST)
            else → reduce by UNNAMED_ENTITY_MULTIPLIER (less certain)
         c. Scale confidence by event_class_confidence × credibility
      3. For named entities not in the graph, add entity-specific self-impacts
         (handles earnings/guidance events where the specific named company is
         the primary subject but isn't pre-mapped in a default graph)
      4. Filter out impacts below MIN_PATHWAY_STRENGTH

    Returns: sorted list of CausalImpact (by |net_impact| descending)
    """
    graph = CAUSAL_GRAPH.get(event_class, {})
    impacts: Dict[str, CausalImpact] = {}

    # ── Step 1: apply causal graph entries ───────────────────────────────────
    for ticker, spec in graph.items():
        direction, base_strength, pathway_type, reasoning, horizon = spec

        is_named = ticker in named_entities
        entity_conf = named_entities.get(ticker, 0.0)

        # Strength: named entity gets full base strength; unnamed gets reduced
        if is_named:
            strength = base_strength * min(1.0, 0.85 + entity_conf * 0.15)
        else:
            strength = base_strength * UNNAMED_ENTITY_MULTIPLIER

        if strength < MIN_PATHWAY_STRENGTH:
            continue

        # Confidence: product of event class confidence and source credibility
        confidence = round(event_class_confidence * max(credibility, 0.3), 3)

        impacts[ticker] = CausalImpact(
            ticker=ticker,
            direction=direction,
            pathway_strength=round(strength, 4),
            pathway_type=pathway_type,
            reasoning=reasoning,
            confidence=confidence,
            impact_horizon=horizon,
            is_primary=is_named,
        )

    # ── Step 2: handle named entities not yet in the graph ───────────────────
    # This covers: company_earnings_guidance events, entity-level sanctions,
    # antitrust actions, and other events where a specific named company
    # is the direct subject but isn't pre-mapped in the default graph.
    for ticker, entity_conf in named_entities.items():
        if ticker in impacts:
            # Already handled; but boost strength since it was explicitly named
            impacts[ticker].pathway_strength = min(
                1.0, impacts[ticker].pathway_strength * (1.0 + 0.2 * entity_conf)
            )
            impacts[ticker].is_primary = True
            continue

        # Named but not in graph — generate a direct self-impact
        if event_class == "company_earnings_guidance":
            direction = _infer_earnings_direction(text)
            if direction == 0.0:
                continue  # unclear direction — skip to avoid noise
            base_strength = 0.75 * entity_conf
            pathway_type  = "direct_earnings_impact"
            reasoning     = f"{ticker} is the company directly reporting earnings/guidance in this article"
            horizon       = _H_SHORT

        elif event_class == "geopolitical_sanctions":
            direction     = -1.0
            base_strength = 0.80 * entity_conf
            pathway_type  = "direct_sanctions_impact"
            reasoning     = f"{ticker} is directly sanctioned; operations and revenue streams are constrained"
            horizon       = _H_MEDIUM

        elif event_class == "regulatory_antitrust":
            direction     = -0.70
            base_strength = 0.72 * entity_conf
            pathway_type  = "regulatory_investigation_risk"
            reasoning     = f"{ticker} is under regulatory investigation; legal costs and operational restrictions create risk"
            horizon       = _H_MEDIUM

        elif event_class == "merger_acquisition_tech":
            # Target company: positive (acquisition premium)
            # Acquirer: often slightly negative (dilution, premium paid)
            # Without more context we default to the target getting a premium signal
            direction     = 0.85
            base_strength = 0.78 * entity_conf
            pathway_type  = "acquisition_target_premium"
            reasoning     = f"{ticker} is named in M&A context; likely acquisition premium applies"
            horizon       = _H_INTRADAY

        else:
            # Named but no specific self-impact rule — skip rather than guess
            continue

        if base_strength < MIN_PATHWAY_STRENGTH:
            continue

        confidence = round(event_class_confidence * max(credibility, 0.3), 3)
        impacts[ticker] = CausalImpact(
            ticker=ticker,
            direction=direction,
            pathway_strength=round(base_strength, 4),
            pathway_type=pathway_type,
            reasoning=reasoning,
            confidence=confidence,
            impact_horizon=horizon,
            is_primary=True,
        )

    # ── Step 3: filter and sort ───────────────────────────────────────────────
    filtered = [
        imp for imp in impacts.values()
        if imp.pathway_strength >= MIN_PATHWAY_STRENGTH
    ]
    filtered.sort(key=lambda x: abs(x.net_impact), reverse=True)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_article(text: str, credibility: float = 0.5) -> CausalAnalysis:
    """
    Full causal analysis pipeline for a single article.

    Steps:
      1. Classify the economic event type
      2. Extract specifically named entities
      3. Compute causal impacts with direction and pathway strength
      4. Filter weak connections

    Returns CausalAnalysis dataclass with all results.
    """
    event_class, event_confidence = classify_economic_event(text)
    named_entities = extract_named_entities(text)

    causal_impacts = compute_causal_impacts(
        event_class=event_class,
        event_class_confidence=event_confidence,
        named_entities=named_entities,
        credibility=credibility,
        text=text,
    )

    # Choose impact horizon from the primary (named + highest strength) impact
    primary_impacts = [i for i in causal_impacts if i.is_primary]
    if primary_impacts:
        impact_horizon = primary_impacts[0].impact_horizon
    elif causal_impacts:
        impact_horizon = causal_impacts[0].impact_horizon
    else:
        impact_horizon = _H_MEDIUM

    # Overall confidence: event confidence × credibility, capped at event evidence
    overall_confidence = round(
        event_confidence * max(credibility, 0.25) *
        min(1.0, 0.5 + len(named_entities) * 0.1),
        3
    )

    # Build mechanism summary
    if event_class != "unknown" and causal_impacts:
        top = causal_impacts[0]
        mechanism_summary = (
            f"Event: {event_class.replace('_', ' ')}. "
            f"Primary causal pathway: {top.pathway_type.replace('_', ' ')} "
            f"→ {top.ticker}. {top.reasoning}"
        )
    elif event_class != "unknown":
        mechanism_summary = f"Event classified as {event_class.replace('_', ' ')} but no tracked tickers are causally affected."
    else:
        mechanism_summary = "Event class could not be determined from article content."

    return CausalAnalysis(
        event_class=event_class,
        event_class_confidence=event_confidence,
        named_entities=named_entities,
        causal_impacts=causal_impacts,
        impact_horizon=impact_horizon,
        overall_confidence=overall_confidence,
        mechanism_summary=mechanism_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event class numeric encoding for ML features (15 classes → 15 values)
# ─────────────────────────────────────────────────────────────────────────────

EVENT_CLASS_NUMERIC: Dict[str, float] = {
    # High-impact geopolitical events → higher values
    "geopolitical_taiwan_risk":        1.00,
    "export_restriction_chips":        0.93,
    "export_restriction_equipment":    0.87,
    "geopolitical_energy_risk":        0.80,
    "geopolitical_sanctions":          0.73,
    "supply_disruption_foundry":       0.67,
    "trade_policy_tariff":             0.60,
    "ai_demand_expansion":             0.53,
    "ai_demand_contraction":           0.47,
    "supply_disruption_memory":        0.40,
    "domestic_manufacturing_subsidy":  0.33,
    "macro_monetary_hawkish":          0.27,
    "macro_monetary_dovish":           0.20,
    "company_earnings_guidance":       0.13,
    "regulatory_antitrust":            0.10,
    "merger_acquisition_tech":         0.07,
    "unknown":                         0.05,
}


def get_event_class_num(event_class: str) -> float:
    """Return normalized numeric encoding of event class for ML feature vector."""
    return EVENT_CLASS_NUMERIC.get(event_class, 0.05)
