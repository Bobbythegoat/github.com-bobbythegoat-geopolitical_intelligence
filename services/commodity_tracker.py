"""
services/commodity_tracker.py
-----------------------------
Strategic Commodity & Industrial-Input Tracker.

Mission
-------
Bolts onto the existing event/article pipeline. For every ingested article:
  1. Scans text for tier-1/2/3 commodity mentions and qualifying context
     (supply disruption, export control, stockpiling, capacity, policy).
  2. Records a CommoditySignal with direction, severity, confidence.
  3. Writes EventTickerImpact rows against the commodity-as-entity, which
     the existing second_order engine already knows how to propagate
     downstream to exposed tickers via RelationshipEdge.
  4. Adjusts the parent Event's geography_tags and credibility-adjacent
     metadata where the commodity carries strong policy/geography signal.

Why bolt-on instead of bolt-in
------------------------------
The existing causal_engine emits ticker impacts directly. Adding a parallel
commodity layer means commodity shocks (uranium spike, gallium export ban,
neon shortage) feed the same StockScore math without rewriting any of the
direct-impact logic.  The commodity registry → RelationshipEdge seeding does
the work of mapping a commodity move to every exposed name.

Tier definitions follow the product spec:
  Tier 1: copper, nat gas, uranium, aluminum, lithium, nickel
          — liquid markets, LME/futures data, COT positioning
  Tier 2: gallium, germanium, indium, rare earths, graphite
          — strategic choke points, policy-sensitive
  Tier 3: neon, argon, krypton, xenon, photoresists, etchants,
          CMP slurries, ultra-pure water
          — semiconductor hidden bottlenecks
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

import database as db


# ---------------------------------------------------------------------------
# Commodity registry
# ---------------------------------------------------------------------------
# Each entry is a dict so it can be diffed/extended easily.  Keywords must be
# lowercase substrings; the scanner does normalised substring matching.

COMMODITY_REGISTRY: List[Dict] = [
    # ============================================================
    # TIER 1 — high importance, easy to track
    # ============================================================
    {
        "symbol": "COPPER", "name": "Copper", "tier": 1,
        "category": "base_metal", "unit": "USD/lb",
        "proxy_ticker": "CPER",
        "supplier_concentration": 0.45,   # Chile + Peru dominate
        "policy_sensitivity":     0.40,
        "ai_demand_linkage":      0.70,   # grid + data-center copper
        "primary_geographies": "Chile,Peru,DRC,China",
        "exposed_tickers": "FCX,SCCO,BHP,RIO,GLEN",
        "keywords": [
            "copper", "lme copper", "copper price", "copper smelter",
            "copper concentrate", "codelco", "escondida", "antofagasta",
            "freeport-mcmoran", "chile copper",
        ],
    },
    {
        "symbol": "NATGAS", "name": "Natural Gas", "tier": 1,
        "category": "energy", "unit": "USD/mmBtu",
        "proxy_ticker": "UNG",
        "supplier_concentration": 0.30,
        "policy_sensitivity":     0.75,
        "ai_demand_linkage":      0.85,   # data-center power
        "primary_geographies": "US,Russia,Qatar,Norway,Australia",
        "exposed_tickers": "LNG,EQT,CHK,WMB,KMI,XOM,CVX,SHEL",
        "keywords": [
            "natural gas", "lng", "henry hub", "ttf", "nordstream",
            "nord stream", "gas pipeline", "gas supply", "liquefied natural gas",
            "gas storage", "qatar lng", "freeport lng",
        ],
    },
    {
        "symbol": "URANIUM", "name": "Uranium", "tier": 1,
        "category": "nuclear_fuel", "unit": "USD/lb U3O8",
        "proxy_ticker": "URA",
        "supplier_concentration": 0.65,   # Kazakhstan + Russia conversion
        "policy_sensitivity":     0.80,
        "ai_demand_linkage":      0.80,   # nuclear baseload for AI
        "primary_geographies": "Kazakhstan,Canada,Australia,Russia,Niger",
        "exposed_tickers": "CCJ,UEC,DNN,NXE,URNM,URA,LEU",
        "keywords": [
            "uranium", "u3o8", "yellowcake", "enriched uranium", "haleu",
            "kazatomprom", "cameco", "nuclear fuel", "nuclear reactor restart",
            "smr", "small modular reactor", "centrus",
        ],
    },
    {
        "symbol": "ALUMINUM", "name": "Aluminum", "tier": 1,
        "category": "base_metal", "unit": "USD/tonne",
        "proxy_ticker": "JJU",
        "supplier_concentration": 0.55,   # China >55% of smelting
        "policy_sensitivity":     0.55,
        "ai_demand_linkage":      0.40,   # data-center heat exchangers, server chassis
        "primary_geographies": "China,Russia,India,Canada,UAE",
        "exposed_tickers": "AA,CENX,RIO,NHYDY,CSTM",
        "keywords": [
            "aluminum", "aluminium", "lme aluminum", "alumina", "bauxite",
            "rusal", "alcoa", "norsk hydro", "smelter", "primary aluminum",
        ],
    },
    {
        "symbol": "LITHIUM", "name": "Lithium", "tier": 1,
        "category": "battery_metal", "unit": "USD/tonne LCE",
        "proxy_ticker": "LIT",
        "supplier_concentration": 0.60,   # Australia + Chile + China refining
        "policy_sensitivity":     0.70,
        "ai_demand_linkage":      0.30,   # mostly EV but some grid storage
        "primary_geographies": "Australia,Chile,China,Argentina,Bolivia",
        "exposed_tickers": "ALB,SQM,LAC,LTHM,PLL,TSLA",
        "keywords": [
            "lithium", "lithium carbonate", "lithium hydroxide", "spodumene",
            "lithium brine", "albemarle", "sqm", "ganfeng", "tianqi",
            "lithium price", "lithium mining",
        ],
    },
    {
        "symbol": "NICKEL", "name": "Nickel", "tier": 1,
        "category": "battery_metal", "unit": "USD/tonne",
        "proxy_ticker": "JJN",
        "supplier_concentration": 0.55,   # Indonesia dominates
        "policy_sensitivity":     0.60,
        "ai_demand_linkage":      0.20,
        "primary_geographies": "Indonesia,Philippines,Russia,Australia",
        "exposed_tickers": "BHP,VALE,GLEN,NILSY,NIM",
        "keywords": [
            "nickel", "lme nickel", "nickel sulphate", "ferronickel",
            "indonesia nickel", "norilsk", "nornickel", "class 1 nickel",
            "nickel pig iron",
        ],
    },

    # ============================================================
    # TIER 2 — strategic choke points
    # ============================================================
    {
        "symbol": "GALLIUM", "name": "Gallium", "tier": 2,
        "category": "minor_metal", "unit": "USD/kg",
        "proxy_ticker": None,
        "supplier_concentration": 0.95,   # China ~98% of refined supply
        "policy_sensitivity":     0.95,
        "ai_demand_linkage":      0.65,   # GaN power semis, RF, data-center power
        "primary_geographies": "China",
        "exposed_tickers": "AXTI,IIVI,WOLF,NVDA,QCOM",
        "keywords": [
            "gallium", "gallium nitride", "gan", "gallium arsenide", "gaas",
            "gallium export", "gallium restriction", "china gallium",
            "gallium ban",
        ],
    },
    {
        "symbol": "GERMANIUM", "name": "Germanium", "tier": 2,
        "category": "minor_metal", "unit": "USD/kg",
        "proxy_ticker": None,
        "supplier_concentration": 0.90,
        "policy_sensitivity":     0.95,
        "ai_demand_linkage":      0.55,   # optical fiber, IR, advanced packaging
        "primary_geographies": "China,Russia",
        "exposed_tickers": "IIVI,LITE,COHR,STM",
        "keywords": [
            "germanium", "germanium export", "germanium restriction",
            "china germanium", "germanium ban", "optical fiber germanium",
        ],
    },
    {
        "symbol": "INDIUM", "name": "Indium", "tier": 2,
        "category": "minor_metal", "unit": "USD/kg",
        "proxy_ticker": None,
        "supplier_concentration": 0.85,
        "policy_sensitivity":     0.80,
        "ai_demand_linkage":      0.40,
        "primary_geographies": "China,South Korea,Japan",
        "exposed_tickers": "IIVI,COHR,AMAT",
        "keywords": [
            "indium", "indium tin oxide", "ito", "indium export",
            "indium phosphide", "inp wafer",
        ],
    },
    {
        "symbol": "RAREEARTH", "name": "Rare Earth Elements", "tier": 2,
        "category": "rare_earth", "unit": "USD/kg NdPr",
        "proxy_ticker": "REMX",
        "supplier_concentration": 0.85,   # China dominates mining + nearly all refining
        "policy_sensitivity":     0.95,
        "ai_demand_linkage":      0.60,   # magnets, motors, data-center cooling fans
        "primary_geographies": "China,Myanmar,Australia,US",
        "exposed_tickers": "MP,LYC.AX,REE,UUUU,TMRC,VALE",
        "keywords": [
            "rare earth", "rare earths", "ndpr", "neodymium", "praseodymium",
            "dysprosium", "terbium", "samarium", "rare earth export",
            "china rare earth", "rare earth magnet", "ree", "mp materials",
            "lynas",
        ],
    },
    {
        "symbol": "GRAPHITE", "name": "Graphite", "tier": 2,
        "category": "battery_material", "unit": "USD/tonne",
        "proxy_ticker": None,
        "supplier_concentration": 0.80,   # China ~80% of anode-grade
        "policy_sensitivity":     0.85,
        "ai_demand_linkage":      0.25,   # mostly EV anodes
        "primary_geographies": "China,Mozambique,Madagascar",
        "exposed_tickers": "NGC,SYR.AX,WWR,NMG",
        "keywords": [
            "graphite", "natural graphite", "synthetic graphite",
            "spherical graphite", "anode graphite", "graphite export",
            "china graphite", "battery anode",
        ],
    },

    # ============================================================
    # TIER 3 — semiconductor hidden bottlenecks
    # ============================================================
    {
        "symbol": "NEON", "name": "Neon Gas", "tier": 3,
        "category": "industrial_gas", "unit": "USD/m3",
        "proxy_ticker": None,
        "supplier_concentration": 0.70,   # Ukraine historically ~50% of semi-grade
        "policy_sensitivity":     0.85,
        "ai_demand_linkage":      0.75,   # lithography lasers
        "primary_geographies": "Ukraine,Russia,China,US",
        "exposed_tickers": "ASML,LRCX,AMAT,KLAC,TSM,INTC",
        "keywords": [
            "neon gas", "semiconductor neon", "neon shortage",
            "ukraine neon", "ingas", "cryoin", "lithography gas",
        ],
    },
    {
        "symbol": "ARGON", "name": "Argon Gas", "tier": 3,
        "category": "industrial_gas", "unit": "USD/m3",
        "proxy_ticker": None,
        "supplier_concentration": 0.40,
        "policy_sensitivity":     0.45,
        "ai_demand_linkage":      0.60,
        "primary_geographies": "global",
        "exposed_tickers": "APD,LIN,AI,TSM,INTC",
        "keywords": [
            "argon gas", "industrial argon", "semiconductor argon",
            "argon shortage",
        ],
    },
    {
        "symbol": "KRYPTON", "name": "Krypton Gas", "tier": 3,
        "category": "industrial_gas", "unit": "USD/m3",
        "proxy_ticker": None,
        "supplier_concentration": 0.75,
        "policy_sensitivity":     0.80,
        "ai_demand_linkage":      0.70,
        "primary_geographies": "Ukraine,Russia,China",
        "exposed_tickers": "ASML,AMAT,LRCX,KLAC,TSM,INTC",
        "keywords": [
            "krypton gas", "krypton shortage", "semiconductor krypton",
        ],
    },
    {
        "symbol": "XENON", "name": "Xenon Gas", "tier": 3,
        "category": "industrial_gas", "unit": "USD/m3",
        "proxy_ticker": None,
        "supplier_concentration": 0.75,
        "policy_sensitivity":     0.80,
        "ai_demand_linkage":      0.75,
        "primary_geographies": "Ukraine,Russia,China",
        "exposed_tickers": "ASML,AMAT,LRCX,KLAC,TSM,INTC",
        "keywords": [
            "xenon gas", "xenon shortage", "semiconductor xenon",
        ],
    },
    {
        "symbol": "PHOTORESIST", "name": "Photoresist", "tier": 3,
        "category": "specialty_chemical", "unit": "USD/kg",
        "proxy_ticker": None,
        "supplier_concentration": 0.90,   # JSR, Tokyo Ohka, Shin-Etsu, Sumitomo
        "policy_sensitivity":     0.85,
        "ai_demand_linkage":      0.85,
        "primary_geographies": "Japan,South Korea,US",
        "exposed_tickers": "JSR,SHECY,TOELY,TSM,INTC,ASML",
        "keywords": [
            "photoresist", "euv photoresist", "duv photoresist",
            "jsr corp", "tokyo ohka", "shin-etsu", "sumitomo chemical",
            "resist supply", "photoresist shortage",
        ],
    },
    {
        "symbol": "ETCHANT", "name": "Etchant Chemicals", "tier": 3,
        "category": "specialty_chemical", "unit": "USD/kg",
        "proxy_ticker": None,
        "supplier_concentration": 0.70,
        "policy_sensitivity":     0.75,
        "ai_demand_linkage":      0.80,
        "primary_geographies": "Japan,South Korea,US,Germany",
        "exposed_tickers": "ENTG,KMG,LRCX,AMAT,TSM,INTC",
        "keywords": [
            "etchant", "hydrofluoric acid", "wet etch chemical",
            "high-purity hf", "entegris", "specialty fluorochemical",
        ],
    },
    {
        "symbol": "CMP_SLURRY", "name": "CMP Slurry", "tier": 3,
        "category": "specialty_chemical", "unit": "USD/L",
        "proxy_ticker": None,
        "supplier_concentration": 0.75,
        "policy_sensitivity":     0.65,
        "ai_demand_linkage":      0.80,
        "primary_geographies": "US,Japan,Korea",
        "exposed_tickers": "CCMP,ENTG,FUJIY,TSM,INTC",
        "keywords": [
            "cmp slurry", "chemical mechanical planarization",
            "polishing slurry", "cabot microelectronics", "ccmp",
        ],
    },
    {
        "symbol": "UPW", "name": "Ultra-Pure Water", "tier": 3,
        "category": "infrastructure_input", "unit": "USD/m3",
        "proxy_ticker": None,
        "supplier_concentration": 0.30,
        "policy_sensitivity":     0.55,
        "ai_demand_linkage":      0.90,   # fab + data-center cooling
        "primary_geographies": "Taiwan,Arizona,Korea",
        "exposed_tickers": "TSM,INTC,XYL,VLTO,WTS",
        "keywords": [
            "ultra-pure water", "upw", "fab water", "taiwan drought",
            "arizona water", "semiconductor water", "water restriction fab",
        ],
    },
]


# ---------------------------------------------------------------------------
# Signal-type keyword maps (qualify what kind of commodity event was detected)
# ---------------------------------------------------------------------------

SIGNAL_PATTERNS: List[Tuple[str, int, float, List[str]]] = [
    # (signal_type, direction (+1 bullish for commodity / -1 bearish), severity, keywords)
    ("export_control",     +1, 0.85, [
        "export ban", "export restriction", "export control", "export curb",
        "export license", "outbound investment ban", "trade restriction",
    ]),
    ("supply_disruption",  +1, 0.80, [
        "shortage", "supply disruption", "supply shock", "force majeure",
        "production halt", "plant fire", "mine strike", "smelter outage",
        "refinery shutdown", "pipeline rupture",
    ]),
    ("stockpiling",        +1, 0.55, [
        "stockpile", "strategic reserve", "national reserve", "spr release",
        "inventory drawdown", "stockpiling",
    ]),
    ("policy_action",      +1, 0.55, [
        "sanction", "sanctions", "tariff", "subsidy", "industrial policy",
        "chips act", "ira", "executive order", "white house",
    ]),
    ("capacity_expansion", -1, 0.45, [
        "new mine", "capacity expansion", "new fab", "groundbreaking",
        "production ramp", "ramp up", "additional capacity",
    ]),
    ("demand_shock",       +1, 0.60, [
        "ai capex", "data center buildout", "hyperscaler spend",
        "ev demand surge", "demand surge", "record demand",
    ]),
    ("geopolitical_risk",  +1, 0.65, [
        "taiwan strait", "south china sea", "blockade", "naval drill",
        "missile test", "invasion", "conflict escalation",
    ]),
    ("price_move",          0, 0.30, [
        "hit record", "all-time high", "multi-year high", "price spike",
        "price plunge", "price crash",
    ]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _commodity_by_symbol_cache(session: Session) -> Dict[str, db.Commodity]:
    rows = session.query(db.Commodity).all()
    return {r.symbol: r for r in rows}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_commodities(session: Session) -> int:
    """
    Upsert the COMMODITY_REGISTRY into the commodities table.
    Idempotent — existing symbols are refreshed (lightweight upsert).
    Returns number of new rows inserted.
    """
    inserted = 0
    for entry in COMMODITY_REGISTRY:
        row = (session.query(db.Commodity)
                      .filter_by(symbol=entry["symbol"])
                      .first())
        kw_csv = ",".join(entry["keywords"])
        if row is None:
            row = db.Commodity(
                symbol=entry["symbol"],
                name=entry["name"],
                tier=entry["tier"],
                category=entry.get("category"),
                unit=entry.get("unit"),
                proxy_ticker=entry.get("proxy_ticker"),
                supplier_concentration=entry.get("supplier_concentration", 0.0),
                policy_sensitivity=entry.get("policy_sensitivity", 0.0),
                ai_demand_linkage=entry.get("ai_demand_linkage", 0.0),
                primary_geographies=entry.get("primary_geographies", ""),
                exposed_tickers=entry.get("exposed_tickers", ""),
                keywords=kw_csv,
            )
            session.add(row)
            inserted += 1
        else:
            # Refresh in case registry changed
            row.tier                   = entry["tier"]
            row.category               = entry.get("category")
            row.unit                   = entry.get("unit")
            row.proxy_ticker           = entry.get("proxy_ticker")
            row.supplier_concentration = entry.get("supplier_concentration", 0.0)
            row.policy_sensitivity     = entry.get("policy_sensitivity", 0.0)
            row.ai_demand_linkage      = entry.get("ai_demand_linkage", 0.0)
            row.primary_geographies    = entry.get("primary_geographies", "")
            row.exposed_tickers        = entry.get("exposed_tickers", "")
            row.keywords               = kw_csv
    session.commit()
    return inserted


def seed_commodity_relationship_edges(session: Session) -> int:
    """
    For each commodity, create RelationshipEdge rows from the commodity symbol
    (as `from_entity`) to each exposed ticker.  This lets the existing
    second_order engine propagate commodity shocks downstream without any
    changes to that engine.

    Edge weight is a function of supplier concentration + policy sensitivity
    + AI-demand linkage, capped at 0.85 so it remains below directly-named
    edges like TSMC → NVDA (0.90).
    """
    inserted = 0
    for entry in COMMODITY_REGISTRY:
        sym       = entry["symbol"]
        exposed   = [t.strip() for t in entry.get("exposed_tickers", "").split(",") if t.strip()]
        if not exposed:
            continue

        # Weight blend — higher choke + policy + AI link → stronger transmission
        w = (
            0.45 * entry.get("supplier_concentration", 0.0) +
            0.35 * entry.get("policy_sensitivity", 0.0) +
            0.20 * entry.get("ai_demand_linkage", 0.0)
        )
        w = round(min(max(w, 0.20), 0.85), 2)

        # Tier-3 specialty inputs decay slowest (fab inventory ~weeks);
        # tier-1 liquid commodities decay fastest (futures repricing fast).
        decay_days = {1: 3, 2: 7, 3: 14}.get(entry["tier"], 5)

        for ticker_sym in exposed:
            # Only seed edges to tickers we actually track (avoids polluting
            # the graph with phantom symbols).
            tk = session.get(db.Ticker, ticker_sym)
            if tk is None:
                continue
            exists = (session.query(db.RelationshipEdge)
                             .filter_by(from_entity=sym,
                                        to_entity=ticker_sym,
                                        relationship_type="commodity_input")
                             .first())
            if exists is not None:
                continue
            session.add(db.RelationshipEdge(
                from_entity=sym,
                to_entity=ticker_sym,
                relationship_type="commodity_input",
                weight=w,
                time_decay_days=decay_days,
                direction="negative",   # commodity supply shock is bad for downstream users
            ))
            inserted += 1
    session.commit()
    return inserted


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_text(headline: str, content: str) -> List[Dict]:
    """
    Return list of detected commodity hits with qualifying signal type(s).

    Each hit:
      {
        "commodity_symbol": str,
        "matched_keywords": [str],
        "signals": [{"signal_type", "direction", "severity"}],
        "confidence": float,
      }

    A commodity only registers a hit if (a) at least one of its keywords
    matches AND (b) the article also matches at least one SIGNAL_PATTERNS
    bucket — i.e. mere mention of "copper" is not a signal; "copper export
    restriction" or "copper supply disruption" is.
    """
    text = _normalise(f"{headline} {content}")
    if not text.strip():
        return []

    # Pre-compute matched signal patterns once (shared across all commodities)
    matched_signals: List[Tuple[str, int, float]] = []
    for sig_type, direction, severity, kws in SIGNAL_PATTERNS:
        if any(kw in text for kw in kws):
            matched_signals.append((sig_type, direction, severity))
    if not matched_signals:
        return []

    hits: List[Dict] = []
    for entry in COMMODITY_REGISTRY:
        matched_kws = [kw for kw in entry["keywords"] if kw in text]
        if not matched_kws:
            continue

        # Confidence rises with number of distinct commodity keywords matched
        # and with the supplier concentration (concentrated supply = higher
        # signal value because disruption is more disruptive)
        kw_strength = min(len(matched_kws) / 3.0, 1.0)
        confidence  = round(0.50 + 0.30 * kw_strength
                                 + 0.20 * entry.get("supplier_concentration", 0.0), 3)
        confidence  = min(confidence, 0.95)

        hits.append({
            "commodity_symbol": entry["symbol"],
            "matched_keywords": matched_kws,
            "signals": [
                {"signal_type": s[0], "direction": s[1], "severity": s[2]}
                for s in matched_signals
            ],
            "confidence": confidence,
        })
    return hits


# ---------------------------------------------------------------------------
# Pipeline integration — called from processing.process_article
# ---------------------------------------------------------------------------

def link_event_to_commodities(article_id: int, session: Session) -> int:
    """
    Scan the article, write CommoditySignal rows, and add EventTickerImpact
    entries against the commodity-as-entity so the second_order engine
    propagates the shock to exposed tickers automatically.

    Returns the number of signals recorded.
    """
    article = session.get(db.Article, article_id)
    if article is None or article.event_id is None:
        return 0

    event = session.get(db.Event, article.event_id)
    if event is None:
        return 0

    hits = scan_text(article.headline or "", article.content or "")
    if not hits:
        return 0

    credibility = float(event.credibility_score or 0.5)
    signals_written = 0

    for hit in hits:
        sym = hit["commodity_symbol"]

        # Pick the dominant signal — highest severity wins
        dominant = max(hit["signals"], key=lambda s: s["severity"])
        sig_type = dominant["signal_type"]
        direction = int(dominant["direction"])
        severity  = float(dominant["severity"])
        conf      = float(hit["confidence"]) * credibility   # blend with event credibility

        # ── Record CommoditySignal ──────────────────────────────────────────
        session.add(db.CommoditySignal(
            commodity_symbol=sym,
            event_id=event.event_id,
            article_id=article.id,
            signal_type=sig_type,
            direction=direction,
            severity=severity,
            confidence=round(conf, 4),
            summary=f"{sig_type} on {sym}: {', '.join(hit['matched_keywords'][:3])}",
        ))
        signals_written += 1

        # ── Bridge into second-order graph ──────────────────────────────────
        # The graph uses ticker symbols as nodes, but RelationshipEdge already
        # accepts arbitrary entity names (e.g. "Taiwan", "AI_capex"). We add
        # a synthetic EventTickerImpact whose "ticker" column holds the
        # commodity symbol so the existing second_order engine picks it up.
        # NOTE: We only do this if the symbol has been added to the Ticker
        # table as a phantom row (see _ensure_commodity_phantom_tickers).
        _ensure_commodity_phantom_ticker(sym, session)

        # Direction-aware net impact: bullish commodity move = negative for
        # downstream consumers (most exposed_tickers), neutral for some.
        # Magnitude = severity × confidence × credibility (capped at 1.0).
        net = round(direction * severity * conf, 4)
        if abs(net) < 0.05:
            continue
        existing = (session.query(db.EventTickerImpact)
                          .filter_by(event_id=event.event_id, ticker=sym)
                          .first())
        if existing:
            existing.impact_score = round(0.6 * net + 0.4 * existing.impact_score, 4)
        else:
            session.add(db.EventTickerImpact(
                event_id=event.event_id,
                ticker=sym,
                impact_score=net,
            ))

    session.commit()
    return signals_written


def _ensure_commodity_phantom_ticker(symbol: str, session: Session) -> None:
    """
    Insert a phantom Ticker row for the commodity symbol so it can be a node
    in EventTickerImpact (the FK requires a tickers row).  These phantom
    rows are marked sector='Commodity' so they can be filtered out of
    stock-screen UIs that should only show real equities.
    """
    if session.get(db.Ticker, symbol) is not None:
        return
    # Look up the registry entry for nicer metadata
    name = symbol
    for entry in COMMODITY_REGISTRY:
        if entry["symbol"] == symbol:
            name = entry["name"]
            break
    session.add(db.Ticker(
        ticker=symbol,
        company_name=f"{name} (commodity)",
        sector="Commodity",
    ))
    # commit handled by caller


# ---------------------------------------------------------------------------
# Public API for routers
# ---------------------------------------------------------------------------

def commodity_heatmap(session: Session, hours: int = 72) -> List[Dict]:
    """
    Return a heatmap-style view: for each commodity, aggregate recent signal
    activity over the last `hours` window.
    """
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    rows = (session.query(db.CommoditySignal)
                  .filter(db.CommoditySignal.timestamp >= cutoff)
                  .all())

    by_sym: Dict[str, List[db.CommoditySignal]] = {}
    for r in rows:
        by_sym.setdefault(r.commodity_symbol, []).append(r)

    out: List[Dict] = []
    all_commods = _commodity_by_symbol_cache(session)
    for sym, c in all_commods.items():
        sigs = by_sym.get(sym, [])
        if sigs:
            net = sum(s.direction * s.severity * s.confidence for s in sigs)
            max_sev = max(s.severity for s in sigs)
            sig_count = len(sigs)
            top_signal = max(sigs, key=lambda s: s.severity * s.confidence).signal_type
        else:
            net = 0.0
            max_sev = 0.0
            sig_count = 0
            top_signal = None
        out.append({
            "symbol": sym,
            "name": c.name,
            "tier": c.tier,
            "category": c.category,
            "proxy_ticker": c.proxy_ticker,
            "supplier_concentration": c.supplier_concentration,
            "policy_sensitivity": c.policy_sensitivity,
            "ai_demand_linkage": c.ai_demand_linkage,
            "net_signal": round(net, 4),
            "max_severity": round(max_sev, 4),
            "signal_count": sig_count,
            "top_signal_type": top_signal,
        })
    # Sort: active first (signal_count desc), then by tier, then policy_sensitivity
    out.sort(key=lambda r: (-r["signal_count"], r["tier"], -r["policy_sensitivity"]))
    return out
