"""
services/second_order.py
-------------------------
Second-Order Impact Engine (Upgrade Spec §4.2)

Detects indirect beneficiaries and indirect losers beyond the obvious
headline target.  Second-order effects are often underpriced because
market participants focus on the first-order name.

Core concept
------------
Build a transmission graph linking:
  countries → sectors, sectors → companies, companies → suppliers/customers,
  policies → affected companies, geographies → supply-chain dependents

For each event, generate:
  DirectImpact(k|event)   : ticker k is directly mentioned or keyword-matched
  IndirectImpact(j|event) : ticker j is affected via relationships to k

Formula (from spec)
-------------------
    IndirectImpact(j|event) = Σ_k [
        DirectImpact(k|event)
        × RelationshipWeight(j,k)
        × TimeDecay(j,k)
        × Confidence(event)
    ]

TimeDecay: effects attenuate over time.  The decay function here uses an
exponential where the half-life equals the relationship's time_decay_days.

Relationship seed data
-----------------------
The system ships with a curated set of semiconductor, energy, and defense
relationships.  These are upserted into the relationship_edges table on
first run and can be extended via API or direct DB editing.

The seed relationship graph covers the CLAUDE.md primary semiconductor universe.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

import database as db

# ---------------------------------------------------------------------------
# Relationship seed graph
# ---------------------------------------------------------------------------
# Format: (from_entity, to_entity, relationship_type, weight, time_decay_days, direction)
# Direction: 'positive' (indirect beneficiary), 'negative' (indirect loser),
#            'uncertain'

SEED_RELATIONSHIPS: List[Tuple[str, str, str, float, int, str]] = [

    # ── Semiconductor supply chain ─────────────────────────────────────────
    # TSMC is the primary foundry for most fabless chip companies
    ("TSMC", "NVDA",  "customer",    0.90, 5,  "negative"),  # TSMC disruption hurts NVDA
    ("TSMC", "AMD",   "customer",    0.85, 5,  "negative"),
    ("TSMC", "AAPL",  "customer",    0.90, 5,  "negative"),
    ("TSMC", "QCOM",  "customer",    0.85, 5,  "negative"),
    ("TSMC", "MRVL",  "customer",    0.80, 5,  "negative"),
    ("TSMC", "ARM",   "customer",    0.70, 7,  "negative"),
    ("TSMC", "AVGO",  "customer",    0.75, 5,  "negative"),
    ("TSMC", "MCHP",  "customer",    0.60, 7,  "negative"),
    ("TSMC", "SNPS",  "customer",    0.50, 10, "negative"),
    ("TSMC", "CDNS",  "customer",    0.50, 10, "negative"),
    # ASML makes EUV lithography machines used by TSMC, Samsung, Intel
    ("ASML", "TSM",   "equipment",   0.85, 10, "negative"),  # ASML restriction hurts TSMC
    ("ASML", "INTC",  "equipment",   0.70, 10, "negative"),
    # Applied Materials and Lam Research supply deposition / etch equipment
    ("AMAT", "TSM",   "equipment",   0.80, 10, "negative"),
    ("LRCX", "TSM",   "equipment",   0.80, 10, "negative"),
    ("KLAC", "TSM",   "equipment",   0.75, 10, "negative"),
    # When NVDA wins, adjacent AI infrastructure names benefit
    ("NVDA", "SMCI",  "customer",    0.70, 3,  "positive"),  # SMCI builds NVDA-based servers
    ("NVDA", "AVGO",  "competitor",  0.40, 7,  "negative"),  # AVGO custom AI chips compete
    ("NVDA", "AMD",   "competitor",  0.50, 5,  "negative"),  # AMD competes in GPU/AI
    # Memory
    ("MU",   "NVDA",  "supplier",    0.60, 5,  "positive"),  # MU HBM goes into NVDA
    ("MU",   "AMD",   "supplier",    0.50, 5,  "positive"),
    # Taiwan geopolitical risk propagation
    ("Taiwan", "TSM",  "geography",  0.95, 3,  "negative"),
    ("Taiwan", "NVDA", "geography",  0.75, 5,  "negative"),
    ("Taiwan", "AAPL", "geography",  0.65, 7,  "negative"),
    ("Taiwan", "AMD",  "geography",  0.65, 7,  "negative"),
    # China export controls → semiconductor equipment names lose revenue
    ("China", "AMAT",  "geography",  0.65, 5,  "negative"),
    ("China", "LRCX",  "geography",  0.65, 5,  "negative"),
    ("China", "KLAC",  "geography",  0.60, 5,  "negative"),
    ("China", "ASML",  "geography",  0.70, 5,  "negative"),
    # Hyperscaler capex is a direct demand driver for NVDA/AMD GPUs
    ("AI_capex", "NVDA", "demand",   0.90, 3,  "positive"),
    ("AI_capex", "AMD",  "demand",   0.70, 3,  "positive"),
    ("AI_capex", "AVGO", "demand",   0.60, 5,  "positive"),  # custom silicon
    ("AI_capex", "MU",   "demand",   0.55, 5,  "positive"),  # HBM demand
    ("AI_capex", "SMCI", "demand",   0.75, 3,  "positive"),

    # ── Energy ────────────────────────────────────────────────────────────
    ("Middle_East_conflict", "XOM",  "geography", 0.70, 3,  "positive"),
    ("Middle_East_conflict", "CVX",  "geography", 0.65, 3,  "positive"),
    ("Middle_East_conflict", "GLD",  "geography", 0.60, 3,  "positive"),
    ("Iran_sanctions",       "XOM",  "policy",    0.55, 7,  "positive"),
    ("Iran_sanctions",       "CVX",  "policy",    0.50, 7,  "positive"),

    # ── Defense ───────────────────────────────────────────────────────────
    ("Ukraine_conflict", "LMT", "policy",  0.80, 3, "positive"),
    ("Ukraine_conflict", "RTX", "policy",  0.80, 3, "positive"),
    ("Ukraine_conflict", "BA",  "policy",  0.60, 5, "positive"),
    ("NATO_spending",    "LMT", "policy",  0.75, 7, "positive"),
    ("NATO_spending",    "RTX", "policy",  0.70, 7, "positive"),
]


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def seed_relationship_graph(session: Session) -> int:
    """
    Upsert the seed relationship graph into the database.
    Safe to call repeatedly — skips existing edges.
    Returns number of edges inserted.
    """
    inserted = 0
    for (frm, to, rel_type, weight, decay, direction) in SEED_RELATIONSHIPS:
        existing = (
            session.query(db.RelationshipEdge)
            .filter_by(from_entity=frm, to_entity=to, relationship_type=rel_type)
            .first()
        )
        if existing is None:
            session.add(db.RelationshipEdge(
                from_entity=frm,
                to_entity=to,
                relationship_type=rel_type,
                weight=weight,
                time_decay_days=decay,
                direction=direction,
            ))
            inserted += 1
    session.commit()
    return inserted


# ---------------------------------------------------------------------------
# Time decay function
# ---------------------------------------------------------------------------

def _time_decay(time_decay_days: int, elapsed_hours: float = 0.0) -> float:
    """
    Exponential decay: effect halves every `time_decay_days` days.
    elapsed_hours : how many hours since the event was first detected.
    Returns a multiplier in (0, 1].
    """
    if elapsed_hours <= 0:
        return 1.0
    half_life_hours = time_decay_days * 24.0
    return math.exp(-math.log(2) * elapsed_hours / half_life_hours)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_indirect_impacts(
    event_id: int,
    session: Session,
) -> List[Dict]:
    """
    Compute second-order (indirect) impacts for all tickers related to an event.

    For each directly impacted ticker k, look up all relationship edges where
    k is the from_entity and apply the indirect impact formula.

    Returns a list of dicts:
      [{"ticker": str, "indirect_impact_score": float,
        "relationship_type": str, "from_entity": str,
        "confidence": float, "time_horizon_days": int}, ...]
    """
    event = session.get(db.Event, event_id)
    if event is None:
        return []

    # Elapsed hours since event was detected
    elapsed = (datetime.utcnow() - event.timestamp).total_seconds() / 3600.0

    # Gather direct impacts
    direct_impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .all()
    )
    if not direct_impacts:
        return []

    credibility = event.credibility_score or 0.5
    direct_map = {imp.ticker: imp.impact_score for imp in direct_impacts}

    # Build a map of geography/theme tags for non-ticker entity matching
    geo_tags = set((event.geography_tags or "").split(",")) - {""}

    # Gather all relationship edges where from_entity is:
    #   (a) a directly impacted ticker
    #   (b) a geography tag in the event
    relevant_entities = set(direct_map.keys()) | geo_tags

    edges = (
        session.query(db.RelationshipEdge)
        .filter(db.RelationshipEdge.from_entity.in_(relevant_entities))
        .all()
    )

    # Accumulate indirect impacts by target ticker
    indirect: Dict[str, Dict] = {}

    for edge in edges:
        # Skip if target is already a direct impact (already scored)
        if edge.to_entity in direct_map:
            continue

        # Get the direct impact of the from_entity
        if edge.from_entity in direct_map:
            direct_score = direct_map[edge.from_entity]
        else:
            # Geography/theme entity: estimate as moderate negative (-0.4) or positive
            direct_score = -0.4 if edge.direction == "negative" else 0.4

        # Apply the indirect impact formula
        decay      = _time_decay(edge.time_decay_days, elapsed)
        sign       = -1.0 if edge.direction == "negative" else 1.0
        ind_score  = abs(direct_score) * edge.weight * decay * credibility * sign

        # Ensure ticker exists in our watchlist
        ticker_row = session.get(db.Ticker, edge.to_entity)
        if ticker_row is None:
            continue

        if edge.to_entity not in indirect:
            indirect[edge.to_entity] = {
                "ticker":               edge.to_entity,
                "indirect_impact_score": 0.0,
                "relationship_type":    edge.relationship_type,
                "from_entity":          edge.from_entity,
                "confidence":           round(credibility * edge.weight, 3),
                "time_horizon_days":    edge.time_decay_days,
            }

        # Accumulate (sum contributions from multiple paths)
        indirect[edge.to_entity]["indirect_impact_score"] += ind_score

    # Clamp and round final scores
    results = []
    for ticker, data in indirect.items():
        data["indirect_impact_score"] = round(
            max(-1.0, min(1.0, data["indirect_impact_score"])), 4
        )
        results.append(data)

    # Sort by absolute indirect impact descending
    results.sort(key=lambda x: abs(x["indirect_impact_score"]), reverse=True)
    return results


def update_stock_indirect_scores(event_id: int, session: Session) -> None:
    """
    Compute indirect impacts and persist them to StockScore.indirect_impact_score.
    Blends new indirect score with existing value using EMA (weight 0.5).
    """
    impacts = compute_indirect_impacts(event_id, session)
    for item in impacts:
        ticker = item["ticker"]
        new_score = item["indirect_impact_score"]
        score_row = session.query(db.StockScore).filter_by(ticker=ticker).first()
        if score_row:
            old = score_row.indirect_impact_score or 0.0
            score_row.indirect_impact_score = round(0.5 * new_score + 0.5 * old, 4)
        else:
            session.add(db.StockScore(
                ticker=ticker,
                indirect_impact_score=round(new_score, 4),
            ))
    session.commit()
