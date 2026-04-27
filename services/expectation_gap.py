"""
services/expectation_gap.py
----------------------------
Expectation Gap Engine (Upgrade Spec §4.1)

Measures the distance between what the market expected and what actually
occurred.  Markets move on surprise relative to consensus, not merely on the
existence of information.

Formula (from spec)
-------------------
    ExpectationGap_i = w1·StandardizedFundamentalSurprise_i
                     + w2·NarrativeShift_i
                     + w3·PreEventPriceDriftAdjustment_i
                     + w4·ImpliedMoveResidual_i

The score is positive when the actual outcome materially exceeds consensus
or when an event invalidates the previous narrative.  The score is reduced
when the stock already drifted strongly in the same direction before the event.

Output
------
A signed float in [-1, 1]:
  > 0  : bullish surprise (market under-estimated the positive news)
  < 0  : bearish surprise (market under-estimated the negative impact)
  ≈ 0  : event largely in line with expectations (priced in)

All four components are individually stored for explainability.
"""

import math
from typing import Optional, Tuple
from sqlalchemy.orm import Session

import database as db

# ---------------------------------------------------------------------------
# Default weights (tunable by the adaptive learning engine in services/learning.py)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "fundamental_surprise": 0.40,
    "narrative_shift":       0.25,
    "price_drift_adj":      -0.20,   # negative: pre-event drift REDUCES surprise
    "implied_move_residual": 0.15,
}


# ---------------------------------------------------------------------------
# Narrative stage numerical encoding
# Stage → proxy expectation: consensus has already "priced in" later stages
# ---------------------------------------------------------------------------

STAGE_EXPECTATION = {
    "emerging":   0.1,    # almost nothing priced in
    "developing": 0.35,
    "peak":       0.65,
    "declining":  0.85,   # most of the move is already in prices
}


# ---------------------------------------------------------------------------
# Core component estimators
# ---------------------------------------------------------------------------

def _estimate_fundamental_surprise(
    event: db.Event,
    ticker_impact: float,
    credibility: float,
) -> float:
    """
    Proxy for standardised fundamental surprise.

    In the absence of live analyst estimates the system approximates surprise
    from:
      (a) the magnitude of credibility-weighted impact
      (b) the event_type: official/primary-source events carry more surprise
      (c) source_count: more sources = less surprise (more anticipated)

    Returns a value in [-1, 1].
    """
    # Directional magnitude of the event's credibility-weighted impact
    magnitude = abs(ticker_impact) * credibility

    # Source count penalty: widely covered events have lower surprise
    source_penalty = min((event.source_count or 1) / 20.0, 0.5)

    # Event type premium: geopolitical/official events are harder to anticipate
    type_premium = 0.15 if event.event_type in ("geopolitical", "macro") else 0.0

    raw = math.copysign(magnitude + type_premium - source_penalty, ticker_impact)
    return max(-1.0, min(1.0, raw))


def _estimate_narrative_shift(
    event: db.Event,
    prev_stage: Optional[str] = None,
) -> float:
    """
    Measures how much the event changes the narrative vs the prior state.

    A jump from 'emerging' to 'developing' is a large shift.
    A move within 'peak' (same stage, more articles) is a small shift.
    A contradiction spike is always treated as a large negative shift.

    Returns a value in [-1, 1].
    """
    current_exp = STAGE_EXPECTATION.get(event.narrative_stage, 0.5)
    if prev_stage:
        prior_exp = STAGE_EXPECTATION.get(prev_stage, current_exp)
        stage_delta = current_exp - prior_exp
    else:
        stage_delta = 0.0

    # Contradiction penalty: high contradiction = negative shift
    contradiction_penalty = -(event.contradiction_rate or 0.0) * 0.3

    shift = stage_delta + contradiction_penalty
    return max(-1.0, min(1.0, shift))


def _estimate_pre_event_drift_adjustment(
    price_drift_pct: Optional[float],
    event_direction: float,
) -> float:
    """
    Pre-event price drift adjustment.

    If the stock already moved strongly in the same direction as the event's
    expected impact before the news broke, the expectation gap shrinks.

    price_drift_pct : % price change in the N days before the event (from yfinance)
    event_direction : sign of expected impact (+1 bullish, -1 bearish)

    Returns a positive float in [0, 1] representing how much is already priced in.
    The weight for this component is NEGATIVE in the formula (reduces gap).
    """
    if price_drift_pct is None:
        return 0.0
    # Align drift direction with event direction
    aligned = price_drift_pct * event_direction
    # Normalise: +5% aligned drift → 0.5 already priced in; +10% → 1.0
    priced_in = min(max(aligned / 10.0, 0.0), 1.0)
    return priced_in


def _estimate_implied_move_residual(
    implied_vol_pct: Optional[float],
    actual_move_pct: Optional[float],
) -> float:
    """
    Implied-move residual: difference between what options implied and what happened.

    If the actual move is much larger than what options implied, the market was
    surprised → positive residual.
    If the actual move is smaller, the market over-estimated → negative residual.

    Both values are expected as absolute percentages (e.g. 3.5 for 3.5%).
    Returns a value in [-1, 1].
    """
    if implied_vol_pct is None or actual_move_pct is None:
        return 0.0
    if implied_vol_pct <= 0:
        return 0.0
    residual = (abs(actual_move_pct) - implied_vol_pct) / max(implied_vol_pct, 1.0)
    return max(-1.0, min(1.0, residual))


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_expectation_gap(
    event: db.Event,
    ticker_impact: float,
    credibility: float,
    weights: Optional[dict] = None,
    prev_narrative_stage: Optional[str] = None,
    price_drift_pct: Optional[float] = None,
    implied_vol_pct: Optional[float] = None,
    actual_move_pct: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Compute the Expectation Gap score for an event → ticker pair.

    Parameters
    ----------
    event               : Event ORM object
    ticker_impact       : CWC impact score for this ticker (signed, [-1,1])
    credibility         : event credibility score [0,1]
    weights             : optional override weight dict; defaults to DEFAULT_WEIGHTS
    prev_narrative_stage: narrative stage before this event arrived
    price_drift_pct     : % price change before the event (from market data)
    implied_vol_pct     : options-implied move size (absolute %)
    actual_move_pct     : realised % price change post-event

    Returns
    -------
    (gap_score, components_dict)
      gap_score    : signed float in [-1, 1]
      components   : dict with each component's value for explainability
    """
    w = weights or DEFAULT_WEIGHTS

    fs  = _estimate_fundamental_surprise(event, ticker_impact, credibility)
    ns  = _estimate_narrative_shift(event, prev_narrative_stage)
    pda = _estimate_pre_event_drift_adjustment(price_drift_pct, math.copysign(1.0, ticker_impact))
    imr = _estimate_implied_move_residual(implied_vol_pct, actual_move_pct)

    gap = (
        w["fundamental_surprise"] * fs
        + w["narrative_shift"]    * ns
        + w["price_drift_adj"]    * pda    # negative weight
        + w["implied_move_residual"] * imr
    )
    gap = max(-1.0, min(1.0, gap))

    components = {
        "fundamental_surprise":  round(fs, 4),
        "narrative_shift":       round(ns, 4),
        "pre_event_drift_adj":   round(pda, 4),
        "implied_move_residual": round(imr, 4),
    }
    return round(gap, 4), components


def interpret_gap(gap_score: float) -> str:
    """Return a human-readable interpretation of the gap score."""
    if gap_score >= 0.40:
        return "Strong bullish surprise — market likely underreacted"
    if gap_score >= 0.15:
        return "Moderate bullish surprise — some upside may remain"
    if gap_score >= -0.15:
        return "Event broadly in line with expectations — limited edge"
    if gap_score >= -0.40:
        return "Moderate bearish surprise — downside may continue"
    return "Strong bearish surprise — market likely underreacted to negative news"


def compute_event_expectation_gaps(
    event_id: int,
    session: Session,
    price_data: dict = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Compute expectation gap for every ticker linked to an event.
    Persists the result back to EventTickerImpact and updates
    the event's expectation_proxy field.

    Parameters
    ----------
    event_id : int
        Event ID to compute gaps for
    session : Session
        SQLAlchemy session
    price_data : dict, optional
        Pre-fetched price data: {ticker: {drift, implied_move, ...}}
    weights : dict, optional
        Custom weights for gap formula

    Returns a mapping {ticker: gap_score}.
    """
    event = session.get(db.Event, event_id)
    if event is None:
        return {}

    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .all()
    )
    if not impacts:
        return {}

    credibility = event.credibility_score or 0.5
    results = {}

    for imp in impacts:
        ticker_prices = (price_data or {}).get(imp.ticker, {})
        drift = ticker_prices.get("drift", 0.0)
        implied = ticker_prices.get("implied_move", 0.0)

        gap, _ = compute_expectation_gap(
            event=event,
            ticker_impact=imp.impact_score,
            credibility=credibility,
            weights=weights,
            price_drift_pct=drift,
            implied_vol_pct=implied,
        )
        results[imp.ticker] = gap

    # Update the event's expectation_proxy with the mean absolute gap
    mean_gap = sum(abs(v) for v in results.values()) / max(len(results), 1)
    event.expectation_proxy = round(mean_gap, 4)
    session.commit()

    return results
