"""
services/narrative_inflection.py
----------------------------------
Narrative Inflection Engine (Upgrade Spec §4.3)

Detects whether a story is strengthening, rolling over, or breaking apart.
Sits on top of the existing narrative-stage logic and converts stage labels
into turning-point detection.

Core signals
------------
1. AttentionVelocity  = Mentions_t / max(Mentions_{t-1}, 1)
   Rising velocity = story is gaining momentum.
   Plateauing or falling velocity = story losing steam.

2. PriceResponse      = |Return_t| / max(RealizedVol_t, ε)
   Normalised price move.  If attention is surging but price response is
   weakening, the story is likely peaking or exhausting.

3. ContradictionRate  = fraction of articles contradicting the cluster
   Spike in contradictions = narrative stability falling.

Composite formula (from spec)
------------------------------
    NarrativeInflection_t = α·AttentionVelocity_t
                           - β·PriceResponse_t
                           + γ·ContradictionRate_t

Signal interpretation
---------------------
  inflection > +0.4 AND price_response strong  → Building / accumulation
  inflection > +0.4 AND price_response weak    → Attention without price = Peak risk
  inflection ≈ 0                               → Neutral / consolidating
  inflection < -0.3                            → Exhausting / declining
  contradiction spike (rate > 0.4)             → Narrative stability collapsing

Signals returned
----------------
  'accumulation'  : price rising quietly while attention still low
  'building'      : attention and price both accelerating
  'neutral'       : no clear trend in either direction
  'peaking'       : attention surging but marginal price response weakening
  'exhausting'    : attention decelerating, price response declining
"""

import math
from typing import Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

import database as db

# ---------------------------------------------------------------------------
# Formula coefficients (tunable)
# ---------------------------------------------------------------------------

ALPHA = 0.45   # weight of attention velocity
BETA  = 0.35   # weight of price response (subtracted — high response = not exhausting)
GAMMA = 0.20   # weight of contradiction rate (high contradiction = instability)

# Thresholds for signal classification
ACCUMULATION_ATTENTION_MAX  = 0.8   # low attention ceiling for accumulation signal
PEAKING_ATTENTION_MIN       = 1.4   # velocity above this = crowd arrival
PEAKING_PRICE_RESPONSE_MAX  = 0.5   # weakening price response ceiling
EXHAUSTING_VELOCITY_MAX     = 0.85  # velocity below this suggests deceleration
CONTRADICTION_SPIKE_MIN     = 0.35  # contradiction fraction above this = instability


# ---------------------------------------------------------------------------
# Component estimators
# ---------------------------------------------------------------------------

def _compute_attention_velocity(
    event: db.Event,
    session: Session,
    window_hours: int = 6,
) -> float:
    """
    Estimate AttentionVelocity = recent_articles / prior_articles.

    Uses article counts within the current window vs the prior window.
    Returns a float >= 0.0 (1.0 = stable, >1.0 = accelerating, <1.0 = decelerating).
    """
    now = datetime.utcnow()
    current_cutoff = now - timedelta(hours=window_hours)
    prior_cutoff   = now - timedelta(hours=window_hours * 2)

    all_articles = (
        session.query(db.Article)
        .filter(db.Article.event_id == event.event_id)
        .all()
    )

    current_count = sum(
        1 for a in all_articles if a.timestamp >= current_cutoff
    )
    prior_count = sum(
        1 for a in all_articles
        if prior_cutoff <= a.timestamp < current_cutoff
    )

    velocity = current_count / max(prior_count, 1)
    return round(velocity, 3)


def _compute_contradiction_rate(event: db.Event, session: Session) -> float:
    """
    Fraction of articles in this event cluster that carry a contradiction flag.
    Returns float in [0, 1].
    """
    articles = (
        session.query(db.Article)
        .filter(db.Article.event_id == event.event_id)
        .all()
    )
    if not articles:
        return 0.0
    flagged = sum(1 for a in articles if a.contradiction_flag)
    rate = flagged / len(articles)
    return round(rate, 3)


def _normalise_price_response(price_response: Optional[float]) -> float:
    """
    Normalise a raw price-response metric to [0, 1].
    Raw metric: |Return_t| / max(RealizedVol_t, ε)
    Values > 3 are capped (extreme moves).
    """
    if price_response is None:
        return 0.5   # neutral default when no price data
    return round(min(abs(price_response) / 3.0, 1.0), 3)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_narrative_inflection(
    event_id: int,
    session: Session,
    price_response: Optional[float] = None,
) -> Tuple[float, str, dict]:
    """
    Compute the narrative inflection score for an event.

    Parameters
    ----------
    event_id       : event to analyse
    session        : database session
    price_response : |Return_t| / RealizedVol_t from live market data (optional)

    Returns
    -------
    (inflection_score, signal_label, components)
      inflection_score : float in [-1, 1]
      signal_label     : 'accumulation' | 'building' | 'neutral' | 'peaking' | 'exhausting'
      components       : dict of raw component values
    """
    event = session.get(db.Event, event_id)
    if event is None:
        return 0.0, "neutral", {}

    attention_vel  = _compute_attention_velocity(event, session)
    contradiction  = _compute_contradiction_rate(event, session)
    price_norm     = _normalise_price_response(price_response)

    inflection = (
        ALPHA * min(attention_vel, 3.0) / 3.0   # normalise velocity to [0,1]
        - BETA  * price_norm
        + GAMMA * contradiction
    )
    inflection = round(max(-1.0, min(1.0, inflection)), 4)

    # Classify signal
    signal = _classify_signal(attention_vel, price_norm, contradiction, inflection)

    # Persist back to event record
    event.narrative_inflection = inflection
    event.attention_velocity   = round(attention_vel, 4)
    event.contradiction_rate   = round(contradiction, 4)
    if price_response is not None:
        event.price_response = round(price_response, 4)
    session.commit()

    components = {
        "attention_velocity": attention_vel,
        "price_response_norm": price_norm,
        "contradiction_rate": contradiction,
    }
    return inflection, signal, components


def _classify_signal(
    attention_velocity: float,
    price_norm: float,
    contradiction_rate: float,
    inflection: float,
) -> str:
    """Map metrics to a readable signal label."""
    # Contradiction spike = narrative instability regardless of other signals
    if contradiction_rate >= CONTRADICTION_SPIKE_MIN:
        return "exhausting"

    # Accumulation: price rising quietly while still low attention
    if attention_velocity <= ACCUMULATION_ATTENTION_MAX and price_norm >= 0.3:
        return "accumulation"

    # Building: both attention and price response accelerating
    if attention_velocity >= 1.2 and price_norm >= 0.4:
        return "building"

    # Peaking: high attention but weakening price response
    if (attention_velocity >= PEAKING_ATTENTION_MIN
            and price_norm <= PEAKING_PRICE_RESPONSE_MAX):
        return "peaking"

    # Exhausting: attention decelerating, price response low
    if attention_velocity <= EXHAUSTING_VELOCITY_MAX and price_norm <= 0.3:
        return "exhausting"

    return "neutral"


def run_inflection_sweep(session: Session) -> dict:
    """
    Compute narrative inflection for all active events.
    Call this on a schedule (e.g. every 30 minutes).
    Returns {event_id: signal_label}.
    """
    active_events = (
        session.query(db.Event)
        .filter(db.Event.narrative_stage.in_(["emerging", "developing", "peak"]))
        .all()
    )
    results = {}
    for event in active_events:
        _, signal, _ = compute_narrative_inflection(event.event_id, session)
        results[event.event_id] = signal
    return results
