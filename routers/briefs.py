"""
routers/briefs.py
-----------------
GET  /daily-brief          — latest daily brief
GET  /daily-brief/history  — list past briefs
POST /daily-brief/generate — generate a new brief on demand
POST /feedback             — user feedback loop

Phase 2+ additions (Upgrade Spec §8.3 interface requirements):
  GET  /outcomes             — list post-alert outcome records
  GET  /outcomes/{id}        — single outcome detail
  POST /outcomes/{id}/review — user review / label submission
  POST /outcomes/update      — trigger forward-return update sweep
  GET  /weights              — inspect current factor weights
  POST /weights/update       — run adaptive weight update (weekly review)
  GET  /regime               — current market regime
  POST /regime               — set active regime label
  GET  /events/{id}/gap      — expectation gap for an event
  GET  /events/{id}/inflection — narrative inflection for an event
  GET  /events/{id}/indirect  — second-order indirect impacts
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import database as db
from database import get_db
import models
from services.alerts import generate_daily_brief

router = APIRouter(tags=["Briefs & Intelligence"])


# ---------------------------------------------------------------------------
# Daily Brief
# ---------------------------------------------------------------------------

@router.get("/daily-brief", response_model=models.DailyBriefOut)
def get_latest_brief(session: Session = Depends(get_db)):
    """Return the most recent daily brief."""
    brief = (
        session.query(db.DailyBrief)
        .order_by(db.DailyBrief.timestamp.desc())
        .first()
    )
    if not brief:
        raise HTTPException(status_code=404, detail="No daily brief yet. POST /daily-brief/generate to create one.")
    return brief


@router.get("/daily-brief/history", response_model=List[models.DailyBriefOut])
def brief_history(limit: int = 10, session: Session = Depends(get_db)):
    return (
        session.query(db.DailyBrief)
        .order_by(db.DailyBrief.timestamp.desc())
        .limit(limit)
        .all()
    )


@router.post("/daily-brief/generate", response_model=models.DailyBriefOut, status_code=201)
def create_brief(session: Session = Depends(get_db)):
    """Generate and store a new daily brief from current data."""
    return generate_daily_brief(session)


# ---------------------------------------------------------------------------
# Feedback (Phase 1)
# ---------------------------------------------------------------------------

@router.post("/feedback", response_model=models.FeedbackOut, status_code=201)
def submit_feedback(payload: models.FeedbackIn, session: Session = Depends(get_db)):
    """Submit user feedback on event quality or stock scores."""
    entry = db.FeedbackEntry(
        event_id=payload.event_id,
        ticker=payload.ticker,
        rating=payload.rating,
        comment=payload.comment,
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


# ---------------------------------------------------------------------------
# Outcome Tracking (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/outcomes", response_model=List[models.AlertOutcomeOut])
def list_outcomes(
    label: Optional[str] = Query(None, description="Filter by outcome_label"),
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_db),
):
    """List post-alert outcome records."""
    q = session.query(db.AlertOutcome)
    if label:
        q = q.filter(db.AlertOutcome.outcome_label == label)
    if ticker:
        q = q.filter(db.AlertOutcome.ticker == ticker.upper())
    return q.order_by(db.AlertOutcome.timestamp.desc()).limit(limit).all()


@router.get("/outcomes/{outcome_id}", response_model=models.AlertOutcomeOut)
def get_outcome(outcome_id: int, session: Session = Depends(get_db)):
    """Get a single outcome record."""
    o = session.get(db.AlertOutcome, outcome_id)
    if not o:
        raise HTTPException(status_code=404, detail="Outcome not found")
    return o


@router.post("/outcomes/{outcome_id}/review", status_code=200)
def review_outcome(
    outcome_id: int,
    payload: models.OutcomeReviewIn,
    session: Session = Depends(get_db),
):
    """
    Submit user review label for a past alert outcome.
    This data is used by the adaptive learning engine.
    """
    o = session.get(db.AlertOutcome, outcome_id)
    if not o:
        raise HTTPException(status_code=404, detail="Outcome not found")
    o.user_override_label = payload.user_override_label
    o.user_comment        = payload.user_comment
    o.reviewed            = 1
    # If auto-label was 'pending' or user disagrees, update to user label
    o.outcome_label       = payload.user_override_label
    session.commit()
    return {"message": f"Outcome {outcome_id} reviewed.", "label": payload.user_override_label}


@router.post("/outcomes/update", status_code=200)
def trigger_outcome_update(
    max_records: int = Query(50, le=200),
    session: Session = Depends(get_db),
):
    """
    Run forward-return update sweep for pending outcome records.
    Requires yfinance to be installed.
    """
    from services.outcome_tracker import run_outcome_update_sweep
    updated = run_outcome_update_sweep(session, max_records=max_records)
    return {"updated": updated, "message": f"{updated} outcome records updated with forward returns."}


# ---------------------------------------------------------------------------
# Factor Weights & Learning (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/weights", response_model=List[models.FactorWeightOut])
def list_weights(
    regime: Optional[str] = Query("base", description="base | risk_on | risk_off | ..."),
    session: Session = Depends(get_db),
):
    """Inspect current adaptive factor weights for a given regime."""
    rows = (
        session.query(db.FactorWeight)
        .filter_by(regime_label=regime)
        .order_by(db.FactorWeight.factor_name)
        .all()
    )
    return rows


@router.post("/weights/update", status_code=200)
def trigger_weight_update(session: Session = Depends(get_db)):
    """
    Run the weekly weight update from labeled outcome data.
    Returns updated weights and confidence calibration.
    """
    from services.learning import run_weekly_review
    result = run_weekly_review(session)
    return result


# ---------------------------------------------------------------------------
# Regime Management (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/regime", response_model=models.RegimeOut)
def get_regime(session: Session = Depends(get_db)):
    """Return the currently active market regime."""
    active = session.query(db.RegimeState).filter_by(is_active=1).first()
    if not active:
        raise HTTPException(status_code=404, detail="No active regime. POST /regime to set one.")
    return active


@router.post("/regime", status_code=201)
def set_regime(
    label: str = Query(..., description="base | risk_on | risk_off | war_escalation | ..."),
    session: Session = Depends(get_db),
):
    """Set the active market regime (closes prior regime)."""
    from services.learning import set_regime as _set_regime
    regime = _set_regime(label, session)
    return {"message": f"Regime set to '{label}'.", "id": regime.id}


# ---------------------------------------------------------------------------
# Expectation Gap (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/events/{event_id}/gap", response_model=models.ExpectationGapOut)
def get_event_gap(event_id: int, session: Session = Depends(get_db)):
    """
    Compute / retrieve expectation gap scores for an event.
    Returns the aggregate and per-ticker breakdown.
    """
    from services.expectation_gap import compute_event_expectation_gaps, interpret_gap

    event = session.get(db.Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    gaps = compute_event_expectation_gaps(event_id, session)
    if not gaps:
        raise HTTPException(status_code=404, detail="No ticker impacts found for this event")

    mean_gap = sum(gaps.values()) / len(gaps)
    return models.ExpectationGapOut(
        event_id=event_id,
        gap_score=round(mean_gap, 4),
        interpretation=interpret_gap(mean_gap),
    )


# ---------------------------------------------------------------------------
# Narrative Inflection (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/events/{event_id}/inflection", response_model=models.NarrativeInflectionOut)
def get_event_inflection(event_id: int, session: Session = Depends(get_db)):
    """Compute narrative inflection score and signal for an event."""
    from services.narrative_inflection import compute_narrative_inflection

    event = session.get(db.Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    score, signal, components = compute_narrative_inflection(event_id, session)
    return models.NarrativeInflectionOut(
        event_id=event_id,
        attention_velocity=components.get("attention_velocity", 1.0),
        price_response=components.get("price_response_norm"),
        contradiction_rate=components.get("contradiction_rate", 0.0),
        inflection_score=score,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Second-Order Indirect Impacts (Phase 2+)
# ---------------------------------------------------------------------------

@router.get("/events/{event_id}/indirect", response_model=models.SecondOrderImpactOut)
def get_event_indirect_impacts(event_id: int, session: Session = Depends(get_db)):
    """
    Compute second-order (indirect) impact map for an event.
    Returns both direct and indirect affected tickers with confidence scores.
    """
    from services.second_order import compute_indirect_impacts

    event = session.get(db.Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Direct impacts
    direct = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .all()
    )
    direct_tickers = [i.ticker for i in direct]

    # Indirect impacts
    indirect_list = compute_indirect_impacts(event_id, session)
    indirect_items = [
        models.IndirectImpactItem(
            ticker=item["ticker"],
            company_name=getattr(session.get(db.Ticker, item["ticker"]), "company_name", None),
            indirect_impact_score=item["indirect_impact_score"],
            relationship_type=item["relationship_type"],
            from_entity=item["from_entity"],
            confidence=item["confidence"],
            time_horizon_days=item["time_horizon_days"],
        )
        for item in indirect_list
    ]

    return models.SecondOrderImpactOut(
        event_id=event_id,
        direct_impacts=direct_tickers,
        indirect_impacts=indirect_items,
    )


# ---------------------------------------------------------------------------
# ML Model Info (also accessible from briefs router for convenience)
# ---------------------------------------------------------------------------

@router.get("/ml/model-info", response_model=models.MLModelInfo)
def get_ml_model_info(session: Session = Depends(get_db)):
    """Return ML model status and training metadata."""
    from services.ml_predictor import get_model_info
    return get_model_info(session)
