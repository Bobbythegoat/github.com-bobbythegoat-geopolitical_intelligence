"""
routers/alerts.py
-----------------
GET  /alerts                          — list recent alerts
POST /alerts/sweep                    — trigger alert sweep
POST /alerts/{id}/dismiss             — dismiss alert
POST /alerts/{id}/outcomes            — record outcome for an alert
GET  /alerts/{id}/outcomes            — list outcomes for an alert
GET  /alerts/outcomes/all             — list all outcome records
POST /alerts/outcomes/{id}/review     — manually label / override outcome
POST /alerts/outcomes/sweep           — update forward returns for pending outcomes
GET  /alerts/outcomes/stats           — outcome statistics summary
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import database as db
from database import get_db
import models
from services.alerts import run_alert_sweep

router = APIRouter(prefix="/alerts", tags=["Alerts"])


# ---------------------------------------------------------------------------
# Alert listing and management
# ---------------------------------------------------------------------------

@router.get("/", response_model=List[models.AlertOut])
def list_alerts(
    tier: Optional[str] = Query(None, description="Critical | High | Monitor"),
    active_only: bool = Query(True, description="Exclude dismissed alerts"),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_db),
):
    """List recent alerts, newest first."""
    q = session.query(db.Alert)
    if tier:
        q = q.filter(db.Alert.tier == tier)
    if active_only:
        q = q.filter(db.Alert.dismissed == 0)
    q = q.order_by(db.Alert.timestamp.desc())
    alerts = q.limit(limit).all()

    result = []
    for a in alerts:
        event = session.get(db.Event, a.event_id)
        result.append(models.AlertOut(
            id=a.id,
            event_id=a.event_id,
            tier=a.tier,
            message=a.message,
            dismissed=a.dismissed,
            timestamp=a.timestamp,
            event_title=event.title if event else None,
            horizon=getattr(a, "horizon", "short_swing"),
            expectation_gap_score=getattr(a, "expectation_gap_score", 0.0),
            confidence_score=getattr(a, "confidence_score", 0.5),
            regime_label=getattr(a, "regime_label", None),
            ml_predicted_outcome=getattr(a, "ml_predicted_outcome", None),
            ml_predicted_direction=getattr(a, "ml_predicted_direction", None),
            ml_confidence=getattr(a, "ml_confidence", 0.0),
        ))
    return result


@router.post("/sweep", status_code=200)
def trigger_alert_sweep(session: Session = Depends(get_db)):
    """Run alert evaluation across all events immediately."""
    triggered = run_alert_sweep(session)
    return {"alerts_triggered": len(triggered)}


@router.post("/{alert_id}/dismiss", status_code=200)
def dismiss_alert(alert_id: int, session: Session = Depends(get_db)):
    """Mark an alert as dismissed."""
    alert = session.get(db.Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.dismissed = 1
    session.commit()
    return {"message": f"Alert {alert_id} dismissed."}


# ---------------------------------------------------------------------------
# Outcome recording — this is what feeds the ML model
# ---------------------------------------------------------------------------

@router.post("/{alert_id}/outcomes", response_model=models.AlertOutcomeOut, status_code=201)
def record_outcome(
    alert_id: int,
    body: models.AlertOutcomeIn,
    session: Session = Depends(get_db),
):
    """
    Record a forward-return outcome for an alert.
    Creates a pending record if none exists yet; updates if one already does.

    The ML model trains on these records once outcome_label is not 'pending'.
    You can supply forward returns directly, or leave them null and call
    POST /alerts/outcomes/sweep later to auto-populate from yfinance.
    """
    alert = session.get(db.Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    existing = (
        session.query(db.AlertOutcome)
        .filter_by(alert_id=alert_id, ticker=body.ticker)
        .first()
    )

    if existing:
        # Update any provided fields
        for field in [
            "forward_return_15m", "forward_return_1h", "forward_return_1d",
            "forward_return_3d", "forward_return_1w", "forward_return_1m",
            "max_favorable_excursion", "max_adverse_excursion",
            "realized_volatility", "realized_sharpe_proxy", "outcome_label",
        ]:
            val = getattr(body, field, None)
            if val is not None:
                setattr(existing, field, val)
        session.commit()
        session.refresh(existing)
        return existing

    outcome = db.AlertOutcome(
        alert_id=alert_id,
        ticker=body.ticker,
        event_id=alert.event_id,
        forward_return_15m=body.forward_return_15m,
        forward_return_1h=body.forward_return_1h,
        forward_return_1d=body.forward_return_1d,
        forward_return_3d=body.forward_return_3d,
        forward_return_1w=body.forward_return_1w,
        forward_return_1m=body.forward_return_1m,
        max_favorable_excursion=body.max_favorable_excursion,
        max_adverse_excursion=body.max_adverse_excursion,
        realized_volatility=body.realized_volatility,
        realized_sharpe_proxy=body.realized_sharpe_proxy,
        outcome_label=body.outcome_label or "pending",
    )
    session.add(outcome)
    session.commit()
    session.refresh(outcome)
    return outcome


@router.get("/{alert_id}/outcomes", response_model=List[models.AlertOutcomeOut])
def get_alert_outcomes(alert_id: int, session: Session = Depends(get_db)):
    """Return all outcome records for a specific alert."""
    alert = session.get(db.Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    outcomes = (
        session.query(db.AlertOutcome)
        .filter_by(alert_id=alert_id)
        .all()
    )
    return outcomes


# ---------------------------------------------------------------------------
# Outcome listing and bulk operations
# ---------------------------------------------------------------------------

@router.get("/outcomes/all", response_model=List[models.AlertOutcomeOut])
def list_all_outcomes(
    label: Optional[str] = Query(None, description="Filter by outcome_label"),
    reviewed_only: bool = Query(False),
    limit: int = Query(100, le=500),
    session: Session = Depends(get_db),
):
    """
    List all outcome records.
    Use label='pending' to see what still needs updating.
    Use label='profitable' etc. to audit the training set.
    """
    q = session.query(db.AlertOutcome)
    if label:
        q = q.filter(db.AlertOutcome.outcome_label == label)
    if reviewed_only:
        q = q.filter(db.AlertOutcome.reviewed == 1)
    return q.order_by(db.AlertOutcome.timestamp.desc()).limit(limit).all()


@router.post("/outcomes/{outcome_id}/review", status_code=200)
def review_outcome(
    outcome_id: int,
    body: models.OutcomeReviewIn,
    session: Session = Depends(get_db),
):
    """
    Manually override an outcome label.
    Use this when the auto-label is wrong (e.g. alert was directionally correct
    but the stock was flat due to market-wide selloff).

    Valid labels: profitable | unprofitable | early | late | neutral | invalidated
    """
    outcome = session.get(db.AlertOutcome, outcome_id)
    if not outcome:
        raise HTTPException(status_code=404, detail="Outcome record not found")

    valid_labels = {"profitable", "unprofitable", "early", "late", "neutral", "invalidated"}
    if body.user_override_label not in valid_labels:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid label. Must be one of: {', '.join(sorted(valid_labels))}",
        )

    outcome.user_override_label = body.user_override_label
    outcome.outcome_label       = body.user_override_label   # override auto-label
    outcome.user_comment        = body.user_comment
    outcome.reviewed            = 1
    session.commit()
    return {
        "message": f"Outcome {outcome_id} labeled '{body.user_override_label}'",
        "outcome_id": outcome_id,
        "label": body.user_override_label,
    }


@router.post("/outcomes/sweep", status_code=200)
def run_outcome_sweep(
    max_records: int = Query(50, le=200, description="Max pending records to update"),
    session: Session = Depends(get_db),
):
    """
    Pull live yfinance prices for pending outcome records and compute forward returns.
    Run this once a day to keep outcome labels current.
    The ML model improves with every batch of newly labeled outcomes.

    Returns how many records were moved from 'pending' to a real label.
    """
    from services.outcome_tracker import run_outcome_update_sweep
    updated = run_outcome_update_sweep(session, max_records=max_records)
    total_pending = (
        session.query(db.AlertOutcome)
        .filter_by(outcome_label="pending")
        .count()
    )
    return {
        "updated": updated,
        "still_pending": total_pending,
        "message": (
            f"Labeled {updated} outcome(s). "
            f"{total_pending} still pending (need more time to pass)."
        ),
    }


@router.get("/outcomes/stats", status_code=200)
def outcome_stats(session: Session = Depends(get_db)):
    """
    Summary of the outcome dataset.
    Shows training readiness for the ML model.
    """
    from collections import Counter
    all_outcomes = session.query(db.AlertOutcome).all()
    label_counts = Counter(o.outcome_label for o in all_outcomes)

    labeled   = sum(v for k, v in label_counts.items() if k != "pending")
    total     = len(all_outcomes)
    reviewed  = sum(1 for o in all_outcomes if o.reviewed)

    from services.ml_predictor import MIN_TRAINING_SAMPLES
    ready     = labeled >= MIN_TRAINING_SAMPLES

    return {
        "total_outcome_records": total,
        "label_distribution":    dict(label_counts),
        "labeled_count":         labeled,
        "pending_count":         label_counts.get("pending", 0),
        "manually_reviewed":     reviewed,
        "ml_training_ready":     ready,
        "samples_needed":        max(0, MIN_TRAINING_SAMPLES - labeled),
        "tip": (
            "Run POST /alerts/outcomes/sweep daily to auto-label pending records. "
            "Run POST /stocks/ml/train once ml_training_ready is true."
            if not ready else
            "Ready! Run POST /stocks/ml/train to train the real ML model."
        ),
    }
