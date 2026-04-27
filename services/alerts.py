"""
services/alerts.py
------------------
Alert trigger logic (Upgrade Spec §8.4 + Blueprint §7).

Phase 2+ changes:
  - Alerts require a threshold COMBINATION of credibility + expectation_gap + opportunity.
    A credible event that is already fully priced in (no expectation gap, no lag)
    does NOT trigger an alert.
  - Duplicate suppression: no second alert unless the event materially changes,
    official confirmation appears, or crowding/inflection dynamics shift.
  - Horizon field included in every alert card (intraday | short_swing | structural).
  - Feature and component snapshots stored for later outcome tracking.

Alert tiers
-----------
Critical : high credibility + strong expectation gap + strong opportunity
High     : moderate credibility + positive expectation gap + decent opportunity
Monitor  : above minimum threshold on all three dimensions

Duplicate suppression
---------------------
  Within SUPPRESS_WINDOW_HOURS an alert for the same event is only re-fired if:
    • Credibility changed by >= MATERIAL_CREDIBILITY_DELTA (official confirmation arrived)
    • Narrative inflection flipped (peaking → exhausting or vice versa)
    • Expectation gap changed materially (e.g. price moved significantly)
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session
import database as db
from services.outcome_tracker import (
    build_feature_snapshot,
    build_component_snapshot,
    create_outcome_record,
)

# ---------------------------------------------------------------------------
# Tiered thresholds — (min_credibility, min_opp_score, min_expectation_gap)
# ---------------------------------------------------------------------------

TIER_THRESHOLDS = {
    "Critical": (0.78, 0.60, 0.20),
    "High":     (0.60, 0.45, 0.10),
    "Monitor":  (0.45, 0.30, 0.0),
}

# Suppression window (hours)
SUPPRESS_WINDOW_HOURS = 6

# Threshold for treating a credibility update as 'material'
MATERIAL_CREDIBILITY_DELTA = 0.15
MATERIAL_GAP_DELTA         = 0.20


# ---------------------------------------------------------------------------
# Relevance score
# ---------------------------------------------------------------------------

def _relevance_score(event: db.Event, session: Session) -> float:
    """
    Proxy relevance: number of distinct tickers impacted, normalised.
    High-impact events touching many tickers = highly relevant.
    """
    impact_count = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event.event_id)
        .count()
    )
    return min(impact_count / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Expectation gap for the event (aggregate across tickers)
# ---------------------------------------------------------------------------

def _event_expectation_gap(event: db.Event) -> float:
    """Return the event-level expectation proxy as gap score (0 if unavailable)."""
    return abs(event.expectation_proxy or 0.0)


# ---------------------------------------------------------------------------
# Duplicate suppression
# ---------------------------------------------------------------------------

def _already_alerted(event_id: int, session: Session) -> tuple:
    """
    Return (suppressed: bool, prior_alert: Optional[Alert]).
    Also checks if material change has occurred that warrants a new alert.
    """
    cutoff = datetime.utcnow() - timedelta(hours=SUPPRESS_WINDOW_HOURS)
    prior = (
        session.query(db.Alert)
        .filter(
            db.Alert.event_id == event_id,
            db.Alert.timestamp >= cutoff,
        )
        .order_by(db.Alert.timestamp.desc())
        .first()
    )
    return (prior is not None), prior


def _has_material_change(event: db.Event, prior_alert: Optional[db.Alert]) -> bool:
    """
    Return True if the event has changed materially since the prior alert,
    warranting a new notification even within the suppression window.
    """
    if prior_alert is None:
        return False

    # Official confirmation arrived since last alert
    if prior_alert.component_scores_snapshot:
        try:
            snap = json.loads(prior_alert.component_scores_snapshot)
            prior_cred = snap.get("credibility", 0.0) or 0.0
            if (event.credibility_score - prior_cred) >= MATERIAL_CREDIBILITY_DELTA:
                return True

            prior_gap = abs(snap.get("expectation_gap", 0.0) or 0.0)
            current_gap = abs(event.expectation_proxy or 0.0)
            if abs(current_gap - prior_gap) >= MATERIAL_GAP_DELTA:
                return True
        except Exception:
            pass

    # Narrative inflection flipped sign (peaking ↔ accumulation)
    if prior_alert.expectation_gap_score is not None:
        current_inflection = event.narrative_inflection or 0.0
        prior_infl = float(prior_alert.expectation_gap_score)
        if (current_inflection > 0.3) != (prior_infl > 0.3):
            return True

    return False


# ---------------------------------------------------------------------------
# Tier determination
# ---------------------------------------------------------------------------

def determine_tier(
    credibility: float,
    opportunity_score: float,
    expectation_gap: float,
) -> Optional[str]:
    """
    Return the highest applicable tier based on three-dimensional thresholds,
    or None if below all thresholds.

    Phase 2+ rule: alert only when all three dimensions clear the bar.
    """
    for tier, (cred_thresh, opp_thresh, gap_thresh) in TIER_THRESHOLDS.items():
        if (credibility >= cred_thresh
                and opportunity_score >= opp_thresh
                and expectation_gap >= gap_thresh):
            return tier
    return None


# ---------------------------------------------------------------------------
# Horizon estimation
# ---------------------------------------------------------------------------

def _estimate_horizon(event: db.Event, opportunity_score: float) -> str:
    """
    Classify the intended signal time frame.
    intraday       : very fast-moving event, likely to resolve within hours
    short_swing    : 1–5 trading day opportunity
    structural     : multi-week or multi-quarter opportunity
    """
    stage = event.narrative_stage
    event_type = event.event_type or "sector"

    # Intraday: emerging geopolitical flash event with fast attention velocity
    if (stage == "emerging"
            and event_type == "geopolitical"
            and (event.attention_velocity or 1.0) >= 2.0):
        return "intraday"

    # Structural: macro/policy events with slow build-up
    if event_type in ("macro", "company") and stage in ("developing", "peak"):
        return "structural"

    return "short_swing"


# ---------------------------------------------------------------------------
# Main alert trigger
# ---------------------------------------------------------------------------

def try_trigger_alert(event_id: int, session: Session) -> Optional[db.Alert]:
    """
    Evaluate event and fire an alert if thresholds are met and no material
    duplicate exists.  Returns the created Alert or None.

    Phase 2+ additions:
      - Three-dimensional threshold check
      - Stores feature_vector_snapshot and component_scores_snapshot
      - Attaches horizon label
      - Creates AlertOutcome pending record for each affected ticker
    """
    event = session.get(db.Event, event_id)
    if event is None:
        return None

    relevance   = _relevance_score(event, session)
    gap         = _event_expectation_gap(event)

    # Get top opportunity score for this event's tickers
    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .all()
    )
    if not impacts:
        return None

    # Find the highest opportunity score among impacted tickers
    best_opp = 0.0
    best_ticker = None
    best_score_row = None
    for imp in impacts:
        sr = session.query(db.StockScore).filter_by(ticker=imp.ticker).first()
        if sr and sr.opportunity_score > best_opp:
            best_opp       = sr.opportunity_score
            best_ticker    = imp.ticker
            best_score_row = sr

    tier = determine_tier(event.credibility_score, best_opp, gap)
    if tier is None:
        return None

    # Suppression check
    suppressed, prior_alert = _already_alerted(event_id, session)
    if suppressed and not _has_material_change(event, prior_alert):
        return None

    # Build snapshots for explainability
    feature_snap   = build_feature_snapshot(event, best_score_row, event.credibility_score, relevance, tier)
    component_snap = build_component_snapshot(best_score_row)

    horizon = _estimate_horizon(event, best_opp)
    message = _build_message(event, tier, relevance, best_opp, gap, horizon, session)

    # Determine current regime
    try:
        from services.learning import get_current_regime
        regime = get_current_regime(session)
    except Exception:
        regime = "base"

    # Run ML inference — predict outcome and direction from the feature snapshot
    ml_outcome    = None
    ml_direction  = None
    ml_confidence = 0.0
    try:
        from services.ml_predictor import predict
        ml_result    = predict(feature_snap, event=event)
        ml_outcome   = ml_result.get("predicted_outcome")
        ml_direction = ml_result.get("predicted_direction")
        ml_confidence = ml_result.get("ml_confidence", 0.0)
    except Exception:
        pass  # ML inference failure is non-fatal — alert still fires

    alert = db.Alert(
        event_id=event_id,
        tier=tier,
        message=message,
        horizon=horizon,
        expectation_gap_score=round(gap, 4),
        feature_vector_snapshot=json.dumps(feature_snap),
        component_scores_snapshot=json.dumps(component_snap),
        confidence_score=round(min(event.credibility_score * best_opp * 2, 1.0), 4),
        regime_label=regime,
        ml_predicted_outcome=ml_outcome,
        ml_predicted_direction=ml_direction,
        ml_confidence=round(ml_confidence, 4),
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)

    # Create pending outcome records for affected tickers
    for imp in impacts:
        sr = session.query(db.StockScore).filter_by(ticker=imp.ticker).first()
        if sr and sr.opportunity_score >= 0.25:
            create_outcome_record(alert.id, imp.ticker, session, event_id=event_id)

    return alert


def _build_message(
    event: db.Event,
    tier: str,
    relevance: float,
    opportunity_score: float,
    expectation_gap: float,
    horizon: str,
    session: Session,
) -> str:
    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event.event_id)
        .order_by(db.EventTickerImpact.impact_score.desc())
        .limit(3)
        .all()
    )
    tickers_str = ", ".join(
        f"{i.ticker} ({'+' if i.impact_score >= 0 else ''}{i.impact_score:.2f})"
        for i in impacts
    ) or "No specific tickers identified"

    # Narrative inflection label
    inflection = event.narrative_inflection or 0.0
    if inflection > 0.3:
        infl_tag = "Building"
    elif inflection < -0.3:
        infl_tag = "Exhausting"
    else:
        infl_tag = "Neutral"

    return (
        f"[{tier.upper()}] {event.title} | "
        f"Cred: {event.credibility_score:.2f} | "
        f"Opp: {opportunity_score:.2f} | "
        f"Gap: {expectation_gap:.2f} | "
        f"Stage: {event.narrative_stage} | "
        f"Inflection: {infl_tag} | "
        f"Horizon: {horizon} | "
        f"Tickers: {tickers_str}"
    )


# ---------------------------------------------------------------------------
# Batch sweep
# ---------------------------------------------------------------------------

def run_alert_sweep(session: Session):
    """Check all events and trigger any pending alerts. Called periodically."""
    events = session.query(db.Event).all()
    triggered = []
    for event in events:
        alert = try_trigger_alert(event.event_id, session)
        if alert:
            triggered.append(alert)
    return triggered


# ---------------------------------------------------------------------------
# Daily brief
# ---------------------------------------------------------------------------

def generate_daily_brief(session: Session) -> db.DailyBrief:
    """
    Build a structured daily brief covering top events, narrative movements,
    opportunity shifts, and alert review summary.
    """
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=24)

    top_events = (
        session.query(db.Event)
        .filter(db.Event.timestamp >= cutoff)
        .order_by(db.Event.credibility_score.desc())
        .limit(10)
        .all()
    )

    alert_counts = {
        "Critical": session.query(db.Alert)
            .filter(db.Alert.tier == "Critical", db.Alert.timestamp >= cutoff).count(),
        "High": session.query(db.Alert)
            .filter(db.Alert.tier == "High", db.Alert.timestamp >= cutoff).count(),
        "Monitor": session.query(db.Alert)
            .filter(db.Alert.tier == "Monitor", db.Alert.timestamp >= cutoff).count(),
    }

    lines = [
        f"=== Daily Intelligence Brief — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===",
        f"Alerts: {alert_counts['Critical']} Critical | {alert_counts['High']} High | {alert_counts['Monitor']} Monitor",
        "",
        "TOP EVENTS (last 24h):",
    ]

    for ev in top_events:
        infl = ev.narrative_inflection or 0.0
        infl_tag = "▲ Building" if infl > 0.3 else ("▼ Exhausting" if infl < -0.3 else "→ Neutral")
        lines.append(
            f"  • [{ev.narrative_stage.upper()}] {ev.title} "
            f"(cred: {ev.credibility_score:.2f} | gap: {ev.expectation_proxy or 0:.2f} | {infl_tag})"
        )
        if ev.summary:
            lines.append(f"    {ev.summary[:200]}")

    # Top opportunities (full composite score)
    top_scores = (
        session.query(db.StockScore)
        .order_by(db.StockScore.opportunity_score.desc())
        .limit(5)
        .all()
    )
    if top_scores:
        lines += ["", "TOP OPPORTUNITIES:"]
        for s in top_scores:
            lines.append(
                f"  • {s.ticker} — Opp: {s.opportunity_score:.2f} | "
                f"Gap: {s.expectation_gap_score:.2f} | "
                f"Lag: {s.lag_score:.2f} | "
                f"Crowd: {s.crowding_score:.2f} | "
                f"Bucket: {s.decision_bucket}"
            )

    # Narrative movements
    building = [
        ev for ev in top_events
        if (ev.narrative_inflection or 0) > 0.3
    ]
    exhausting = [
        ev for ev in top_events
        if (ev.narrative_inflection or 0) < -0.3
    ]

    if building:
        lines += ["", "BUILDING NARRATIVES:"]
        for ev in building[:3]:
            lines.append(f"  ▲ {ev.title[:80]}")
    if exhausting:
        lines += ["", "EXHAUSTING NARRATIVES:"]
        for ev in exhausting[:3]:
            lines.append(f"  ▼ {ev.title[:80]}")

    summary = "\n".join(lines)
    brief   = db.DailyBrief(summary_text=summary)
    session.add(brief)
    session.commit()
    session.refresh(brief)
    return brief
