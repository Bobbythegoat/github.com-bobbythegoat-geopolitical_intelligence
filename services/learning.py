"""
services/learning.py
---------------------
Adaptive Learning Engine (Upgrade Spec §6)

Converts the system from static analysis into a continuously improving
decision engine by updating factor weights according to realised outcomes.

Design principles (from spec)
------------------------------
1. Not an unconstrained self-modifying black box.
2. Only weights and calibration update automatically.
3. Structure and factor definitions require human review to change.
4. Weights are regularised so one lucky event cannot swing them wildly.
5. Regime-aware: maintains separate weight sets per market regime.
6. Explainable: every update is logged with its input data.

Weight update rule (from spec)
-------------------------------
    w_{t+1} = (1 - λ)·w_t + η·(y_t - ŷ_t)·x_t

Where:
    w_t   = current factor weight vector
    λ     = regularisation term (prevents runaway growth)
    η     = learning rate
    y_t   = realised standardised outcome (Sharpe proxy or ±1 label)
    ŷ_t   = model's predicted outcome (opportunity score rescaled)
    x_t   = feature vector at alert time

Confidence calibration
----------------------
The engine also tracks hit rates per factor bucket and adjusts confidence
labels accordingly.  Overconfident signals (high score, poor outcome) are
downweighted; underconfident signals (low score, strong outcome) are
upweighted.

Regime conditioning
-------------------
Market regime labels: base | risk_on | risk_off | rate_sensitive |
                      war_escalation | earnings_season | policy_shock

Each regime maintains its own weight set.  When a new regime is detected the
base weights are used until sufficient regime-specific data has been collected.
"""

import json
import math
import time
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

import database as db
from services.outcome_tracker import get_labeled_outcomes

# ---------------------------------------------------------------------------
# VIX cache — avoid hitting Yahoo Finance on every article processed
# ---------------------------------------------------------------------------

_VIX_CACHE: Optional[float] = None
_VIX_CACHE_TS: float        = 0.0
_VIX_CACHE_TTL: int         = 600   # seconds (10 minutes)


def _get_cached_vix() -> Optional[float]:
    """Return cached VIX value, refreshing at most every 10 minutes."""
    global _VIX_CACHE, _VIX_CACHE_TS
    now = time.monotonic()
    if _VIX_CACHE is not None and (now - _VIX_CACHE_TS) < _VIX_CACHE_TTL:
        return _VIX_CACHE
    try:
        from services.price_fetcher import get_vix
        vix = get_vix()
        if vix is not None:
            _VIX_CACHE    = vix
            _VIX_CACHE_TS = now
        return vix
    except Exception:
        return _VIX_CACHE   # return stale value if refresh fails

# ---------------------------------------------------------------------------
# Hyperparameters (conservative defaults)
# ---------------------------------------------------------------------------

LEARNING_RATE      = 0.02    # η — small step to prevent instability
REGULARISATION     = 0.005   # λ — L2 decay per update
MIN_SAMPLES_LEARN  = 15      # don't update until we have this many labeled outcomes
MAX_WEIGHT         = 0.80    # absolute weight ceiling
MIN_WEIGHT         = -0.80   # absolute weight floor
GUARDRAIL_STEP     = 0.05    # maximum single-update weight change

# ---------------------------------------------------------------------------
# Factor names aligned with the scoring formula
# ---------------------------------------------------------------------------

FACTOR_NAMES = [
    "exposure",
    "credibility",
    "expectation_gap",
    "indirect_impact",
    "lag",
    "asymmetry",
    "crowding",      # penalty factor (negative contribution)
    "risk",          # penalty factor (negative contribution)
]

# ---------------------------------------------------------------------------
# Default base weights (from scoring.py, mirrored here for learning init)
# ---------------------------------------------------------------------------

BASE_WEIGHTS: Dict[str, float] = {
    "exposure":         0.15,
    "credibility":      0.20,
    "expectation_gap":  0.18,   # synced with scoring.py OPPORTUNITY_WEIGHTS
    "indirect_impact":  0.10,
    "lag":              0.15,
    "asymmetry":        0.10,
    "crowding":        -0.20,   # synced with scoring.py (raised penalty)
    "risk":            -0.08,   # synced with scoring.py (raised penalty)
}

# ---------------------------------------------------------------------------
# Regime detection heuristics
# ---------------------------------------------------------------------------

REGIME_KEYWORDS: Dict[str, List[str]] = {
    "war_escalation":  ["war", "invasion", "military", "missile", "nuclear", "nato"],
    "rate_sensitive":  ["fed", "rate hike", "rate cut", "inflation", "cpi", "yield"],
    "earnings_season": ["earnings", "quarterly", "guidance", "beat", "miss", "revenue"],
    "policy_shock":    ["sanctions", "export ban", "tariff", "embargo", "regulation"],
    "risk_off":        ["recession", "bank failure", "crisis", "collapse", "panic"],
    "risk_on":         ["rally", "growth", "expansion", "deal", "ceasefire", "recovery"],
}


def detect_regime(event: db.Event) -> str:
    """
    Detect market regime using VIX as primary signal, keyword counts as fallback.
    Regimes: base | risk_on | risk_off | war_escalation | earnings_season | policy_shock | rate_sensitive

    VIX is cached for 10 minutes to avoid a network call on every article.
    """
    # Try VIX-based detection first (cached)
    try:
        vix = _get_cached_vix()
        if vix is not None:
            # VIX-based regime classification
            text = f"{event.title or ''} {event.summary or ''}".lower()

            # War/geopolitical escalation overlay (keyword check on top of VIX)
            war_kws = ["war", "military", "attack", "missile", "invasion", "conflict", "airstrike", "bomb"]
            if sum(1 for kw in war_kws if kw in text) >= 2:
                return "war_escalation"

            # Policy shock overlay
            policy_kws = ["sanctions", "export control", "ban", "tariff", "embargo"]
            if sum(1 for kw in policy_kws if kw in text) >= 2:
                return "policy_shock"

            # VIX thresholds
            if vix >= 30:
                return "risk_off"
            elif vix >= 20:
                # Moderate fear — check event type for refinement
                rate_kws = ["fed", "interest rate", "fomc", "central bank", "yield", "inflation", "cpi"]
                if sum(1 for kw in rate_kws if kw in text) >= 1:
                    return "rate_sensitive"
                return "risk_off"
            elif vix <= 13:
                return "risk_on"
            else:
                # VIX 13-20 = moderate/base
                earnings_kws = ["earnings", "guidance", "revenue", "eps", "quarterly", "beat", "miss"]
                if sum(1 for kw in earnings_kws if kw in text) >= 2:
                    return "earnings_season"
                return "base"
    except Exception:
        pass

    # Keyword-only fallback (original logic)
    text = f"{event.title or ''} {event.summary or ''}".lower()
    scores = {
        "war_escalation":  sum(1 for kw in ["war","military","attack","missile","invasion","conflict"] if kw in text),
        "policy_shock":    sum(1 for kw in ["sanctions","export control","ban","tariff","embargo"] if kw in text),
        "rate_sensitive":  sum(1 for kw in ["fed","interest rate","fomc","yield","inflation","cpi"] if kw in text),
        "earnings_season": sum(1 for kw in ["earnings","guidance","revenue","eps","quarterly"] if kw in text),
        "risk_off":        sum(1 for kw in ["crisis","crash","collapse","recession","default"] if kw in text),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "base"


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def ensure_weights_exist(session: Session) -> None:
    """
    Initialise factor_weights table with BASE_WEIGHTS for each factor
    and each known regime.  Safe to call repeatedly — skips existing rows.
    """
    regimes = ["base"] + list(REGIME_KEYWORDS.keys())
    for regime in regimes:
        for factor, default_val in BASE_WEIGHTS.items():
            existing = (
                session.query(db.FactorWeight)
                .filter_by(factor_name=factor, regime_label=regime)
                .first()
            )
            if existing is None:
                session.add(db.FactorWeight(
                    factor_name=factor,
                    weight_value=default_val,
                    regime_label=regime,
                    sample_count=0,
                ))
    session.commit()


def get_weights(session: Session, regime_label: str = "base") -> Dict[str, float]:
    """
    Retrieve current weights for a given regime.
    Falls back to 'base' if the regime has insufficient data.
    """
    rows = (
        session.query(db.FactorWeight)
        .filter_by(regime_label=regime_label)
        .all()
    )
    if not rows:
        # Fallback to base weights dict
        return dict(BASE_WEIGHTS)

    weights = {r.factor_name: r.weight_value for r in rows}
    return weights


# ---------------------------------------------------------------------------
# Online weight update
# ---------------------------------------------------------------------------

def _extract_feature_vector(features: dict) -> Dict[str, float]:
    """
    Extract the factor feature values from a feature snapshot dict.
    Maps alert-snapshot keys to factor names.
    """
    return {
        "exposure":         float(features.get("opportunity_score", 0.5)),
        "credibility":      float(features.get("credibility_score", 0.5)),
        "expectation_gap":  float(features.get("expectation_gap_score") or 0.0),
        "indirect_impact":  float(features.get("indirect_impact_score") or 0.0),
        "lag":              float(features.get("lag_score") or 0.0),
        "asymmetry":        float(features.get("asymmetry_score") or 0.0),
        "crowding":         float(features.get("crowding_score") or 0.0),
        "risk":             float(features.get("risk_score") or 0.0),
    }


def _standardise_outcome(record: dict) -> float:
    """
    Convert a labeled outcome to a standardised scalar y_t in [-1, 1].
    Uses the realised Sharpe proxy where available, else a label-based mapping.
    """
    sharpe = record.get("sharpe")
    if sharpe is not None:
        # Clip Sharpe proxy to [-3, 3] then normalise to [-1, 1]
        return max(-1.0, min(1.0, sharpe / 3.0))

    label_map = {
        "profitable":   0.8,
        "early":        0.3,    # was right, just early
        "neutral":      0.0,
        "late":        -0.2,
        "unprofitable":-0.8,
        "invalidated": -0.5,
    }
    return label_map.get(record.get("outcome_label", "neutral"), 0.0)


def _predicted_outcome(features: dict) -> float:
    """
    The model's predicted outcome at alert time, derived from the opportunity score.
    Rescaled from [0,1] to [-0.5, 0.5] so errors are centred around zero.
    """
    opp = float(features.get("opportunity_score") or 0.5)
    return (opp - 0.5)


def update_weights(session: Session, regime_label: str = "base") -> Dict[str, float]:
    """
    Run one pass of the online weight update rule over all labeled outcomes
    for the given regime.

    Returns updated weight dict.
    Guardrails:
      - Maximum per-update change capped at GUARDRAIL_STEP
      - Weights clamped to [MIN_WEIGHT, MAX_WEIGHT]
      - Regularisation (L2) applied at each update
    """
    ensure_weights_exist(session)
    outcomes = get_labeled_outcomes(session, regime_label=regime_label)

    if not outcomes:
        return get_weights(session, regime_label)

    current_weights = get_weights(session, regime_label)

    for record in outcomes:
        features = record.get("features", {})
        if not features:
            continue

        x_t  = _extract_feature_vector(features)
        y_t  = _standardise_outcome(record)
        yhat = _predicted_outcome(features)
        error = y_t - yhat

        new_weights = {}
        for factor in FACTOR_NAMES:
            w_t = current_weights.get(factor, BASE_WEIGHTS.get(factor, 0.0))
            x_i = x_t.get(factor, 0.0)

            # Core update rule: w_{t+1} = (1 - λ)·w_t + η·error·x_i
            raw_update = (1 - REGULARISATION) * w_t + LEARNING_RATE * error * x_i

            # Guardrail: cap single-step change
            delta = raw_update - w_t
            delta = max(-GUARDRAIL_STEP, min(GUARDRAIL_STEP, delta))
            w_new = w_t + delta

            # Absolute clamp
            w_new = max(MIN_WEIGHT, min(MAX_WEIGHT, w_new))
            new_weights[factor] = round(w_new, 5)

        current_weights = new_weights

    # Persist updated weights
    for factor, new_val in current_weights.items():
        row = (
            session.query(db.FactorWeight)
            .filter_by(factor_name=factor, regime_label=regime_label)
            .first()
        )
        if row:
            row.weight_value = new_val
            row.sample_count += len(outcomes)
            row.updated_at   = datetime.utcnow()
    session.commit()

    return current_weights


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def calibrate_confidence(session: Session) -> dict:
    """
    Bucket predictions by opportunity-score range and compute hit rate.
    Downweight overconfident signals; upweight underconfident ones.

    Returns a dict of {bucket_label: hit_rate}.
    """
    outcomes = get_labeled_outcomes(session)
    if not outcomes:
        return {}

    buckets = {
        "low":    {"count": 0, "hits": 0},    # opp < 0.4
        "medium": {"count": 0, "hits": 0},    # 0.4–0.65
        "high":   {"count": 0, "hits": 0},    # > 0.65
    }

    for record in outcomes:
        opp = float(record.get("features", {}).get("opportunity_score") or 0.0)
        label = record.get("outcome_label", "neutral")
        hit   = 1 if label in ("profitable", "early") else 0

        if opp < 0.4:
            buckets["low"]["count"]   += 1
            buckets["low"]["hits"]    += hit
        elif opp < 0.65:
            buckets["medium"]["count"] += 1
            buckets["medium"]["hits"]  += hit
        else:
            buckets["high"]["count"]  += 1
            buckets["high"]["hits"]   += hit

    hit_rates = {}
    for label, data in buckets.items():
        if data["count"] > 0:
            hit_rates[label] = round(data["hits"] / data["count"], 3)

    # Persist hit_rate for every factor in the current regime using the "high"
    # bucket rate as the system-level precision proxy (most actionable signals).
    high_hit_rate = hit_rates.get("high")
    if high_hit_rate is not None:
        rows = (
            session.query(db.FactorWeight)
            .filter_by(regime_label="base")
            .all()
        )
        for row in rows:
            row.hit_rate = high_hit_rate
    session.commit()
    return hit_rates


# ---------------------------------------------------------------------------
# Regime state management
# ---------------------------------------------------------------------------

def set_regime(regime_label: str, session: Session) -> db.RegimeState:
    """
    Record a new regime.  Closes the previous active regime first.
    """
    # Close existing active regime
    active = session.query(db.RegimeState).filter_by(is_active=1).all()
    for r in active:
        r.is_active = 0
        r.ended_at  = datetime.utcnow()

    new_regime = db.RegimeState(label=regime_label, is_active=1)
    session.add(new_regime)
    session.commit()
    session.refresh(new_regime)
    return new_regime


def get_current_regime(session: Session) -> str:
    """Return the currently active regime label, defaulting to 'base'."""
    active = session.query(db.RegimeState).filter_by(is_active=1).first()
    return active.label if active else "base"


def run_weekly_review(session: Session) -> dict:
    """
    Run weight updates and confidence calibration.
    Called weekly (or on demand via API endpoint).
    Returns a summary dict.
    """
    regime = get_current_regime(session)
    updated_weights = update_weights(session, regime_label=regime)
    hit_rates       = calibrate_confidence(session)
    return {
        "regime":         regime,
        "updated_weights": updated_weights,
        "hit_rates":       hit_rates,
        "timestamp":       datetime.utcnow().isoformat(),
    }
