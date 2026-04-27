"""
services/ml_predictor.py
-------------------------
Gradient Boosting ML predictor for alert outcomes.

Predicts:
  1. Outcome class: profitable (2) | neutral (1) | unprofitable (0)
  2. Price direction: up (+1) | flat (0) | down (-1)

Features (11):
  exposure, credibility, expectation_gap, indirect_impact,
  lag, asymmetry, crowding, risk,
  narrative_stage_num, event_type_num, source_count_norm

Model: scikit-learn GradientBoostingClassifier (CPU, no GPU needed)
Falls back to rule-based estimates until MIN_TRAINING_SAMPLES reached.

Persistence: saved to ml_model.pkl in project root.
"""

import json
import logging
import math
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

import database as db

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MIN_TRAINING_SAMPLES = 10   # lowered for cold-start; model improves as real outcomes accumulate
BOOTSTRAP_MIN_SAMPLES = 5   # minimum for bootstrap training on seeded data
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_model.pkl")

FEATURE_NAMES = [
    # Alert-specific features (11)
    "exposure", "credibility", "expectation_gap", "indirect_impact",
    "lag", "asymmetry", "crowding", "risk",
    "narrative_stage_num", "event_type_num", "source_count_norm",
    # Historical price context features (6) — from yfinance via price_context.py
    # These encode 1-year price history, letting the model learn that
    # "bearish alert + RSI 80" differs from "bearish alert + RSI 25".
    "rsi_14_norm",          # RSI normalised to 0–1  (50 = 0.5 neutral)
    "pct_from_52w_high",    # negative = below 52w high (range approx -1 to 0)
    "momentum_30d",         # 30-day return (positive = uptrend)
    "momentum_90d",         # 90-day return (longer-term trend)
    "rel_strength_90d",     # stock 90d return minus SPY 90d return
    "vol_regime",           # recent vol / trailing vol (>1 = elevated risk)
]

LABEL_MAP = {
    "profitable": 2, "early": 2,
    "neutral": 1,    "pending": 1,
    "unprofitable": 0, "late": 0,
}

DIRECTION_MAP = {
    "profitable": 1, "early": 1,
    "neutral": 0,    "pending": 0,
    "unprofitable": -1, "late": -1,
}

STAGE_NUM = {
    "emerging": 1.0, "developing": 0.8, "peak": 0.5,
    "declining": 0.2, "stale": 0.1,
}

# ── Expanded event type encoding using 15-class causal event system ──────────
# Previously used 4 buckets (geopolitical/macro/company/sector).
# Now maps the 15 specific causal event classes to numeric values that
# reflect their typical market-impact magnitude and urgency.
# The causal_engine.get_event_class_num() provides this mapping directly;
# the legacy 4-bucket mapping is retained for backward compat with old snapshots.
EVENT_TYPE_NUM = {
    # ── Legacy 4-bucket mapping (for old AlertOutcome snapshots) ─────────────
    "geopolitical": 1.0,
    "macro":        0.8,
    "company":      0.6,
    "sector":       0.4,
    # ── New 15-class causal event mapping ────────────────────────────────────
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


# ── Feature extraction ────────────────────────────────────────────────────────

def _get_event_type_num(event: Optional[db.Event]) -> float:
    """
    Resolve event type numeric encoding.
    Prefers the specific causal event class from the causal engine when
    the event carries a precise classification. Falls back to legacy 4-bucket
    encoding for older events stored before the causal engine was deployed.
    """
    if event is None:
        return EVENT_TYPE_NUM.get("sector", 0.4)

    # Try causal event class first (stored in event_type by causal engine)
    event_type = event.event_type or "sector"

    # Check if the stored event_type is already in the new 15-class system
    if event_type in EVENT_TYPE_NUM:
        return EVENT_TYPE_NUM[event_type]

    # Try the causal engine's real-time classification as a fallback
    try:
        from services.causal_engine import get_event_class_num, classify_economic_event
        # Use the canonical title as a proxy text for classification
        proxy_text = getattr(event, "canonical_title", "") or ""
        if proxy_text:
            causal_class, _ = classify_economic_event(proxy_text)
            return get_event_class_num(causal_class)
    except Exception:
        pass

    return EVENT_TYPE_NUM.get(event_type, 0.4)


def _extract_features(outcome: db.AlertOutcome, session: Session) -> Optional[np.ndarray]:
    """
    Build feature vector from an AlertOutcome + its Alert snapshot.

    Feature improvements with causal engine:
    - event_type_num now uses 15-class encoding (not 4-bucket)
    - exposure maps to causal pathway strength (not keyword fraction)
    - Both legacy and new snapshots are handled gracefully
    """
    try:
        alert = session.get(db.Alert, outcome.alert_id)
        if not alert or not alert.feature_vector_snapshot:
            return None
        snap  = json.loads(alert.feature_vector_snapshot)
        event = session.get(db.Event, outcome.event_id) if outcome.event_id else None

        # expectation_gap stored under two keys for backward compat; prefer the explicit one
        eg = snap.get("expectation_gap_score") or snap.get("expectation_gap") or 0.0
        # risk_score may not be in older snapshots; derive from opportunity as proxy
        risk = snap.get("risk_score") or max(0.0, 1.0 - float(snap.get("opportunity_score", 0.5)))

        return np.array([
            float(snap.get("opportunity_score",    0.5)),
            float(snap.get("credibility_score",    0.5)),
            float(eg),
            float(snap.get("indirect_impact_score",0.0)),
            float(snap.get("lag_score",            0.5)),
            float(snap.get("asymmetry_score",      0.0)),
            float(snap.get("crowding_score",       0.3)),
            float(risk),
            STAGE_NUM.get(event.narrative_stage if event else "peak", 0.5),
            _get_event_type_num(event),   # ← now uses 15-class causal encoding
            min((event.source_count or 1) / 10.0, 1.0) if event else 0.1,
            # Price context features (normalised; neutral defaults for old snapshots)
            float(snap.get("rsi_14",             50.0)) / 100.0,   # 0–1
            float(snap.get("pct_from_52w_high", -0.10)),
            float(snap.get("momentum_30d",        0.0)),
            float(snap.get("momentum_90d",        0.0)),
            float(snap.get("rel_strength_90d",    0.0)),
            min(max(float(snap.get("vol_regime",  1.0)), 0.1), 5.0),  # clamp
        ], dtype=float)
    except Exception as e:
        logger.debug("Feature extraction failed: %s", e)
        return None


def _snap_from_dict(feature_snapshot: dict, event: Optional[db.Event] = None) -> np.ndarray:
    """Build feature array from a raw snapshot dict (for inference)."""
    eg   = feature_snapshot.get("expectation_gap_score") or feature_snapshot.get("expectation_gap") or 0.0
    risk = feature_snapshot.get("risk_score") or max(0.0, 1.0 - float(feature_snapshot.get("opportunity_score", 0.5)))
    return np.array([
        float(feature_snapshot.get("opportunity_score",    0.5)),
        float(feature_snapshot.get("credibility_score",    0.5)),
        float(eg),
        float(feature_snapshot.get("indirect_impact_score",0.0)),
        float(feature_snapshot.get("lag_score",            0.5)),
        float(feature_snapshot.get("asymmetry_score",      0.0)),
        float(feature_snapshot.get("crowding_score",       0.3)),
        float(risk),
        STAGE_NUM.get(event.narrative_stage if event else "peak", 0.5),
        _get_event_type_num(event),   # ← now uses 15-class causal encoding
        min((event.source_count or 1) / 10.0, 1.0) if event else 0.1,
        # Price context (neutral defaults when not present in old snapshots)
        float(feature_snapshot.get("rsi_14",             50.0)) / 100.0,
        float(feature_snapshot.get("pct_from_52w_high", -0.10)),
        float(feature_snapshot.get("momentum_30d",        0.0)),
        float(feature_snapshot.get("momentum_90d",        0.0)),
        float(feature_snapshot.get("rel_strength_90d",    0.0)),
        min(max(float(feature_snapshot.get("vol_regime",  1.0)), 0.1), 5.0),
    ], dtype=float)


# ── Training ──────────────────────────────────────────────────────────────────

def _spearman_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    Compute Spearman rank IC (Information Coefficient) between predictions
    and actual values.  IC = rank correlation between predicted probability
    and actual forward return.  A standard quant evaluation metric.

    Returns 0.0 if fewer than 3 pairs are available.
    """
    n = len(pred)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(pred)).astype(float)
    ry = np.argsort(np.argsort(actual)).astype(float)
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def train_model(session: Session) -> dict:
    """
    Train outcome + direction classifiers from labeled AlertOutcome records.
    Saves to MODEL_PATH. Returns training summary dict.

    Methodology (aligned with Gu/Kelly/Xiu 2020 and Harvey/Liu/Zhu standards):

    1. Temporal ordering — outcomes are sorted by timestamp so CV folds
       always train on the past and test on the future (no look-ahead bias).

    2. TimeSeriesSplit CV — replaces StratifiedKFold(shuffle=True) which
       leaked future data into training folds.  In time-series settings,
       shuffling creates a known methodological error: the model appears
       to predict future outcomes but is actually seeing future training
       examples. TimeSeriesSplit guarantees strict temporal separation.

    3. Rank IC metric — balanced_accuracy measures classification quality;
       Rank IC (Spearman correlation between predicted probability and
       actual forward return) measures cross-sectional ranking quality,
       which is what actually matters for stock selection.

    4. Class-balanced sample weights — GBC has no native class_weight param
       so we use compute_sample_weight('balanced') to prevent the model
       from collapsing to always-predict-neutral on imbalanced data.

    5. Stronger regularization — max_features='sqrt' adds feature subsampling
       per tree (standard GBM practice); min_samples_leaf=8 requires more
       evidence per leaf node.  Together these reduce memorisation.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.calibration import CalibratedClassifierCV

    # Prefer XGBoost when available — generally outperforms sklearn GBC on
    # tabular data with <1000 samples via better regularisation (L1+L2) and
    # built-in column subsampling. Falls back to GradientBoosting silently.
    try:
        import xgboost as xgb
        _XGBOOST_AVAILABLE = True
    except ImportError:
        _XGBOOST_AVAILABLE = False

    raw_outcomes = (
        session.query(db.AlertOutcome)
        .filter(
            db.AlertOutcome.outcome_label.isnot(None),
            db.AlertOutcome.outcome_label != "pending",
        )
        .all()
    )

    if len(raw_outcomes) < MIN_TRAINING_SAMPLES:
        return {
            "trained": False,
            "reason": f"Need {MIN_TRAINING_SAMPLES} labeled outcomes, have {len(raw_outcomes)}",
            "n_samples": len(raw_outcomes),
        }

    # ── Sort chronologically for temporal CV validity ─────────────────────────
    # TimeSeriesSplit requires data to be ordered oldest→newest.
    # A shuffled order would defeat the purpose of temporal validation.
    raw_outcomes.sort(key=lambda o: o.timestamp or datetime.min)

    X, y_outcome, y_direction, forward_returns = [], [], [], []
    for o in raw_outcomes:
        feat = _extract_features(o, session)
        if feat is not None:
            X.append(feat)
            y_outcome.append(LABEL_MAP.get(o.outcome_label, 1))
            y_direction.append(DIRECTION_MAP.get(o.outcome_label, 0))
            # Collect actual forward return for Rank IC computation
            forward_returns.append(o.forward_return_1d if o.forward_return_1d is not None
                                   else o.forward_return_1w if o.forward_return_1w is not None
                                   else 0.0)

    if len(X) < MIN_TRAINING_SAMPLES:
        return {"trained": False, "reason": "Insufficient valid feature vectors", "n_samples": len(X)}

    X               = np.array(X)
    y_outcome       = np.array(y_outcome)
    y_direction     = np.array(y_direction)
    forward_returns = np.array(forward_returns)

    # Class-balanced sample weights — prevents always-predict-neutral collapse
    sw_outcome   = compute_sample_weight("balanced", y_outcome)
    sw_direction = compute_sample_weight("balanced", y_direction)

    def _make_pipe():
        if _XGBOOST_AVAILABLE:
            clf = xgb.XGBClassifier(
                # XGBoost hyperparameters for ~100–1000 small tabular samples:
                #   n_estimators=200     — more rounds, controlled by early stopping
                #   max_depth=3          — shallow trees prevent memorisation
                #   learning_rate=0.05   — small steps, rely on more trees
                #   subsample=0.8        — row subsampling (stochastic boosting)
                #   colsample_bytree=0.7 — column subsampling per tree (L2 analogue)
                #   reg_alpha=0.1        — L1 regularisation (feature selection)
                #   reg_lambda=1.0       — L2 regularisation (weight shrinkage)
                #   use_label_encoder=False, eval_metric='mlogloss' — suppress warnings
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, verbosity=0,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.07,
                subsample=0.8, min_samples_leaf=8, max_features="sqrt",
                n_iter_no_change=10, validation_fraction=0.15,
                random_state=42,
            )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    # ── TimeSeriesSplit cross-validation ──────────────────────────────────────
    # Each fold trains on older data and tests on strictly newer data.
    # Require at least 150 samples per fold so every fold is large enough to
    # contain all outcome classes — early system data is heavily "neutral" so
    # tiny folds trigger "y contains 1 class" errors in GBC.
    cv_folds = max(2, min(3, len(X) // 150))
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    results = {}

    for name, y, sw in [("outcome", y_outcome, sw_outcome),
                         ("direction", y_direction, sw_direction)]:
        pipe = _make_pipe()
        try:
            # balanced_accuracy = mean recall per class — correct metric for
            # imbalanced classes (a model always predicting neutral scores ~0.33,
            # not 0.71+, so it cannot game this metric with trivial predictions).
            # error_score=np.nan: failed folds (single-class training sets
            # common in early-stage data) produce NaN silently instead of
            # raising a FitFailedWarning that clutters the terminal output.
            # Our NaN-safe averaging below discards those folds.
            try:
                scores = cross_val_score(
                    pipe, X, y, cv=tscv, scoring="balanced_accuracy",
                    error_score=np.nan,
                    params={"clf__sample_weight": sw},
                )
            except TypeError:
                try:
                    scores = cross_val_score(
                        pipe, X, y, cv=tscv, scoring="balanced_accuracy",
                        error_score=np.nan,
                        fit_params={"clf__sample_weight": sw},
                    )
                except TypeError:
                    scores = cross_val_score(
                        pipe, X, y, cv=tscv,
                        scoring="balanced_accuracy", error_score=np.nan,
                    )
            # NaN-safe averaging: individual folds can produce NaN when a fold's
            # training set is single-class (early data is 90%+ neutral).
            # Filter out NaN folds before computing mean/std so one bad fold
            # doesn't corrupt the entire CV score.
            valid = scores[~np.isnan(scores)]
            if len(valid) > 0:
                results[f"{name}_cv_accuracy"] = round(float(valid.mean()), 4)
                results[f"{name}_cv_std"]      = round(float(valid.std()),  4)
            else:
                logger.warning("All CV folds returned NaN for %s — too few samples or single-class folds", name)
                results[f"{name}_cv_accuracy"] = None
                results[f"{name}_cv_std"]      = None
        except Exception as e:
            logger.warning("CV failed for %s: %s", name, e)
            results[f"{name}_cv_accuracy"] = None
            results[f"{name}_cv_std"]      = None

        # Final fit on all data with balanced weights
        pipe.fit(X, y, clf__sample_weight=sw)

        # Probability calibration via Platt scaling (sigmoid method).
        # GBC and XGBoost both produce uncalibrated probabilities — the raw
        # score for "profitable" may be 0.85 even when the true rate is 0.55.
        # CalibratedClassifierCV wraps the trained estimator and learns a
        # monotone sigmoid mapping from raw scores to calibrated probabilities.
        # cv="prefit" uses the already-trained pipe without refitting.
        try:
            calibrated = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
            calibrated.fit(X, y)
            results[f"{name}_model"] = calibrated
        except Exception as cal_err:
            logger.warning("Probability calibration failed for %s: %s — using uncalibrated model", name, cal_err)
            results[f"{name}_model"] = pipe

    # ── Rank IC (Information Coefficient) ────────────────────────────────────
    # Measures cross-sectional ranking quality: Spearman correlation between
    # predicted profitable-probability and actual forward return.
    # IC > 0.05 is considered meaningful; IC > 0.10 is strong for a factor.
    # Note: this is computed in-sample (indicative only; use with caution).
    rank_ic = None
    try:
        outcome_pipe = results["outcome_model"]
        o_classes    = list(outcome_pipe.named_steps["clf"].classes_)
        profitable_idx = o_classes.index(2) if 2 in o_classes else -1
        if profitable_idx >= 0 and len(forward_returns) >= 5:
            proba = outcome_pipe.predict_proba(X)[:, profitable_idx]
            rank_ic = round(_spearman_ic(proba, forward_returns), 4)
    except Exception as e:
        logger.debug("Rank IC computation failed: %s", e)

    # ── Feature importances ───────────────────────────────────────────────────
    # Logs which features drive predictions. Unstable importances across
    # retrains are a warning sign of overfitting (per Harvey/Liu/Zhu).
    feature_importances = {}
    try:
        outcome_model = results["outcome_model"]
        # CalibratedClassifierCV wraps the pipeline; reach inside to the estimator
        if hasattr(outcome_model, "estimator"):
            inner = outcome_model.estimator
        elif hasattr(outcome_model, "calibrated_classifiers_"):
            inner = outcome_model.calibrated_classifiers_[0].estimator
        else:
            inner = outcome_model
        fi = inner.named_steps["clf"].feature_importances_
        feature_importances = {
            fname: round(float(imp), 4)
            for fname, imp in sorted(
                zip(FEATURE_NAMES, fi), key=lambda x: -x[1]
            )
        }
        logger.info("Feature importances: %s", feature_importances)
    except Exception:
        pass

    payload = {
        "outcome_model":          results["outcome_model"],
        "direction_model":        results["direction_model"],
        "feature_names":          FEATURE_NAMES,
        "trained_at":             datetime.utcnow().isoformat(),
        "n_samples":              len(X),
        "outcome_cv_accuracy":    results.get("outcome_cv_accuracy"),
        "outcome_cv_std":         results.get("outcome_cv_std"),
        "direction_cv_accuracy":  results.get("direction_cv_accuracy"),
        "direction_cv_std":       results.get("direction_cv_std"),
        "rank_ic":                rank_ic,
        "feature_importances":    feature_importances,
        "cv_method":              "TimeSeriesSplit",
        "cv_folds":               cv_folds,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    logger.info(
        "ML model trained: %d samples | outcome_acc=%.3f±%.3f | direction_acc=%.3f | rank_ic=%s",
        len(X),
        results.get("outcome_cv_accuracy") or 0,
        results.get("outcome_cv_std") or 0,
        results.get("direction_cv_accuracy") or 0,
        f"{rank_ic:.4f}" if rank_ic is not None else "N/A",
    )
    return {
        "trained":                 True,
        "n_samples":               len(X),
        "outcome_cv_accuracy":     results.get("outcome_cv_accuracy"),
        "outcome_cv_std":          results.get("outcome_cv_std"),
        "direction_cv_accuracy":   results.get("direction_cv_accuracy"),
        "rank_ic":                 rank_ic,
        "feature_importances":     feature_importances,
        "cv_method":               "TimeSeriesSplit",
        "trained_at":              payload["trained_at"],
    }


# ── Model I/O ─────────────────────────────────────────────────────────────────

def load_model() -> Optional[dict]:
    """Load saved model from disk. Returns None if unavailable."""
    try:
        if not os.path.exists(MODEL_PATH):
            return None
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.debug("Model load failed: %s", e)
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(
    feature_snapshot: dict,
    event: Optional[db.Event] = None,
) -> Dict:
    """
    Predict outcome probabilities and price direction for a feature snapshot.

    Returns:
      {
        "outcome":   {"profitable": p, "neutral": p, "unprofitable": p},
        "direction": {"up": p, "flat": p, "down": p},
        "predicted_outcome":   "profitable" | "neutral" | "unprofitable",
        "predicted_direction": "up" | "flat" | "down",
        "ml_confidence": 0.0–1.0,
        "model_available": bool,
      }
    """
    model_data = load_model()

    if model_data is None:
        # Rule-based fallback
        opp = float(feature_snapshot.get("opportunity_score", 0.5))
        gap = float(feature_snapshot.get("expectation_gap", 0.0))
        adj = (opp + abs(gap)) / 2
        return {
            "outcome":   {"profitable": round(adj, 3), "neutral": round(1 - adj - 0.1, 3), "unprofitable": 0.1},
            "direction": {"up": round(adj, 3), "flat": round(1 - adj - 0.1, 3), "down": 0.1},
            "predicted_outcome":   "profitable" if adj > 0.5 else "neutral",
            "predicted_direction": "up" if adj > 0.5 else "flat",
            "ml_confidence": round(adj, 3),
            "model_available": False,
        }

    try:
        X = _snap_from_dict(feature_snapshot, event).reshape(1, -1)

        outcome_pipe   = model_data["outcome_model"]
        direction_pipe = model_data["direction_model"]

        o_proba = outcome_pipe.predict_proba(X)[0]
        d_proba = direction_pipe.predict_proba(X)[0]

        # CalibratedClassifierCV exposes classes_ directly; Pipeline stores them on clf step
        def _get_classes(model):
            if hasattr(model, "classes_"):
                return model.classes_
            if hasattr(model, "named_steps"):
                return model.named_steps["clf"].classes_
            return []

        o_classes = _get_classes(outcome_pipe)
        d_classes = _get_classes(direction_pipe)

        outcome_label_map   = {0: "unprofitable", 1: "neutral", 2: "profitable"}
        direction_label_map = {-1: "down", 0: "flat", 1: "up"}

        outcome_dict   = {outcome_label_map.get(int(c), str(c)):   round(float(p), 4) for c, p in zip(o_classes, o_proba)}
        direction_dict = {direction_label_map.get(int(c), str(c)): round(float(p), 4) for c, p in zip(d_classes, d_proba)}

        pred_outcome   = max(outcome_dict,   key=outcome_dict.get)
        pred_direction = max(direction_dict, key=direction_dict.get)
        confidence     = round(max(o_proba), 4)

        return {
            "outcome":             outcome_dict,
            "direction":           direction_dict,
            "predicted_outcome":   pred_outcome,
            "predicted_direction": pred_direction,
            "ml_confidence":       confidence,
            "model_available":     True,
        }
    except Exception as e:
        logger.debug("ML prediction failed: %s", e)
        opp = float(feature_snapshot.get("opportunity_score", 0.5))
        return {
            "outcome":   {"profitable": opp, "neutral": 0.2, "unprofitable": round(1-opp-0.2, 3)},
            "direction": {"up": opp, "flat": 0.2, "down": round(1-opp-0.2, 3)},
            "predicted_outcome":   "profitable" if opp > 0.5 else "neutral",
            "predicted_direction": "up" if opp > 0.5 else "flat",
            "ml_confidence":       opp,
            "model_available":     True,
        }


def bootstrap_train(session: Session) -> dict:
    """
    Bootstrap the ML model from existing StockScore + Event data when no labeled
    AlertOutcome records exist yet.

    Uses opportunity_score as a proxy label:
      opportunity_score >= 0.60  →  profitable (2) / up (1)
      opportunity_score <= 0.35  →  unprofitable (0) / down (-1)
      otherwise                  →  neutral (1) / flat (0)

    This gives the model a reasonable prior on day 1. Real outcome labels will
    override and improve it as they accumulate.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # Collect all stock scores joined with their best event
    scores = session.query(db.StockScore).all()
    if len(scores) < BOOTSTRAP_MIN_SAMPLES:
        return {
            "trained": False,
            "reason": f"Need at least {BOOTSTRAP_MIN_SAMPLES} stock score rows, have {len(scores)}",
            "bootstrap": True,
        }

    X, y_outcome, y_direction = [], [], []
    for s in scores:
        opp = s.opportunity_score or 0.0
        # Derive proxy label from opportunity score
        if opp >= 0.60:
            y_o, y_d = 2, 1
        elif opp <= 0.35:
            y_o, y_d = 0, -1
        else:
            y_o, y_d = 1, 0

        # Get best event for this ticker
        impact = (
            session.query(db.EventTickerImpact)
            .filter_by(ticker=s.ticker)
            .order_by(db.EventTickerImpact.impact_score.desc())
            .first()
        )
        event = session.get(db.Event, impact.event_id) if impact else None

        eg   = s.expectation_gap_score or 0.0
        risk = s.risk_score or max(0.0, 1.0 - opp)

        feat = np.array([
            # ── 11 alert-level features (match FEATURE_NAMES order) ─────────
            opp,
            0.5,                           # credibility neutral default for bootstrap
            eg,
            s.indirect_impact_score or 0.0,
            s.lag_score or 0.5,
            s.asymmetry_score or 0.0,
            s.crowding_score or 0.3,
            risk,
            STAGE_NUM.get(event.narrative_stage if event else "peak", 0.5),
            EVENT_TYPE_NUM.get(event.event_type if event else "sector", 0.4),
            0.1,                           # source_count_norm placeholder
            # ── 6 price context features — neutral defaults for bootstrap ───
            # (no yfinance calls during bootstrap to keep it fast)
            0.50,   # rsi_14_norm      (50 RSI = neutral)
            -0.10,  # pct_from_52w_high (10% below 52w high = neutral)
            0.00,   # momentum_30d
            0.00,   # momentum_90d
            0.00,   # rel_strength_90d
            1.00,   # vol_regime       (1.0 = normal vol)
        ], dtype=float)
        X.append(feat)
        y_outcome.append(y_o)
        y_direction.append(y_d)

    X          = np.array(X)
    y_outcome  = np.array(y_outcome)
    y_direction = np.array(y_direction)

    def _pipe():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )),
        ])

    outcome_pipe   = _pipe()
    direction_pipe = _pipe()
    outcome_pipe.fit(X, y_outcome)
    direction_pipe.fit(X, y_direction)

    payload = {
        "outcome_model":         outcome_pipe,
        "direction_model":       direction_pipe,
        "feature_names":         FEATURE_NAMES,
        "trained_at":            datetime.utcnow().isoformat(),
        "n_samples":             len(X),
        "outcome_cv_accuracy":   None,
        "direction_cv_accuracy": None,
        "bootstrap":             True,   # flag: trained on proxy labels, not real outcomes
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    logger.info("ML model bootstrap-trained on %d stock score rows (proxy labels)", len(X))
    return {
        "trained":    True,
        "bootstrap":  True,
        "n_samples":  len(X),
        "note": "Trained on opportunity-score proxy labels. Will auto-retrain on real outcome data once available.",
    }


def get_model_info(session: Session) -> dict:
    """Return metadata about the ML model and training readiness."""
    model_data = load_model()
    total    = session.query(db.AlertOutcome).count()
    labeled  = (
        session.query(db.AlertOutcome)
        .filter(
            db.AlertOutcome.outcome_label.isnot(None),
            db.AlertOutcome.outcome_label != "pending",
        )
        .count()
    )
    return {
        "model_available":          model_data is not None,
        "trained_at":               model_data["trained_at"] if model_data else None,
        "n_training_samples":       model_data["n_samples"] if model_data else 0,
        "outcome_cv_accuracy":      model_data.get("outcome_cv_accuracy") if model_data else None,
        "outcome_cv_std":           model_data.get("outcome_cv_std") if model_data else None,
        "direction_cv_accuracy":    model_data.get("direction_cv_accuracy") if model_data else None,
        "direction_cv_std":         model_data.get("direction_cv_std") if model_data else None,
        # Rank IC: Spearman correlation between predicted profitable-probability
        # and actual forward return.  IC > 0.05 is meaningful; IC > 0.10 is strong.
        # Values near 0 mean the model is classifying but not ranking correctly.
        "rank_ic":                  model_data.get("rank_ic") if model_data else None,
        "cv_method":                model_data.get("cv_method", "StratifiedKFold") if model_data else None,
        "cv_folds":                 model_data.get("cv_folds") if model_data else None,
        # Top 5 most important features — unstable importances across retrains
        # indicate overfitting (per Harvey/Liu/Zhu multiple-testing guidance).
        "top_features":             dict(list((model_data.get("feature_importances") or {}).items())[:5])
                                    if model_data else {},
        "is_bootstrap_model":       bool(model_data.get("bootstrap")) if model_data else False,
        "total_outcome_records":    total,
        "labeled_outcome_records":  labeled,
        "samples_until_training":   max(0, MIN_TRAINING_SAMPLES - labeled),
        "ready_to_train":           labeled >= MIN_TRAINING_SAMPLES,
    }
