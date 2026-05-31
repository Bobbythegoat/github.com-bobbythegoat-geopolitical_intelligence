"""
services/outcome_tracker.py
----------------------------
Decision Outcome Tracker (Upgrade Spec §4.6)

Converts the system from static analysis into a learning loop.
Every alert is judged against subsequent price action.

Responsibilities
----------------
1. Record the full feature snapshot at alert creation time (JSON stored in alerts table).
2. Collect forward returns at multiple horizons (15m, 1h, 1d, 3d, 1w, 1m).
3. Compute max adverse excursion, max favourable excursion, realised Sharpe proxy.
4. Auto-label the outcome: profitable | unprofitable | early | late | neutral | invalidated.
5. Expose the labeled data for the adaptive learning engine in services/learning.py.

Data flow
---------
  alert fires
    → outcome_tracker.create_outcome_record()   (pending record created)
    → [time passes]
    → outcome_tracker.update_forward_returns()  (called periodically or via API)
    → outcome_tracker.label_outcome()           (automatic labelling)
    → learning.update_weights()                 (weight update from labelled data)

Usage
-----
The update_forward_returns function expects yfinance to be available.
If not installed the function degrades gracefully.
"""

import json
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from sqlalchemy.orm import Session

import database as db


# ---------------------------------------------------------------------------
# Outcome label thresholds
# ---------------------------------------------------------------------------

# A return beyond these thresholds (in abs pct) counts as meaningful movement
PROFITABLE_RETURN_THRESHOLD    = 0.02   # 2 % gain in aligned direction
UNPROFITABLE_RETURN_THRESHOLD  = -0.02  # 2 % loss in aligned direction
EARLY_MULTIPLIER               = 0.5    # 1d weak but 1w strong = 'early'
SHARPE_PROXY_DENOMINATOR       = 0.15   # rough annualised vol for ratio estimate


# ---------------------------------------------------------------------------
# Feature snapshot helpers
# ---------------------------------------------------------------------------

def build_feature_snapshot(
    event: db.Event,
    score_row: Optional[db.StockScore],
    credibility: float,
    relevance: float,
    tier: str,
) -> dict:
    """
    Build a JSON-serialisable dict of all model features at alert time.
    Stored in alerts.feature_vector_snapshot so later review can trace
    exactly what drove the decision.

    Price context features (rsi_14, pct_from_52w_high, momentum_30d/90d,
    rel_strength_90d, vol_regime) come from yfinance via services.price_context.
    They give the ML model genuine historical market context — the model can
    now learn "bearish alert on oversold stock" vs "bearish alert on stock at
    52-week high" as meaningfully different situations.  These use up to 1 year
    of daily price history and are cached for 4 hours.
    """
    ticker = score_row.ticker if score_row else None

    # Fetch historical price context (4h cached, graceful fallback to neutral)
    price_ctx: dict = {}
    if ticker:
        try:
            from services.price_context import fetch_price_context
            price_ctx = fetch_price_context(ticker)
        except Exception:
            pass

    return {
        "event_id":             event.event_id,
        "event_type":           event.event_type,
        "narrative_stage":      event.narrative_stage,
        "credibility_score":    round(credibility, 4),
        "relevance_score":      round(relevance, 4),
        "tier":                 tier,
        "source_count":         event.source_count,
        "expectation_proxy":    event.expectation_proxy,
        "narrative_inflection": event.narrative_inflection,
        "attention_velocity":   event.attention_velocity,
        "contradiction_rate":   event.contradiction_rate,
        "opportunity_score":    score_row.opportunity_score    if score_row else None,
        "crowding_score":       score_row.crowding_score       if score_row else None,
        "lag_score":            score_row.lag_score            if score_row else None,
        "asymmetry_score":      score_row.asymmetry_score      if score_row else None,
        # Stored under BOTH keys so ml_predictor.py can find it either way
        "expectation_gap":      score_row.expectation_gap_score if score_row else None,
        "expectation_gap_score":score_row.expectation_gap_score if score_row else None,
        "indirect_impact_score":score_row.indirect_impact_score if score_row else None,
        # risk_score derived from impact credibility (used by ml_predictor feature vector)
        "risk_score":           score_row.risk_score            if score_row else None,
        "exposure_score":       score_row.exposure_score        if score_row else None,
        # ── Historical price context (yfinance, up to 1y of daily data) ──────
        "rsi_14":            price_ctx.get("rsi_14",             50.0),
        "pct_from_52w_high": price_ctx.get("pct_from_52w_high", -0.10),
        "momentum_30d":      price_ctx.get("momentum_30d",        0.0),
        "momentum_90d":      price_ctx.get("momentum_90d",        0.0),
        "rel_strength_90d":  price_ctx.get("rel_strength_90d",    0.0),
        "vol_regime":        price_ctx.get("vol_regime",           1.0),
        "recorded_at":       datetime.utcnow().isoformat(),
    }


def build_component_snapshot(score_row: Optional[db.StockScore]) -> dict:
    """Compact sub-score snapshot for the alerts.component_scores_snapshot field."""
    if score_row is None:
        return {}
    return {
        "exposure":       score_row.exposure_score,
        "impact":         score_row.impact_score,
        "narrative":      score_row.narrative_score,
        "lag":            score_row.lag_score,
        "asymmetry":      score_row.asymmetry_score,
        "crowding":       score_row.crowding_score,
        "risk":           score_row.risk_score,
        "expectation_gap": score_row.expectation_gap_score,
        "indirect_impact": score_row.indirect_impact_score,
        "opportunity":    score_row.opportunity_score,
        "decision_bucket": score_row.decision_bucket,
    }


# ---------------------------------------------------------------------------
# Outcome record management
# ---------------------------------------------------------------------------

def create_outcome_record(
    alert_id: int,
    ticker: str,
    session: Session,
    event_id: Optional[int] = None,
) -> db.AlertOutcome:
    """
    Create a pending outcome record immediately after an alert fires.
    Forward returns are filled in later by update_forward_returns().
    """
    # Avoid duplicates
    existing = (
        session.query(db.AlertOutcome)
        .filter_by(alert_id=alert_id, ticker=ticker)
        .first()
    )
    if existing:
        return existing

    outcome = db.AlertOutcome(
        alert_id=alert_id,
        ticker=ticker,
        event_id=event_id,
        outcome_label="pending",
    )
    session.add(outcome)
    session.commit()
    session.refresh(outcome)
    return outcome


def update_forward_returns(
    outcome_id: int,
    session: Session,
) -> Optional[db.AlertOutcome]:
    """
    Fetch live price data for the alert's ticker and compute forward returns.
    Requires yfinance.  If unavailable returns None.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    outcome = session.get(db.AlertOutcome, outcome_id)
    if outcome is None or outcome.outcome_label != "pending":
        return outcome

    alert    = session.get(db.Alert, outcome.alert_id)
    if alert is None:
        return outcome

    alert_ts = alert.timestamp
    now      = datetime.utcnow()
    elapsed  = (now - alert_ts).total_seconds() / 3600.0   # hours since alert

    try:
        ticker_yf = yf.Ticker(outcome.ticker)
        # Pull 2-month history at 1-hour intervals where possible
        hist = ticker_yf.history(period="2mo", interval="1d")
        if hist.empty:
            return outcome
        closes = hist["Close"].tolist()
        dates  = [d.to_pydatetime() for d in hist.index]
    except Exception:
        return outcome

    # Find the alert day's close price as baseline
    baseline_price = None
    for i, d in enumerate(dates):
        if d.date() >= alert_ts.date():
            baseline_price = closes[i]
            break

    if baseline_price is None or baseline_price == 0:
        return outcome

    def _ret(n_days: int) -> Optional[float]:
        target = alert_ts + timedelta(days=n_days)
        for i, d in enumerate(dates):
            if d.date() >= target.date() and i < len(closes):
                return round((closes[i] - baseline_price) / baseline_price, 5)
        return None

    outcome.forward_return_1d  = _ret(1)  if elapsed >= 24  else None
    outcome.forward_return_3d  = _ret(3)  if elapsed >= 72  else None
    outcome.forward_return_1w  = _ret(5)  if elapsed >= 120 else None
    outcome.forward_return_1m  = _ret(21) if elapsed >= 504 else None

    # Max favourable / adverse excursion over 1-week window
    window_closes = []
    start = alert_ts.date()
    for i, d in enumerate(dates):
        if start <= d.date() <= (alert_ts + timedelta(days=7)).date():
            window_closes.append(closes[i])

    if len(window_closes) > 1:
        rets = [(c - baseline_price) / baseline_price for c in window_closes]
        outcome.max_favorable_excursion = round(max(rets), 5)
        outcome.max_adverse_excursion   = round(min(rets), 5)
        vol = _std(rets)
        mean_ret = sum(rets) / len(rets)
        outcome.realized_volatility = round(vol, 5)
        outcome.realized_sharpe_proxy = (
            round(mean_ret / max(vol, 0.001), 4) if vol else None
        )

    # Auto-label using direction-aware logic
    outcome.outcome_label = _auto_label(outcome, alert, session=session)
    session.commit()
    return outcome


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var  = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def _get_expected_direction(outcome: db.AlertOutcome, session: Session) -> int:
    """
    Determine the expected price direction for this alert from the
    EventTickerImpact record created by the causal engine.

    Returns:
      +1  if the alert expected the stock to go UP   (bullish causal impact)
      -1  if the alert expected the stock to go DOWN  (bearish causal impact)
       0  if no impact record found (direction unknown — skip directional scoring)

    Why this matters:
      A bearish export-restriction alert on NVDA is "profitable" if NVDA falls 3%.
      Under the old direction-blind labeler, that same -3% return was labeled
      "unprofitable" — poisoning the training data in a bull market where all
      stocks tend to rise regardless of signal direction.
    """
    if not outcome.event_id or not outcome.ticker:
        return 0

    impact = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=outcome.event_id, ticker=outcome.ticker)
        .first()
    )

    if impact is None or impact.impact_score is None or impact.impact_score == 0.0:
        # Fall back to feature snapshot asymmetry_score as a proxy
        alert = session.get(db.Alert, outcome.alert_id)
        if alert and alert.feature_vector_snapshot:
            try:
                snap = json.loads(alert.feature_vector_snapshot)
                asym = snap.get("asymmetry_score", 0.0) or 0.0
                if asym > 0.05:
                    return 1
                if asym < -0.05:
                    return -1
            except Exception:
                pass
        return 0  # unknown direction — cannot label directionally

    return 1 if impact.impact_score > 0 else -1


def _auto_label(outcome: db.AlertOutcome, alert: db.Alert,
                session: Optional[Session] = None) -> str:
    """
    Direction-aware outcome labeling.

    Labels the outcome as profitable/unprofitable based on whether the stock
    moved in the DIRECTION the alert predicted, not just whether it went up.

    Example:
      Alert: NVDA bearish (export restriction, impact_score = -0.72)
      1d return: -3.1%  → direction-adjusted: +3.1% → PROFITABLE  ✓
      1d return: +2.5%  → direction-adjusted: -2.5% → UNPROFITABLE ✗

    If no direction can be determined (no impact record), falls back to the
    absolute-return labeling as before.

    Previous bug: the labeler always treated positive returns as "profitable",
    which meant every stock in an AI bull market got labeled "profitable"
    regardless of signal direction → model learned "always say profitable"
    → CV accuracy 1.0 (predicting the dominant class trivially).
    """
    r1d = outcome.forward_return_1d
    r1w = outcome.forward_return_1w

    if r1d is None and r1w is None:
        return "pending"

    # Determine the expected price direction from the causal impact record
    expected_dir = _get_expected_direction(outcome, session) if session else 0

    if expected_dir != 0:
        # Direction-adjusted returns: positive means moved in expected direction
        adj_1d = (r1d * expected_dir) if r1d is not None else None
        adj_1w = (r1w * expected_dir) if r1w is not None else None

        if adj_1d is not None:
            if adj_1d >= PROFITABLE_RETURN_THRESHOLD:
                return "profitable"
            if adj_1d <= UNPROFITABLE_RETURN_THRESHOLD:
                if adj_1w is not None and adj_1w >= PROFITABLE_RETURN_THRESHOLD:
                    return "early"
                return "unprofitable"

        if adj_1w is not None:
            if adj_1w >= PROFITABLE_RETURN_THRESHOLD:
                return "early"
            if adj_1w <= UNPROFITABLE_RETURN_THRESHOLD:
                return "unprofitable"

        return "neutral"

    else:
        # Fallback: no direction information — use absolute returns
        # (same as old logic, applied only when causal impact is unavailable)
        if r1d is not None:
            if r1d >= PROFITABLE_RETURN_THRESHOLD:
                return "profitable"
            if r1d <= UNPROFITABLE_RETURN_THRESHOLD:
                if r1w is not None and r1w >= PROFITABLE_RETURN_THRESHOLD:
                    return "early"
                return "unprofitable"

        if r1w is not None:
            if r1w >= PROFITABLE_RETURN_THRESHOLD:
                return "early"
            if r1w <= UNPROFITABLE_RETURN_THRESHOLD:
                return "unprofitable"

        return "neutral"


# ---------------------------------------------------------------------------
# Batch update (for periodic sweep)
# ---------------------------------------------------------------------------

def run_outcome_update_sweep(session: Session, max_records: int = 200) -> int:
    """
    Update forward returns for the oldest pending outcome records.
    Called periodically (e.g. daily).
    Returns number of records updated.
    """
    q = (
        session.query(db.AlertOutcome)
        .filter_by(outcome_label="pending")
        .order_by(db.AlertOutcome.timestamp)
    )
    if max_records is not None:
        q = q.limit(max_records)
    pending = q.all()
    updated = 0
    for record in pending:
        result = update_forward_returns(record.id, session)
        if result and result.outcome_label != "pending":
            updated += 1
    return updated


def bulk_label_samples(session: Session, max_age_days: int = 90) -> dict:
    """
    Efficiently labels all pending outcomes by batch-fetching prices for all
    tickers at once with yfinance.download() instead of one API call per record.

    This is the fast path for labeling 100-1000+ samples — reduces N yfinance
    API calls down to a single batch request per ticker group.

    Returns dict with counts: updated, labeled, skipped, total_pending.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not available", "updated": 0, "labeled": 0}

    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    pending = (
        session.query(db.AlertOutcome)
        .filter(db.AlertOutcome.outcome_label == "pending")
        .all()
    )

    if not pending:
        return {"updated": 0, "labeled": 0, "skipped": 0, "total_pending": 0,
                "message": "No pending outcomes found"}

    tickers = list({o.ticker for o in pending if o.ticker})
    if not tickers:
        return {"updated": 0, "labeled": 0, "skipped": 0, "total_pending": len(pending),
                "message": "No valid tickers in pending outcomes"}

    print(f"[bulk_label] Fetching price history for {len(tickers)} tickers, {len(pending)} pending outcomes...")

    # Single batch download for all tickers — drastically faster than individual calls
    hist_by_ticker: dict = {}
    try:
        if len(tickers) == 1:
            raw = yf.download(tickers[0], period="3mo", interval="1d", progress=False)
            if not raw.empty:
                hist_by_ticker[tickers[0]] = raw
        else:
            raw = yf.download(tickers, period="3mo", interval="1d",
                              group_by="ticker", progress=False, threads=True)
            for t in tickers:
                try:
                    ticker_df = raw[t] if len(tickers) > 1 else raw
                    if not ticker_df.empty:
                        hist_by_ticker[t] = ticker_df
                except (KeyError, Exception):
                    pass
    except Exception as e:
        return {"error": f"yfinance batch download failed: {e}", "updated": 0, "labeled": 0}

    updated = 0
    labeled = 0
    skipped = 0

    for outcome in pending:
        if not outcome.ticker:
            skipped += 1
            continue

        ticker_hist = hist_by_ticker.get(outcome.ticker)
        if ticker_hist is None or ticker_hist.empty:
            skipped += 1
            continue

        try:
            alert = session.get(db.Alert, outcome.alert_id)
            if not alert:
                skipped += 1
                continue

            alert_ts = alert.timestamp
            now = datetime.utcnow()
            elapsed = (now - alert_ts).total_seconds() / 3600.0

            closes = ticker_hist["Close"].tolist()
            dates = [d.to_pydatetime() for d in ticker_hist.index]

            # Find baseline price at alert date
            baseline_price = None
            for i, d in enumerate(dates):
                d_date = d.date() if hasattr(d, "date") else d
                if d_date >= alert_ts.date():
                    baseline_price = closes[i]
                    break

            if not baseline_price or baseline_price == 0:
                skipped += 1
                continue

            def _ret(n_days: int):
                target_date = (alert_ts + timedelta(days=n_days)).date()
                for i, d in enumerate(dates):
                    d_date = d.date() if hasattr(d, "date") else d
                    if d_date >= target_date and i < len(closes):
                        return round((closes[i] - baseline_price) / baseline_price, 5)
                return None

            outcome.forward_return_1d = _ret(1)  if elapsed >= 24  else None
            outcome.forward_return_3d = _ret(3)  if elapsed >= 72  else None
            outcome.forward_return_1w = _ret(5)  if elapsed >= 120 else None
            outcome.forward_return_1m = _ret(21) if elapsed >= 504 else None

            # Compute excursion stats
            window_closes = []
            start_date = alert_ts.date()
            end_date = (alert_ts + timedelta(days=7)).date()
            for i, d in enumerate(dates):
                d_date = d.date() if hasattr(d, "date") else d
                if start_date <= d_date <= end_date:
                    window_closes.append(closes[i])

            if len(window_closes) > 1:
                rets = [(c - baseline_price) / baseline_price for c in window_closes]
                outcome.max_favorable_excursion = round(max(rets), 5)
                outcome.max_adverse_excursion   = round(min(rets), 5)
                vol = _std(rets)
                mean_ret = sum(rets) / len(rets)
                outcome.realized_volatility = round(vol, 5)
                outcome.realized_sharpe_proxy = (
                    round(mean_ret / max(vol, 0.001), 4) if vol else None
                )

            outcome.outcome_label = _auto_label(outcome, alert, session=session)
            updated += 1
            if outcome.outcome_label != "pending":
                labeled += 1

        except Exception as e:
            skipped += 1
            continue

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        return {"error": f"Commit failed: {e}", "updated": updated, "labeled": labeled}

    print(f"[bulk_label] Done: {labeled} labeled, {updated} updated, {skipped} skipped")
    return {
        "updated": updated,
        "labeled": labeled,
        "skipped": skipped,
        "total_pending": len(pending),
        "tickers_fetched": len(hist_by_ticker),
    }


# ---------------------------------------------------------------------------
# Outcome statistics for the learning engine
# ---------------------------------------------------------------------------

def get_labeled_outcomes(
    session: Session,
    regime_label: Optional[str] = None,
    min_records: int = 10,
) -> List[dict]:
    """
    Return all reviewed/auto-labeled outcome records with their feature snapshots.
    Used by services/learning.py to update factor weights.
    """
    q = session.query(db.AlertOutcome).filter(
        db.AlertOutcome.outcome_label.notin_(["pending"])
    )
    outcomes = q.all()

    result = []
    for o in outcomes:
        alert = session.get(db.Alert, o.alert_id)
        if alert is None:
            continue
        if regime_label and alert.regime_label != regime_label:
            continue
        # Decode feature vector
        features = {}
        if alert.feature_vector_snapshot:
            try:
                features = json.loads(alert.feature_vector_snapshot)
            except Exception:
                pass
        result.append({
            "outcome_id":    o.id,
            "alert_id":      o.alert_id,
            "ticker":        o.ticker,
            "outcome_label": o.outcome_label,
            "forward_1d":    o.forward_return_1d,
            "forward_1w":    o.forward_return_1w,
            "sharpe":        o.realized_sharpe_proxy,
            "features":      features,
            "regime_label":  alert.regime_label,
        })

    if len(result) < min_records:
        return []   # not enough data to learn from yet

    return result
