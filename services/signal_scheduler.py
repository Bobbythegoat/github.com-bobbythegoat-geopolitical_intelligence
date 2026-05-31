"""
services/signal_scheduler.py
------------------------------
Two independent scheduler jobs for consistent signal generation:

Job 1 — start_rescoring_loop(interval_minutes=30):
  Re-scores all tickers every 30 min using fresh price data + adaptive weights.
  Ensures scores stay current even during quiet news periods.

Job 2 — start_daily_scan_loop(scan_hour=16, scan_minute=30):
  Daily price/volume anomaly scan at market close.
  Creates synthetic 'price_anomaly' events for unusual moves,
  flowing them through the standard alert pipeline.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import database as db
from database import SessionLocal
from models import OpportunityFactors
from services.scoring import calculate_opportunity
from services.alerts import try_trigger_alert

logger = logging.getLogger(__name__)

# Maximum tickers to scan per daily anomaly cycle (avoids timeout)
MAX_TICKERS_PER_SCAN = 200

_last_scan_date = None


# ---------------------------------------------------------------------------
# Lazy import helpers to avoid circular imports
# ---------------------------------------------------------------------------

def _get_ingest_lock():
    try:
        from main import _INGEST_LOCK
        return _INGEST_LOCK
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Job 1: Re-scoring loop
# ---------------------------------------------------------------------------

def start_rescoring_loop(interval_minutes: int = 30):
    """Run re-scoring every interval_minutes. Designed for daemon thread.

    Also runs run_event_staleness_cleanup() once every 24 hours to mark old
    events as stale and prune their EventTickerImpact rows.
    """
    _last_cleanup_time: Optional[datetime] = None

    while True:
        time.sleep(interval_minutes * 60)
        try:
            run_rescore_cycle()
        except Exception as e:
            logger.error("Re-score cycle failed: %s", e)

        # Run staleness cleanup at most once per 24 hours
        now = datetime.now(timezone.utc)
        if (
            _last_cleanup_time is None
            or (now - _last_cleanup_time).total_seconds() >= 86400
        ):
            try:
                run_event_staleness_cleanup()
                _last_cleanup_time = now
            except Exception as e:
                logger.error("Staleness cleanup failed: %s", e)


def run_rescore_cycle() -> dict:
    """
    Re-score all tickers using current adaptive weights + fresh price context.
    Returns summary dict: {tickers_scored, alerts_triggered, duration_seconds}
    """
    start_time = time.monotonic()

    # Acquire _INGEST_LOCK non-blocking — skip cycle if ingestion is running
    lock = _get_ingest_lock()
    if lock is not None:
        acquired = lock.acquire(blocking=False)
        if not acquired:
            logger.info("Re-score cycle skipped: ingestion in progress.")
            return {"tickers_scored": 0, "alerts_triggered": 0, "duration_seconds": 0.0}
    else:
        acquired = False

    session = SessionLocal()
    tickers_scored = 0
    alerts_triggered = 0

    try:
        from services.price_context import fetch_price_context
        from services.learning import get_weights, get_current_regime

        tickers = session.query(db.Ticker).all()
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None)
        from datetime import timedelta
        seven_days_ago = cutoff - timedelta(days=7)

        # Get current regime and adaptive weights once per cycle
        try:
            regime = get_current_regime(session)
            weights = get_weights(session, regime_label=regime)
        except Exception as e:
            logger.warning("Could not load adaptive weights (%s); using defaults.", e)
            weights = None

        for ticker_row in tickers:
            ticker_sym = ticker_row.ticker
            try:
                # Fetch fresh price context
                price_ctx = fetch_price_context(ticker_sym)

                # Pull active EventTickerImpacts from last 7 days
                impacts = (
                    session.query(db.EventTickerImpact)
                    .filter(db.EventTickerImpact.ticker == ticker_sym)
                    .join(db.Event, db.Event.event_id == db.EventTickerImpact.event_id)
                    .filter(db.Event.timestamp >= seven_days_ago)
                    .all()
                )

                # Skip if no event exposure
                if not impacts:
                    continue

                # Pick the impact with the largest absolute score
                best_impact = max(impacts, key=lambda i: abs(i.impact_score or 0))
                event = session.get(db.Event, best_impact.event_id)
                if event is None:
                    continue

                # Clamp inputs to OpportunityFactors' declared bounds.
                # momentum_30d is signed (can be negative) but price_reaction_lag
                # requires [0,1]; rel_strength_90d can overshoot ±1 but
                # asymmetry requires [-1,1]; vol_regime ratio can blow past 1.
                _exposure  = min(max(abs(best_impact.impact_score or 0), 0.0), 1.0)
                _cred      = min(max(event.credibility_score or 0.5,    0.0), 1.0)
                _lag       = min(max(price_ctx.get("momentum_30d", 0.0), 0.0), 1.0)
                _asym      = min(max(price_ctx.get("rel_strength_90d", 0.0), -1.0), 1.0)
                _crowd     = min(max((event.source_count or 1) / 20.0,  0.0), 1.0)
                _risk      = min(max(price_ctx.get("vol_regime", 1.0) / 5.0, 0.0), 1.0)

                factors = OpportunityFactors(
                    exposure=_exposure,
                    credibility=_cred,
                    expectation_gap=0.0,          # neutral default
                    indirect_impact=0.0,
                    price_reaction_lag=_lag,       # underreaction proxy
                    asymmetry=_asym,
                    crowding=_crowd,
                    risk=_risk,
                    narrative_stage=event.narrative_stage or "peak",
                )

                opp_score = calculate_opportunity(factors, weights=weights)

                # Upsert StockScore row
                score_row = session.query(db.StockScore).filter_by(ticker=ticker_sym).first()
                if score_row is None:
                    score_row = db.StockScore(ticker=ticker_sym)
                    session.add(score_row)

                score_row.opportunity_score = opp_score
                score_row.exposure_score    = factors.exposure
                score_row.crowding_score    = factors.crowding
                score_row.risk_score        = factors.risk
                score_row.lag_score         = max(0.0, min(1.0, factors.price_reaction_lag))
                score_row.asymmetry_score   = (factors.asymmetry + 1.0) / 2.0
                score_row.impact_score      = abs(best_impact.impact_score or 0)
                score_row.updated_at        = datetime.now(timezone.utc)

                tickers_scored += 1

            except Exception as e:
                logger.warning("Failed to re-score ticker %s: %s", ticker_sym, e)

        session.commit()

        # Alert sweep: check credible events from the last 7 days only.
        # Excluding stale events prevents re-triggering alerts for months-old
        # events that no longer have market relevance.
        try:
            seven_days_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
            credible_events = (
                session.query(db.Event)
                .filter(
                    db.Event.credibility_score > 0.5,
                    db.Event.timestamp >= seven_days_ago,
                )
                .all()
            )
            for event in credible_events:
                try:
                    alert = try_trigger_alert(event.event_id, session)
                    if alert:
                        alerts_triggered += 1
                except Exception as e:
                    logger.warning("Alert trigger failed for event %d: %s", event.event_id, e)
        except Exception as e:
            logger.warning("Alert sweep failed: %s", e)

        # Invalidate stock_recommender cache
        try:
            from services.stock_recommender import invalidate_cache
            invalidate_cache()
        except Exception as e:
            logger.warning("Cache invalidation failed: %s", e)

        duration = round(time.monotonic() - start_time, 2)
        logger.info(
            "Re-score cycle complete: %d tickers scored, %d alerts triggered, %.2fs",
            tickers_scored, alerts_triggered, duration,
        )
        return {
            "tickers_scored":  tickers_scored,
            "alerts_triggered": alerts_triggered,
            "duration_seconds": duration,
        }

    finally:
        session.close()
        if lock is not None and acquired:
            lock.release()


# ---------------------------------------------------------------------------
# Staleness cleanup
# ---------------------------------------------------------------------------

def run_event_staleness_cleanup(session=None) -> dict:
    """
    Housekeeping pass that:
      1. Marks events older than 21 days as narrative_stage = "stale" if not
         already marked.
      2. Removes EventTickerImpact rows linked to stale events so they are
         excluded from the active scoring window.

    Returns a dict: {events_marked_stale, impacts_removed}
    """
    owned_session = session is None
    if owned_session:
        session = SessionLocal()

    events_marked_stale = 0
    impacts_removed = 0

    try:
        cutoff_21d = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=21)

        # Find events older than 21 days that are not already marked stale
        old_events = (
            session.query(db.Event)
            .filter(
                db.Event.timestamp < cutoff_21d,
                db.Event.narrative_stage != "stale",
            )
            .all()
        )

        stale_ids = []
        for event in old_events:
            event.narrative_stage = "stale"
            stale_ids.append(event.event_id)
            events_marked_stale += 1

        if stale_ids:
            # Remove EventTickerImpact rows for stale events
            deleted = (
                session.query(db.EventTickerImpact)
                .filter(db.EventTickerImpact.event_id.in_(stale_ids))
                .delete(synchronize_session=False)
            )
            impacts_removed = deleted or 0

        if owned_session:
            session.commit()

        logger.info(
            "Staleness cleanup: %d events marked stale, %d impacts removed",
            events_marked_stale, impacts_removed,
        )
        return {
            "events_marked_stale": events_marked_stale,
            "impacts_removed": impacts_removed,
        }

    except Exception as e:
        if owned_session:
            session.rollback()
        logger.error("Staleness cleanup failed: %s", e)
        return {"events_marked_stale": 0, "impacts_removed": 0}

    finally:
        if owned_session:
            session.close()


# ---------------------------------------------------------------------------
# Job 2: Daily anomaly scan loop
# ---------------------------------------------------------------------------

def start_daily_scan_loop(scan_hour: int = 16, scan_minute: int = 30):
    """Run daily anomaly scan at scan_hour:scan_minute. Daemon thread."""
    global _last_scan_date
    import time as _time
    while True:
        _time.sleep(60)
        try:
            now = datetime.now()
            today = now.date()
            # Only fire once per calendar day, after the target time
            if (
                today != _last_scan_date
                and (
                    now.hour > scan_hour
                    or (now.hour == scan_hour and now.minute >= scan_minute)
                )
            ):
                logger.info("[DailyScan] Starting anomaly scan...")
                result = run_daily_anomaly_scan()
                _last_scan_date = today
                logger.info("[DailyScan] Complete: %s", result)
        except Exception as e:
            logger.error("[DailyScan] Error: %s", e)


def run_daily_anomaly_scan() -> dict:
    """
    Detect unusual price/volume moves and create synthetic events.
    Returns {tickers_scanned, anomalies_found, events_created, predictions_created}
    """
    import numpy as np
    import yfinance as yf
    from services.anomaly_predictor import batch_predict_anomalies

    session = SessionLocal()
    tickers_scanned = 0
    anomalies_found = 0
    events_created  = 0
    predictions_created = 0

    try:
        tickers = session.query(db.Ticker).limit(MAX_TICKERS_PER_SCAN).all()

        # --- Pre-anomaly prediction pass ---
        # Run BEFORE the post-hoc detection so users see signals in advance.
        all_ticker_syms = [t.ticker for t in tickers]
        try:
            predictions = batch_predict_anomalies(all_ticker_syms, session)
        except Exception as e:
            logger.warning("[DailyScan] Pre-anomaly prediction pass failed: %s", e)
            predictions = []

        for pred in predictions:
            if pred["prediction_score"] < 0.35:
                continue   # only surface moderate+ predictions
            ticker_sym = pred["ticker"]
            try:
                # Check for existing pre_anomaly event in last 24h for this ticker
                cutoff_24h = datetime.utcnow() - timedelta(hours=24)
                existing_pred = (
                    session.query(db.Event)
                    .join(db.EventTickerImpact, db.EventTickerImpact.event_id == db.Event.event_id)
                    .filter(
                        db.EventTickerImpact.ticker == ticker_sym,
                        db.Event.event_type == "pre_anomaly",
                        db.Event.timestamp >= cutoff_24h,
                    )
                    .first()
                )
                if existing_pred:
                    logger.debug(
                        "Skipping duplicate pre_anomaly for %s (event %d already exists)",
                        ticker_sym, existing_pred.event_id,
                    )
                    continue

                direction_word = (
                    "bullish" if pred["direction"] > 0
                    else ("bearish" if pred["direction"] < 0 else "neutral")
                )
                event = db.Event(
                    title=f"Pre-Anomaly Signal: {ticker_sym} ({direction_word})",
                    event_type="pre_anomaly",
                    narrative_stage="emerging",
                    credibility_score=round(pred["prediction_score"], 3),
                    summary=pred["reasoning"],
                    source_count=1,
                )
                session.add(event)
                session.flush()

                impact = db.EventTickerImpact(
                    event_id=event.event_id,
                    ticker=ticker_sym,
                    impact_score=round(pred["direction"] * pred["prediction_score"], 3),
                )
                session.add(impact)
                session.flush()
                session.commit()
                predictions_created += 1

                logger.info(
                    "Pre-anomaly event created for %s (score=%.3f, direction=%s)",
                    ticker_sym, pred["prediction_score"], direction_word,
                )

                try:
                    try_trigger_alert(event.event_id, session)
                except Exception:
                    pass

            except Exception as e:
                logger.warning("Failed to persist pre-anomaly prediction for %s: %s", ticker_sym, e)

        # --- Post-hoc anomaly detection (existing logic) ---
        for ticker_row in tickers:
            ticker_sym = ticker_row.ticker
            tickers_scanned += 1

            try:
                hist = yf.Ticker(ticker_sym).history(period="1mo", interval="1d")
                if hist is None or len(hist) < 3:
                    continue

                closes  = hist["Close"].tolist()
                volumes = hist["Volume"].tolist()

                if len(closes) < 3 or len(volumes) < 3:
                    continue

                # Today vs prior days
                today_return = (closes[-1] - closes[-2]) / closes[-2]
                prior_returns = [
                    (closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes) - 1)
                ]
                avg_return = np.mean(prior_returns)
                std_return = np.std(prior_returns)

                today_vol = volumes[-1]
                avg_vol   = np.mean(volumes[:-1])
                std_vol   = np.std(volumes[:-1])

                price_zscore = (today_return - avg_return) / std_return if std_return > 0 else 0
                vol_zscore   = (today_vol - avg_vol) / std_vol           if std_vol > 0   else 0

                # Anomaly threshold
                if vol_zscore > 2.0 or abs(price_zscore) > 2.0:
                    anomalies_found += 1
                    logger.info(
                        "Anomaly detected for %s — price_z=%.2f, vol_z=%.2f",
                        ticker_sym, price_zscore, vol_zscore,
                    )

                    # Check for existing price_anomaly event for this ticker in last 24h
                    cutoff = datetime.utcnow() - timedelta(hours=24)
                    existing = (
                        session.query(db.Event)
                        .join(db.EventTickerImpact, db.EventTickerImpact.event_id == db.Event.event_id)
                        .filter(
                            db.EventTickerImpact.ticker == ticker_sym,
                            db.Event.event_type == "price_anomaly",
                            db.Event.timestamp >= cutoff,
                        )
                        .first()
                    )
                    if existing:
                        logger.debug("Skipping duplicate price_anomaly for %s (event %d already exists)", ticker_sym, existing.event_id)
                        continue

                    # Create synthetic Event
                    event = db.Event(
                        title=f"Price/Volume Anomaly: {ticker_sym}",
                        event_type="price_anomaly",
                        narrative_stage="emerging",
                        credibility_score=min(abs(vol_zscore) / 4.0, 0.85),
                        summary=(
                            f"{ticker_sym} shows unusual "
                            f"{'volume' if vol_zscore > 2 else 'price'} activity. "
                            f"Price z-score: {price_zscore:.2f}, "
                            f"Volume z-score: {vol_zscore:.2f}"
                        ),
                        source_count=1,
                    )
                    session.add(event)
                    session.flush()  # populate event.event_id

                    # Create EventTickerImpact
                    impact = db.EventTickerImpact(
                        event_id=event.event_id,
                        ticker=ticker_sym,
                        impact_score=min(max(price_zscore / 3.0, -1.0), 1.0),
                    )
                    session.add(impact)
                    session.flush()

                    session.commit()
                    events_created += 1

                    # Attempt alert trigger for the new event
                    try:
                        try_trigger_alert(event.event_id, session)
                    except Exception as e:
                        logger.warning(
                            "Alert trigger failed for anomaly event %d (%s): %s",
                            event.event_id, ticker_sym, e,
                        )

            except Exception as e:
                logger.warning("Anomaly scan failed for ticker %s: %s", ticker_sym, e)

        logger.info(
            "Daily anomaly scan complete: %d scanned, %d anomalies, %d events created, "
            "%d pre-anomaly predictions created",
            tickers_scanned, anomalies_found, events_created, predictions_created,
        )
        return {
            "tickers_scanned":     tickers_scanned,
            "anomalies_found":     anomalies_found,
            "events_created":      events_created,
            "predictions_created": predictions_created,
        }

    finally:
        session.close()


# ---------------------------------------------------------------------------
# API-callable: on-demand pre-anomaly prediction
# ---------------------------------------------------------------------------

def run_anomaly_prediction_now(tickers: List[str] = None) -> dict:
    """
    Run pre-anomaly prediction right now (callable from API or manually).

    Args:
        tickers: Optional list of ticker symbols to run predictions for.
                 If None, uses all tickers in the database (up to MAX_TICKERS_PER_SCAN).

    Returns:
        {
            "predictions": List[dict],   # sorted by score desc, score >= 0.25
            "tickers_evaluated": int,
            "predictions_found": int,
            "duration_seconds": float,
        }
    """
    import time as _time
    from services.anomaly_predictor import batch_predict_anomalies

    start_time = _time.monotonic()
    session = SessionLocal()

    try:
        if tickers is None:
            ticker_rows = session.query(db.Ticker).limit(MAX_TICKERS_PER_SCAN).all()
            tickers = [t.ticker for t in ticker_rows]

        predictions = batch_predict_anomalies(tickers, session)

        duration = round(_time.monotonic() - start_time, 2)
        logger.info(
            "run_anomaly_prediction_now: evaluated %d tickers, %d predictions (score>=0.25), %.2fs",
            len(tickers), len(predictions), duration,
        )
        return {
            "predictions":       predictions,
            "tickers_evaluated": len(tickers),
            "predictions_found": len(predictions),
            "duration_seconds":  duration,
        }

    except Exception as e:
        logger.error("run_anomaly_prediction_now failed: %s", e)
        duration = round(_time.monotonic() - start_time, 2)
        return {
            "predictions":       [],
            "tickers_evaluated": len(tickers) if tickers else 0,
            "predictions_found": 0,
            "duration_seconds":  duration,
            "error":             str(e),
        }

    finally:
        session.close()
