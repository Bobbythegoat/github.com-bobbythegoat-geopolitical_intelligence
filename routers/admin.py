"""
routers/admin.py
-----------------
Admin/operations endpoints for manual system control.

POST /admin/refresh              — trigger a full feed ingestion cycle immediately
POST /admin/recalculate-scores   — recompute all stock scores from existing event data
POST /admin/retrain              — retrain the ML model on current labeled outcomes
GET  /admin/status               — current system health snapshot
"""

import threading
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db, SessionLocal

router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _do_ingest_cycle() -> dict:
    """Run one ingestion cycle and return a result summary."""
    try:
        from services.ingestion import ingest_all_feeds, DEFAULT_FEEDS
        s = SessionLocal()
        try:
            results = ingest_all_feeds(s)
            total = sum(results.values())
        finally:
            s.close()
        return {"ok": True, "new_articles": total, "feeds": len(DEFAULT_FEEDS)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _do_recalculate_scores() -> dict:
    """
    Re-run _update_stock_scores() for every ticker that has recent event exposure.
    This pushes the latest EventTickerImpact values back into stock_scores without
    needing to ingest new articles.
    """
    s = SessionLocal()
    try:
        import database as db
        from services.processing import _update_stock_scores

        # Count distinct tickers for reporting purposes
        ticker_count = (
            s.query(db.EventTickerImpact.ticker)
            .distinct()
            .count()
        )
        _update_stock_scores(s)
        s.commit()
        return {"ok": True, "tickers_recalculated": ticker_count}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/refresh", summary="Trigger a manual feed ingestion cycle")
def trigger_refresh():
    """
    Immediately runs a full RSS ingestion cycle in the background.
    New articles are processed, events updated, and stock scores recalculated.
    Returns quickly — the actual work runs in a daemon thread.
    """
    def _worker():
        result = _do_ingest_cycle()
        print(f"[Admin/refresh] {result}")

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "status": "started",
        "message": "Feed ingestion cycle started in background. Check server logs for progress.",
        "triggered_at": datetime.utcnow().isoformat(),
    }


@router.post("/recalculate-scores", summary="Force recalculation of all stock scores")
def trigger_recalculate():
    """
    Re-runs the stock score computation for every ticker with active event exposure.
    Use this if scores appear stale after a fresh ingestion but before the next
    scheduled cycle has run.
    Returns synchronously with a count of tickers updated.
    """
    result = _do_recalculate_scores()
    return {
        "status": "ok" if result["ok"] else "error",
        "tickers_recalculated": result.get("tickers_recalculated", 0),
        "error": result.get("error"),
        "completed_at": datetime.utcnow().isoformat(),
    }


@router.post("/retrain", summary="Retrain the ML model on current labeled outcomes")
def trigger_retrain(session: Session = Depends(get_db)):
    """
    Retrains the GradientBoosting outcome predictor on all currently labeled
    AlertOutcome records. Use this after accumulating 30+ new outcomes since
    the last training run, or after running relabel_and_retrain.py.
    """
    from services.ml_predictor import train_model
    result = train_model(session)
    return {
        "status": "trained" if result.get("trained") else "skipped",
        "n_samples": result.get("n_samples", 0),
        "outcome_cv_accuracy": result.get("outcome_cv_accuracy"),
        "direction_cv_accuracy": result.get("direction_cv_accuracy"),
        "reason": result.get("reason"),
        "trained_at": result.get("trained_at"),
    }


@router.post("/universe-sweep", tags=["Admin"])
def manual_universe_sweep(session: Session = Depends(get_db)):
    """Manually trigger ETF holdings sweep to expand tracked universe."""
    try:
        from services.universe_manager import sweep_etf_holdings
        n = sweep_etf_holdings(session)
        return {"status": "ok", "new_tickers_added": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", summary="System health and ingestion status")
def get_status(session: Session = Depends(get_db)):
    """
    Returns a snapshot of: model info, labeled outcome counts, recent alert counts,
    and whether periodic ingestion is expected to be running.
    """
    import database as db
    from sqlalchemy import text
    from services.ml_predictor import get_model_info

    ml = get_model_info(session)

    total_events  = session.query(db.Event).count()
    total_alerts  = session.query(db.Alert).count()
    total_outcomes = session.query(db.AlertOutcome).count()
    labeled_outcomes = (
        session.query(db.AlertOutcome)
        .filter(
            db.AlertOutcome.outcome_label.isnot(None),
            db.AlertOutcome.outcome_label != "pending",
        )
        .count()
    )
    pending_outcomes = total_outcomes - labeled_outcomes

    # Latest article timestamp — column is "timestamp" in the articles table
    latest_article = session.execute(
        text("SELECT MAX(timestamp) FROM articles")
    ).scalar()

    try:
        from services.universe_manager import get_universe_status
        universe = get_universe_status(session)
        universe_stats = {
            "total_tickers": universe["total"],
            "by_source": universe["by_source"],
        }
    except Exception:
        universe_stats = {"total_tickers": 0, "by_source": {}}

    return {
        "system": "ok",
        "ingestion": {
            "scheduled_interval_minutes": 30,
            "latest_article_at": latest_article,
        },
        "data": {
            "total_events": total_events,
            "total_alerts": total_alerts,
            "total_outcomes": total_outcomes,
            "labeled_outcomes": labeled_outcomes,
            "pending_outcomes": pending_outcomes,
        },
        "ml_model": {
            "available": ml["model_available"],
            "is_bootstrap": ml["is_bootstrap_model"],
            "trained_at": ml["trained_at"],
            "n_training_samples": ml["n_training_samples"],
            "outcome_cv_accuracy": ml["outcome_cv_accuracy"],
            "direction_cv_accuracy": ml["direction_cv_accuracy"],
        },
        "universe": universe_stats,
        "checked_at": datetime.utcnow().isoformat(),
    }
