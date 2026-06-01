"""
main.py — Geopolitical & Market Intelligence System
====================================================
FastAPI entry point.

Run:
    uvicorn main:app --reload --port 8000

Then open:
    http://localhost:8000          → dashboard UI
    http://localhost:8000/docs     → Swagger API explorer
    http://localhost:8000/redoc    → ReDoc API docs
"""

import math
import json as _json
import os
import logging
import threading
import warnings
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Library noise suppression (must run before importing the libraries below)
# ---------------------------------------------------------------------------
# yfinance prints raw "HTTP Error 404: ..." lines via its own logger and emits
# DeprecationWarnings for `Ticker.earnings` from internal helpers we don't call
# directly. Pydantic v2 also warns when a field name starts with `model_`.
# These are all noise we don't act on, so quiet them globally.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"yfinance(\..*)?",
)
warnings.filterwarnings(
    "ignore",
    message=r'Field "model_.*" .* has conflict with protected namespace "model_"',
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from database import create_tables
from routers import events, stocks, alerts, briefs, admin, commodities, commodity_reference

# Global lock prevents concurrent ingestion cycles (scheduler + manual refresh)
_INGEST_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# NaN-safe JSON response
# ---------------------------------------------------------------------------
# yfinance and sklearn can produce float('nan') values that Python's standard
# json.dumps rejects with "Out of range float values are not JSON compliant".
# This custom response class sanitizes the payload before serialisation,
# converting NaN / Inf → null so every endpoint is protected globally.

def _sanitize_nan(obj):
    """Recursively replace float NaN / Inf with None for JSON safety."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_nan(v) for v in obj)
    return obj


class NaNSafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return _json.dumps(
            _sanitize_nan(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# ---------------------------------------------------------------------------
# Silence uvicorn's per-request access log.
# All 200 OK lines are suppressed; startup messages, errors and warnings
# are still printed normally.  Set to logging.INFO to restore full logs.
# ---------------------------------------------------------------------------
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Background worker helpers (keep startup non-blocking)
# ---------------------------------------------------------------------------

def _run_ingestion_cycle(label: str = "Scheduled") -> int:
    """
    Run one full ingestion cycle across all configured RSS feeds.
    Returns the number of new articles ingested.
    Each feed gets its own DB session for isolation.
    Guarded by _INGEST_LOCK so scheduler and manual refresh cannot overlap.
    """
    if not _INGEST_LOCK.acquire(blocking=False):
        print(f"  [{label}] Skipped — ingestion already in progress.")
        return 0

    from database import SessionLocal
    from services.ingestion import ingest_rss_feed, DEFAULT_FEEDS

    total = 0
    try:
        for feed in DEFAULT_FEEDS:
            s = SessionLocal()
            try:
                ids = ingest_rss_feed(
                    feed["url"], feed["name"], s,
                    feed.get("is_primary", 0),
                    strict_filter=feed.get("strict_filter", False),
                )
                total += len(ids)
            except Exception as e:
                print(f"  [{label}] SKIPPED {feed['name']}: {e}")
            finally:
                try:
                    s.rollback()
                except Exception:
                    pass
                s.close()
    finally:
        _INGEST_LOCK.release()
    return total


def _background_ingest():
    """
    Fetch live RSS feeds in a background thread — server starts immediately.
    Each feed runs in its own DB session so one failure cannot poison the rest.
    """
    import time
    time.sleep(1)   # brief pause so the server finishes binding its port first
    try:
        from services.ingestion import DEFAULT_FEEDS
        print("\n=== Starting feed ingestion (server already accepting requests) ===")
        total = _run_ingestion_cycle("Startup")
        print(f"=== Ingestion complete: {total} new article(s) across {len(DEFAULT_FEEDS)} feeds ===")
        print("=== Dashboard: http://localhost:8000  |  API docs: http://localhost:8000/docs ===\n")
    except Exception as e:
        print(f"[background ingest] {e}")


def _scheduled_ingest(interval_minutes: int = 30):
    """
    Periodic re-ingestion loop — runs every `interval_minutes` in a daemon thread.
    This is the fix for frozen stock scores: without this, ingestion only ran
    once at startup and scores never updated afterward.

    Uses ingest_all_feeds() (same path as the manual "⬇ Ingest" button) so that
    process_article → _update_stock_scores fires on every new article, keeping
    scores fresh automatically.
    """
    import time
    # Wait for the startup ingestion to finish first
    time.sleep(interval_minutes * 60)
    while True:
        try:
            from database import SessionLocal as _SL
            from services.ingestion import ingest_all_feeds, DEFAULT_FEEDS
            print(f"\n[Scheduler] Starting scheduled feed ingestion ({interval_minutes}m cycle)...")
            s = _SL()
            try:
                results = ingest_all_feeds(s)
                total = sum(results.values())
            finally:
                s.close()
            print(f"[Scheduler] Cycle complete: {total} new article(s) across {len(DEFAULT_FEEDS)} feeds.")
        except Exception as e:
            print(f"[Scheduler] Ingestion cycle error: {e}")
        time.sleep(interval_minutes * 60)


def _background_bootstrap(is_first_run: bool):
    """
    One-shot startup bootstrap, run in a single background thread so lifespan()
    can yield immediately (server serves requests right away).

    The one-time, write-heavy seeding runs *to completion first*, THEN the
    recurring writer threads (ETF sweep, ingestion, scheduler) are started.
    Sequencing matters: SQLite allows only one writer at a time, so launching
    seeding and the recurring writers simultaneously caused 'database is locked'
    contention and rolled-back seed transactions. Ordering the writes removes it.
    """
    from database import SessionLocal
    from services.learning import ensure_weights_exist
    from services.second_order import seed_relationship_graph
    from services.commodity_tracker import seed_commodities, seed_commodity_relationship_edges

    # ── Phase 1: one-time seeding (sequential, must finish before recurring writers) ──
    s = SessionLocal()
    try:
        if is_first_run:
            print("Empty database detected — seeding sample data...")
            from seed_data import seed
            seed(s)
            print("[bootstrap] Seed complete.")

        n_edges = seed_relationship_graph(s)
        if n_edges:
            print(f"[bootstrap] Added {n_edges} new relationship edges.")

        ensure_weights_exist(s)

        n_commod = seed_commodities(s)
        n_cedges = seed_commodity_relationship_edges(s)
        if n_commod or n_cedges:
            print(f"[bootstrap] Commodities: +{n_commod} new, commodity edges: +{n_cedges}")

        print("[bootstrap] Seeding done.")
    except Exception as e:
        print(f"[bootstrap] seeding error: {e}")
    finally:
        s.close()

    # ── Phase 2: start recurring background workers (after seeding settles) ──
    try:
        from database import SessionLocal as _SL
        from services import signal_scheduler
        threading.Thread(target=_background_etf_sweep, daemon=True).start()
        threading.Thread(target=_background_ml,        args=(_SL,), daemon=True).start()
        threading.Thread(target=_background_ingest,    daemon=True).start()
        threading.Thread(target=_scheduled_ingest,     args=(30,),  daemon=True).start()
        threading.Thread(target=signal_scheduler.start_rescoring_loop,  daemon=True).start()
        threading.Thread(target=signal_scheduler.start_daily_scan_loop, daemon=True).start()
        print("[bootstrap] Background workers started "
              "(ETF sweep, ML, ingestion, scheduler).")
    except Exception as e:
        print(f"[bootstrap] worker start error: {e}")


def _background_etf_sweep():
    import time
    time.sleep(5)   # let ingestion start first
    try:
        from services.universe_manager import sweep_etf_holdings
        from database import SessionLocal as _SL2
        s2 = _SL2()
        n = sweep_etf_holdings(s2)
        s2.close()
        if n:
            print(f"[ETF Sweep] Added {n} new tickers from ETF holdings.")
    except Exception as e:
        print(f"[ETF Sweep] {e}")


def _background_ml(session_factory):
    """
    Bootstrap/train the ML model and pre-warm the sentence-transformer model.
    Runs in a daemon thread so the server stays responsive throughout.

    Timeline
    --------
    t+2s  : pre-warm sentence-transformer (downloads ~80 MB on first run)
    t+??s : ML bootstrap/train once model is cached
    """
    import time
    time.sleep(2)

    # ── Step 1: Pre-warm the sentence-transformer ─────────────────────────
    # This triggers the one-time 80 MB Hugging Face download so that the
    # first article clustering doesn't silently block.  Progress is shown
    # here rather than inside the ingestion thread.
    try:
        from services.semantic_clustering import _get_model
        print("[Startup] Loading sentence-transformer model (first run: ~80 MB download)...")
        model = _get_model()
        if model:
            print("[Startup] Sentence-transformer ready.")
        else:
            print("[Startup] Sentence-transformer unavailable — TF-IDF clustering will be used.")
    except Exception as e:
        print(f"[Startup] Sentence-transformer load skipped: {e}")

    # ── Step 2: ML model bootstrap / real training ────────────────────────
    try:
        from services.ml_predictor import train_model, load_model, bootstrap_train
        s = session_factory()
        if load_model() is None:
            print("[ML] No saved model found — attempting training...")
            result = train_model(s)
            if not result.get("trained"):
                print(f"[ML] Insufficient labeled outcomes ({result.get('n_samples', 0)}) — running bootstrap...")
                boot = bootstrap_train(s)
                if boot.get("trained"):
                    print(f"[ML] Bootstrap complete: {boot['n_samples']} proxy samples. "
                          f"Train the real model at POST /stocks/ml/train once outcomes accumulate.")
                else:
                    print(f"[ML] Bootstrap skipped: {boot.get('reason', 'unknown')}")
            else:
                print(f"[ML] Trained on {result['n_samples']} real outcomes. "
                      f"Outcome accuracy: {result.get('outcome_cv_accuracy', 'n/a')}")
        elif load_model().get("bootstrap"):
            print("[ML] Bootstrap model loaded — checking for real outcome data to upgrade...")
            retrain = train_model(s)
            if retrain.get("trained"):
                print(f"[ML] Upgraded to real-outcome model: {retrain['n_samples']} samples.")
            else:
                print(f"[ML] Still using bootstrap model ({retrain.get('n_samples', 0)} real outcomes so far).")
        else:
            print("[ML] Saved model loaded from disk.")
        s.close()
    except Exception as e:
        print(f"[background ML] {e}")


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

def _migrate_db():
    """
    Safe schema migration for SQLite.
    Adds any missing columns introduced after initial DB creation
    without dropping existing data.

    Optimised: runs ONE PRAGMA per table (not per column) so startup
    overhead is O(tables) instead of O(columns).
    """
    from database import engine
    from sqlalchemy import text

    # (table, column, ddl) — grouped so we do one PRAGMA per table
    migrations = [
        # ── events table ────────────────────────────────────────────────────
        ("events", "event_type",           "ALTER TABLE events ADD COLUMN event_type VARCHAR(32) DEFAULT 'sector'"),
        ("events", "source_count",         "ALTER TABLE events ADD COLUMN source_count INTEGER DEFAULT 1"),
        ("events", "geography_tags",       "ALTER TABLE events ADD COLUMN geography_tags VARCHAR(512)"),
        ("events", "summary",              "ALTER TABLE events ADD COLUMN summary TEXT"),
        ("events", "expectation_proxy",    "ALTER TABLE events ADD COLUMN expectation_proxy FLOAT"),
        ("events", "narrative_shift_score","ALTER TABLE events ADD COLUMN narrative_shift_score FLOAT DEFAULT 0.0"),
        ("events", "contradiction_rate",   "ALTER TABLE events ADD COLUMN contradiction_rate FLOAT DEFAULT 0.0"),
        ("events", "source_breadth",       "ALTER TABLE events ADD COLUMN source_breadth INTEGER DEFAULT 1"),
        ("events", "narrative_inflection", "ALTER TABLE events ADD COLUMN narrative_inflection FLOAT DEFAULT 0.0"),
        ("events", "attention_velocity",   "ALTER TABLE events ADD COLUMN attention_velocity FLOAT DEFAULT 1.0"),
        ("events", "price_response",       "ALTER TABLE events ADD COLUMN price_response FLOAT"),
        # ── stock_scores table ───────────────────────────────────────────────
        ("stock_scores", "exposure_score",        "ALTER TABLE stock_scores ADD COLUMN exposure_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "impact_score",          "ALTER TABLE stock_scores ADD COLUMN impact_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "narrative_score",       "ALTER TABLE stock_scores ADD COLUMN narrative_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "lag_score",             "ALTER TABLE stock_scores ADD COLUMN lag_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "asymmetry_score",       "ALTER TABLE stock_scores ADD COLUMN asymmetry_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "decision_bucket",       "ALTER TABLE stock_scores ADD COLUMN decision_bucket VARCHAR(64) DEFAULT 'Watch'"),
        ("stock_scores", "expectation_gap_score", "ALTER TABLE stock_scores ADD COLUMN expectation_gap_score FLOAT DEFAULT 0.0"),
        ("stock_scores", "indirect_impact_score", "ALTER TABLE stock_scores ADD COLUMN indirect_impact_score FLOAT DEFAULT 0.0"),
        # ── alerts table ─────────────────────────────────────────────────────
        ("alerts", "horizon",                   "ALTER TABLE alerts ADD COLUMN horizon VARCHAR(32) DEFAULT 'short_swing'"),
        ("alerts", "expectation_gap_score",     "ALTER TABLE alerts ADD COLUMN expectation_gap_score FLOAT DEFAULT 0.0"),
        ("alerts", "feature_vector_snapshot",   "ALTER TABLE alerts ADD COLUMN feature_vector_snapshot TEXT"),
        ("alerts", "component_scores_snapshot", "ALTER TABLE alerts ADD COLUMN component_scores_snapshot TEXT"),
        ("alerts", "confidence_score",          "ALTER TABLE alerts ADD COLUMN confidence_score FLOAT DEFAULT 0.5"),
        ("alerts", "regime_label",              "ALTER TABLE alerts ADD COLUMN regime_label VARCHAR(64)"),
        ("alerts", "ml_predicted_outcome",      "ALTER TABLE alerts ADD COLUMN ml_predicted_outcome VARCHAR(32)"),
        ("alerts", "ml_predicted_direction",    "ALTER TABLE alerts ADD COLUMN ml_predicted_direction VARCHAR(16)"),
        ("alerts", "ml_confidence",             "ALTER TABLE alerts ADD COLUMN ml_confidence FLOAT DEFAULT 0.0"),
        # ── articles table ────────────────────────────────────────────────────────
        ("articles", "event_class", "ALTER TABLE articles ADD COLUMN event_class VARCHAR(32) DEFAULT 'geopolitical'"),
        # ── tickers table ─────────────────────────────────────────────────────────
        ("tickers", "source",       "ALTER TABLE tickers ADD COLUMN source VARCHAR(32) DEFAULT 'seed'"),
        ("tickers", "enrolled_at",  "ALTER TABLE tickers ADD COLUMN enrolled_at DATETIME"),
        ("tickers", "market_cap",   "ALTER TABLE tickers ADD COLUMN market_cap FLOAT"),
    ]

    # Group by table — ONE PRAGMA call per table instead of one per column
    from collections import defaultdict as _dd
    by_table: dict = _dd(list)
    for table, column, ddl in migrations:
        by_table[table].append((column, ddl))

    added = 0
    with engine.connect() as conn:
        for table, col_ddls in by_table.items():
            try:
                result       = conn.execute(text(f"PRAGMA table_info({table})"))
                existing_cols = {row[1] for row in result}
                for column, ddl in col_ddls:
                    if column not in existing_cols:
                        try:
                            conn.execute(text(ddl))
                            conn.commit()
                            print(f"Migration: added {table}.{column}")
                            added += 1
                        except Exception as e:
                            print(f"Migration warning ({table}.{column}): {e}")
            except Exception as e:
                print(f"Migration warning (PRAGMA {table}): {e}")

    if added == 0:
        print("Migrations: schema up-to-date, nothing to add.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB schema, then hand all slow work to background threads."""
    # Tier 1: fast, synchronous DB-only operations — complete before yielding
    create_tables()
    _migrate_db()

    from database import SessionLocal
    from sqlalchemy import text
    session = SessionLocal()
    try:
        count = session.execute(text("SELECT COUNT(*) FROM tickers")).scalar()
    except Exception:
        count = 0
    finally:
        session.close()

    # Tier 2: all slow work runs off the event loop. A single bootstrap thread
    # seeds the DB (sequentially), then starts the recurring writer threads — so
    # the server yields immediately AND concurrent writers don't fight over the
    # SQLite write lock during seeding.
    threading.Thread(target=_background_bootstrap, args=(count == 0,), daemon=True).start()
    print("Server ready. Seeding, ML training, and feed ingestion running in background.")

    yield
    # Shutdown: nothing to clean up for SQLite


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Geopolitical & Market Intelligence System",
    description=(
        "Real-time intelligence system that ingests news, evaluates credibility, "
        "maps market impact, and produces structured outputs for decision support."
    ),
    version="2.0.0",
    lifespan=lifespan,
    default_response_class=NaNSafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(events.router)
app.include_router(stocks.router)
app.include_router(alerts.router)
app.include_router(briefs.router)
app.include_router(admin.router)
app.include_router(commodity_reference.router)   # specific /commodities/reference/* first
app.include_router(commodities.router)


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_dashboard():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "message": "Geopolitical & Market Intelligence API",
            "docs": "/docs",
            "redoc": "/redoc",
        }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, access_log=False)
