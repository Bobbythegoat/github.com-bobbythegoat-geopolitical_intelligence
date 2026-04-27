"""
routers/events.py
-----------------
GET /events            — list events (with optional filters)
GET /events/{id}       — event detail with articles and ticker impacts
GET /articles          — list raw articles with precise filters
POST /events/ingest    — manually submit a raw article
POST /events/ingest-feeds — pull all RSS feeds now
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import database as db
from database import get_db
import models
from services.ingestion import ingest_manual_article, ingest_all_feeds

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("/", response_model=List[models.EventSummary])
def list_events(
    stage: Optional[str] = Query(None, description="Filter by narrative_stage"),
    event_type: Optional[str] = Query(None, description="geopolitical | macro | company | sector"),
    source: Optional[str] = Query(None, description="Filter by source name (partial match)"),
    geography: Optional[str] = Query(None, description="Filter by geography tag (partial match)"),
    ticker: Optional[str] = Query(None, description="Filter to events impacting this ticker"),
    min_credibility: float = Query(0.0, ge=0, le=1),
    hours: Optional[int] = Query(None, description="Only events from last N hours"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db),
):
    """List all clustered events, optionally filtered by multiple dimensions."""
    from datetime import datetime, timedelta
    q = session.query(db.Event)
    if stage:
        q = q.filter(db.Event.narrative_stage == stage.lower())
    if event_type:
        q = q.filter(db.Event.event_type == event_type.lower())
    if geography:
        q = q.filter(db.Event.geography_tags.ilike(f"%{geography}%"))
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(db.Event.timestamp >= cutoff)
    q = q.filter(db.Event.credibility_score >= min_credibility)
    if source:
        # Filter events that have at least one article from this source
        event_ids_with_source = (
            session.query(db.Article.event_id)
            .filter(db.Article.source.ilike(f"%{source}%"))
            .filter(db.Article.event_id.isnot(None))
            .distinct()
            .subquery()
        )
        q = q.filter(db.Event.event_id.in_(event_ids_with_source))
    if ticker:
        event_ids_for_ticker = (
            session.query(db.EventTickerImpact.event_id)
            .filter(db.EventTickerImpact.ticker == ticker.upper())
            .distinct()
            .subquery()
        )
        q = q.filter(db.Event.event_id.in_(event_ids_for_ticker))
    q = q.order_by(db.Event.timestamp.desc())
    return q.offset(offset).limit(limit).all()


@router.get("/articles", response_model=List[models.ArticleOut])
def list_articles(
    source: Optional[str] = Query(None, description="Filter by source name (partial match)"),
    has_event: Optional[bool] = Query(None, description="True=clustered, False=unclustered, None=all"),
    ticker: Optional[str] = Query(None, description="Filter articles containing this ticker keyword"),
    is_primary: Optional[bool] = Query(None, description="Filter primary sources only"),
    hours: Optional[int] = Query(None, description="Only articles from last N hours"),
    search: Optional[str] = Query(None, description="Free text search in headline/content"),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db),
):
    """List raw articles with precise multi-dimension filtering."""
    from datetime import datetime, timedelta
    q = session.query(db.Article)
    if source:
        q = q.filter(db.Article.source.ilike(f"%{source}%"))
    if has_event is True:
        q = q.filter(db.Article.event_id.isnot(None))
    elif has_event is False:
        q = q.filter(db.Article.event_id.is_(None))
    if is_primary is True:
        q = q.filter(db.Article.is_primary_source == 1)
    if hours:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        q = q.filter(db.Article.timestamp >= cutoff)
    if search:
        q = q.filter(
            db.Article.headline.ilike(f"%{search}%") |
            db.Article.content.ilike(f"%{search}%")
        )
    if ticker:
        from services.processing import TICKER_KEYWORDS
        kws = TICKER_KEYWORDS.get(ticker.upper(), [ticker.lower()])
        from sqlalchemy import or_
        conditions = [
            db.Article.headline.ilike(f"%{kw}%") | db.Article.content.ilike(f"%{kw}%")
            for kw in kws[:5]  # limit to top 5 keywords for performance
        ]
        if conditions:
            q = q.filter(or_(*conditions))
    q = q.order_by(db.Article.timestamp.desc())
    return q.offset(offset).limit(limit).all()


@router.get("/sources", tags=["Events"])
def list_sources(session: Session = Depends(get_db)):
    """Return distinct source names for filter dropdowns."""
    from sqlalchemy import distinct
    rows = session.query(distinct(db.Article.source)).order_by(db.Article.source).all()
    return [r[0] for r in rows if r[0]]


@router.get("/{event_id}", response_model=models.EventOut)
def get_event(event_id: int, session: Session = Depends(get_db)):
    """Full event detail including articles and ticker impacts."""
    event = session.get(db.Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.get("/{event_id}/impacts", response_model=List[models.ImpactOut])
def get_event_impacts(event_id: int, session: Session = Depends(get_db)):
    """Return ticker impact scores for a specific event."""
    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .order_by(db.EventTickerImpact.impact_score.desc())
        .all()
    )
    result = []
    for i in impacts:
        ticker_row = session.get(db.Ticker, i.ticker)
        result.append(models.ImpactOut(
            ticker=i.ticker,
            impact_score=i.impact_score,
            company_name=ticker_row.company_name if ticker_row else None,
            sector=ticker_row.sector if ticker_row else None,
        ))
    return result


class ArticleSubmit(models.ArticleBase):
    pass


@router.post("/ingest", response_model=models.ArticleOut, status_code=201)
def ingest_article(
    payload: ArticleSubmit,
    session: Session = Depends(get_db),
):
    """Manually ingest a single article. Pipeline runs synchronously."""
    article = ingest_manual_article(
        headline=payload.headline,
        content=payload.content,
        source=payload.source,
        session=session,
        url=payload.url,
        is_primary=payload.is_primary_source,
        has_official_confirm=payload.has_official_confirm,
    )
    return article


@router.post("/ingest-feeds", status_code=200)
def ingest_feeds(session: Session = Depends(get_db)):
    """
    Pull all RSS feeds now and wait for results.
    Returns a breakdown of how many new articles came from each source.
    """
    results = ingest_all_feeds(session)
    total = sum(results.values())
    return {
        "message": f"Done. {total} new articles ingested.",
        "breakdown": results,
    }
