"""
routers/commodities.py
----------------------
Strategic commodity & industrial-input tracker endpoints.

GET  /commodities/                      — list all tracked commodities
GET  /commodities/{symbol}              — full detail for one commodity
GET  /commodities/{symbol}/signals      — recent signals for one commodity
GET  /commodities/heatmap               — aggregated signal heatmap
GET  /commodities/tier/{n}              — list a single tier (1|2|3)
POST /commodities/scan                  — ad-hoc text scan (debug / preview)
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import database as db
from database import get_db
import models
from services import commodity_tracker as ct

router = APIRouter(prefix="/commodities", tags=["Commodities"])


@router.get("/", response_model=List[models.CommodityOut])
def list_commodities(
    tier: Optional[int] = Query(None, ge=1, le=3),
    db_: Session = Depends(get_db),
):
    q = db_.query(db.Commodity)
    if tier is not None:
        q = q.filter(db.Commodity.tier == tier)
    return q.order_by(db.Commodity.tier, db.Commodity.symbol).all()


@router.get("/heatmap", response_model=List[models.CommodityHeatmapRow])
def heatmap(hours: int = 72, db_: Session = Depends(get_db)):
    """
    Aggregated signal activity over the last `hours` window per commodity.
    Useful as a dashboard widget — net_signal > 0 = bullish for the commodity
    (typically bearish for downstream consumers).
    """
    return ct.commodity_heatmap(db_, hours=hours)


@router.get("/tier/{n}", response_model=List[models.CommodityOut])
def list_by_tier(n: int, db_: Session = Depends(get_db)):
    if n not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="tier must be 1, 2 or 3")
    return (db_.query(db.Commodity)
                .filter(db.Commodity.tier == n)
                .order_by(db.Commodity.symbol)
                .all())


@router.get("/{symbol}", response_model=models.CommodityOut)
def get_commodity(symbol: str, db_: Session = Depends(get_db)):
    row = db_.query(db.Commodity).filter(db.Commodity.symbol == symbol.upper()).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"commodity '{symbol}' not tracked")
    return row


@router.get("/{symbol}/signals", response_model=List[models.CommoditySignalOut])
def get_commodity_signals(
    symbol: str,
    hours: int = 168,    # last 7 days by default
    limit: int = 100,
    db_: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    rows = (db_.query(db.CommoditySignal)
                .filter(db.CommoditySignal.commodity_symbol == symbol.upper(),
                        db.CommoditySignal.timestamp >= cutoff)
                .order_by(db.CommoditySignal.timestamp.desc())
                .limit(limit)
                .all())
    return rows


@router.get("/{symbol}/exposed-tickers")
def exposed_tickers(symbol: str, db_: Session = Depends(get_db)):
    """
    Return the tickers downstream of this commodity along with their current
    StockScore — lets the UI render a "if X commodity moves, watch these
    names" panel.
    """
    row = db_.query(db.Commodity).filter(db.Commodity.symbol == symbol.upper()).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"commodity '{symbol}' not tracked")
    tickers = [t.strip() for t in (row.exposed_tickers or "").split(",") if t.strip()]
    if not tickers:
        return {"symbol": row.symbol, "exposed": []}
    scores = {s.ticker: s for s in
              db_.query(db.StockScore).filter(db.StockScore.ticker.in_(tickers)).all()}
    out = []
    for t in tickers:
        s = scores.get(t)
        out.append({
            "ticker": t,
            "opportunity_score":     getattr(s, "opportunity_score", None),
            "risk_score":            getattr(s, "risk_score", None),
            "indirect_impact_score": getattr(s, "indirect_impact_score", None),
            "decision_bucket":       getattr(s, "decision_bucket", None),
        })
    return {"symbol": row.symbol, "exposed": out}


@router.post("/scan")
def scan_text(payload: dict):
    """
    Ad-hoc scan — POST { headline, content } and get back the commodity hits
    without persisting anything.  Useful for testing the registry.
    """
    headline = payload.get("headline", "")
    content  = payload.get("content", "")
    return {"hits": ct.scan_text(headline, content)}
