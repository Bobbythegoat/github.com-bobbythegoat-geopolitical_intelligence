"""
routers/stocks.py
-----------------
GET /stocks/                       — list all stock scorecards
GET /stocks/{ticker}               — scorecard for a specific ticker
GET /stocks/{ticker}/events        — events currently impacting this ticker
GET /stocks/{ticker}/fundamentals  — live fundamentals + technicals via yfinance
"""

import math
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import database as db
from database import get_db
import models

_YF_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

try:
    import yfinance as yf
    import requests as _req
    import requests.utils as _req_utils

    # Patch requests' default User-Agent globally so ALL yfinance calls
    # (regardless of which internal endpoint they hit) look like Chrome.
    # This is more reliable than patching yfinance internals, which change
    # between minor versions.
    _req_utils.default_user_agent = lambda: _YF_UA

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

router = APIRouter(prefix="/stocks", tags=["Stocks"])


# ---------------------------------------------------------------------------
# RSI helper (pure Python, no extra deps)
# ---------------------------------------------------------------------------

def _compute_rsi(closes: list, period: int = 14) -> Optional[float]:
    """Compute Wilder's RSI for the last `period` days."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    if not gains and not losses:
        return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs  = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _safe(val):
    """Return None if value is NaN/None, else the value rounded to 4dp."""
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _de_ratio(raw) -> Optional[float]:
    """
    Normalize yfinance's debtToEquity to the standard D/E ratio convention.

    yfinance returns debtToEquity already multiplied by 100, so:
        170  → 1.70x  (healthy tech company)
        1100 → 11.0x  (Boeing-level extreme; likely negative equity)

    Returns None when equity is negative or near-zero (D/E > 50 or < -50),
    because the ratio is mathematically undefined in those cases and would
    only mislead the reader.
    """
    v = _safe(raw)
    if v is None:
        return None
    ratio = round(v / 100, 2)
    if abs(ratio) > 50:   # negative or near-zero equity — not a meaningful number
        return None
    return ratio


@router.get("/", response_model=List[models.StockScoreOut])
def list_stock_scores(
    sector: str = Query(None, description="Filter by sector"),
    min_opportunity: float = Query(0.0, ge=0, le=1),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_db),
):
    """List stock scorecards ranked by opportunity score."""
    q = (
        session.query(db.StockScore)
        .filter(db.StockScore.opportunity_score >= min_opportunity)
        .order_by(db.StockScore.opportunity_score.desc())
    )
    scores = q.limit(limit).all()
    result = []
    for s in scores:
        ticker_row = session.get(db.Ticker, s.ticker)
        if sector and ticker_row and ticker_row.sector != sector:
            continue
        result.append(models.StockScoreOut(
            ticker=s.ticker,
            company_name=ticker_row.company_name if ticker_row else None,
            sector=ticker_row.sector if ticker_row else None,
            opportunity_score=s.opportunity_score,
            crowding_score=s.crowding_score,
            risk_score=s.risk_score,
            exposure_score=getattr(s, "exposure_score", 0.0) or 0.0,
            impact_score=getattr(s, "impact_score", 0.0) or 0.0,
            narrative_score=getattr(s, "narrative_score", 0.0) or 0.0,
            lag_score=getattr(s, "lag_score", 0.0) or 0.0,
            asymmetry_score=getattr(s, "asymmetry_score", 0.0) or 0.0,
            decision_bucket=getattr(s, "decision_bucket", "Watch") or "Watch",
            updated_at=s.updated_at,
        ))
    return result


@router.get("/emerging", summary="Real-time stock recommendations from live news")
def get_emerging_stocks(
    hours: int = Query(24, ge=1, le=168, description="Hours of recent articles to scan"),
    min_confidence: float = Query(0.45, ge=0.0, le=1.0, description="Min discovery confidence"),
    limit: int = Query(20, ge=1, le=50, description="Max results"),
    force_refresh: bool = Query(False, description="Bypass 5-min cache and re-scan"),
    exclude_extended: bool = Query(False, description="Drop already-exploded (overextended) names"),
    session: Session = Depends(get_db),
):
    """
    Dynamically discovers stocks mentioned in recent news articles that are
    outside the hardcoded 30-ticker universe.

    Pipeline:
      1. Fetch articles from the last `hours` hours
      2. Extract company / ticker mentions via regex NER
      3. Validate each candidate with yfinance (real price + market-cap gate)
      4. Score by credibility × mention frequency × confidence
      5. Return ranked list, cached for 5 minutes

    Results refresh automatically as new articles are ingested.
    Force a re-scan with ?force_refresh=true.
    """
    from services.stock_recommender import get_emerging_recommendations
    recs = get_emerging_recommendations(
        session,
        hours=hours,
        min_confidence=min_confidence,
        limit=limit,
        force_refresh=force_refresh,
        exclude_extended=exclude_extended,
    )
    return {"items": recs, "count": len(recs), "hours_scanned": hours}


@router.get("/{ticker}/events", response_model=List[models.EventSummary])
def get_ticker_events(
    ticker: str,
    limit: int = Query(20, le=100),
    session: Session = Depends(get_db),
):
    """Return events that are currently impacting this ticker."""
    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(ticker=ticker.upper())
        .order_by(db.EventTickerImpact.impact_score.desc())
        .all()
    )
    if not impacts:
        raise HTTPException(status_code=404, detail=f"No events found for {ticker.upper()}")

    event_ids = [i.event_id for i in impacts][:limit]
    events = (
        session.query(db.Event)
        .filter(db.Event.event_id.in_(event_ids))
        .order_by(db.Event.credibility_score.desc())
        .all()
    )
    return events


@router.get("/{ticker}/fundamentals", response_model=models.StockFundamentalsOut)
def get_stock_fundamentals(ticker: str, session: Session = Depends(get_db)):
    """
    Live fundamentals, technicals, historical financials, and analyst consensus
    from Yahoo Finance, combined with our geopolitical scores and company context.

    Strategy: fast_info + history() are always reliable. .info is used as a
    supplemental fallback only — it is rate-limited by Yahoo Finance and
    should never be the single source of truth.
    """
    sym = ticker.upper()

    if not YFINANCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="yfinance not installed.")

    yf_ticker = yf.Ticker(sym)

    # ── Step 1: fast_info — always works, gives price/market data ────────
    fast     = yf_ticker.fast_info
    price    = _safe(getattr(fast, "last_price", None))
    high52   = _safe(getattr(fast, "year_high",  None))
    low52    = _safe(getattr(fast, "year_low",   None))
    mktcap   = _safe(getattr(fast, "market_cap", None))

    if price is None:
        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance returned no price for {sym}. "
                   "This can be a temporary rate-limit — wait 30s and retry.",
        )

    # ── Step 2: price history — always reliable ───────────────────────────
    try:
        hist    = yf_ticker.history(period="1y", interval="1d")
        closes  = hist["Close"].tolist()  if not hist.empty else []
        volumes = hist["Volume"].tolist() if not hist.empty else []
    except Exception:
        closes, volumes = [], []

    # ── Step 3: .info — supplemental only, graceful fallback ─────────────
    info: dict = {}
    try:
        _raw = yf_ticker.info or {}
        # Only accept if it looks like a real response (has at least longName or symbol)
        if _raw.get("longName") or _raw.get("symbol") or _raw.get("shortName"):
            info = _raw
    except Exception:
        pass

    # ── Step 4: quarterly financials for historical context ───────────────
    revenue_history:    list = []
    earnings_history:   list = []
    gross_margin_history: list = []
    company_description: Optional[str] = info.get("longBusinessSummary") or None

    try:
        qfin = yf_ticker.quarterly_income_stmt  # columns = quarters, rows = line items
        if qfin is not None and not qfin.empty:
            # Normalize index labels (may differ by yfinance version)
            idx = {str(i).lower().replace(" ", ""): i for i in qfin.index}
            rev_key  = next((v for k, v in idx.items() if "totalrevenue" in k or "revenue" in k), None)
            gp_key   = next((v for k, v in idx.items() if "grossprofit" in k), None)
            ni_key   = next((v for k, v in idx.items() if "netincome" in k and "minority" not in k), None)

            cols = list(qfin.columns)[:4]  # last 4 quarters newest first

            for i, col in enumerate(cols):
                period = str(col)[:10]
                rev  = float(qfin.loc[rev_key, col]) if rev_key and rev_key in qfin.index else None
                gp   = float(qfin.loc[gp_key,  col]) if gp_key  and gp_key  in qfin.index else None
                ni   = float(qfin.loc[ni_key,  col]) if ni_key  and ni_key  in qfin.index else None

                # QoQ revenue growth
                rev_growth = None
                if i + 1 < len(cols) and rev is not None:
                    prev_rev = None
                    try:
                        prev_rev = float(qfin.loc[rev_key, cols[i + 1]]) if rev_key else None
                    except Exception:
                        pass
                    if prev_rev and prev_rev != 0:
                        rev_growth = round((rev - prev_rev) / abs(prev_rev) * 100, 1)

                gm = round(gp / rev * 100, 1) if rev and gp and rev != 0 else None

                revenue_history.append({
                    "period": period,
                    "revenue": round(rev / 1e9, 2) if rev else None,  # billions
                    "revenue_growth_qoq_pct": rev_growth,
                    "gross_margin_pct": gm,
                    "net_income": round(ni / 1e9, 2) if ni else None,
                })
                if gm is not None:
                    gross_margin_history.append({"period": period, "gross_margin_pct": gm})
    except Exception:
        pass

    try:
        qe = yf_ticker.quarterly_earnings  # DataFrame with Earnings and EPS columns
        if qe is not None and not qe.empty:
            for period, row in qe.iloc[:4].iterrows():
                earnings_history.append({
                    "period": str(period)[:10],
                    "eps_actual":   round(float(row.get("Earnings", 0) or 0), 2),
                    "eps_estimate": round(float(row.get("Revenue", 0) or 0) / 1e9, 2) if row.get("Revenue") else None,
                })
    except Exception:
        pass

    # ── Price calculations ────────────────────────────────────────────────
    pct_from_high = None
    if price and high52 and high52 > 0:
        pct_from_high = round((price - high52) / high52 * 100, 2)

    def _pct_chg(n_days: int) -> Optional[float]:
        if len(closes) < n_days + 1:
            return None
        past = closes[-(n_days + 1)]
        return round((closes[-1] - past) / past * 100, 2) if past else None

    above_50 = above_200 = vol_ratio = None
    if len(closes) >= 50:
        above_50 = closes[-1] > sum(closes[-50:]) / 50
    if len(closes) >= 200:
        above_200 = closes[-1] > sum(closes[-200:]) / 200
    if len(volumes) >= 2:
        avg_v = sum(volumes[-31:-1]) / max(len(volumes[-31:-1]), 1)
        if avg_v:
            vol_ratio = round(volumes[-1] / avg_v, 2)

    rsi = _compute_rsi(closes)

    # ── Analyst consensus ─────────────────────────────────────────────────
    target  = _safe(info.get("targetMeanPrice"))
    rec_str = info.get("recommendationKey", "").lower() or None
    upside  = round((target - price) / price * 100, 2) if target and price else None

    # ── Geopolitical scores ───────────────────────────────────────────────
    score_row  = session.query(db.StockScore).filter_by(ticker=sym).first()
    ticker_row = session.get(db.Ticker, sym)

    return models.StockFundamentalsOut(
        ticker=sym,
        company_name=info.get("longName") or info.get("shortName") or (ticker_row.company_name if ticker_row else sym),
        sector=info.get("sector") or (ticker_row.sector if ticker_row else None),
        industry=info.get("industry"),
        company_description=company_description,

        current_price=price,
        price_change_1d_pct=_pct_chg(1),
        price_change_5d_pct=_pct_chg(5),
        price_change_1m_pct=_pct_chg(21),
        week_52_high=high52,
        week_52_low=low52,
        week_52_pct_from_high=pct_from_high,

        market_cap=mktcap,
        pe_ratio=_safe(info.get("trailingPE")),
        forward_pe=_safe(info.get("forwardPE")),
        pb_ratio=_safe(info.get("priceToBook")),
        ev_ebitda=_safe(info.get("enterpriseToEbitda")),
        dividend_yield=_safe(info.get("dividendYield")),

        rsi_14=rsi,
        above_sma_50=above_50,
        above_sma_200=above_200,
        volume_ratio=vol_ratio,

        revenue_growth_yoy=_safe(info.get("revenueGrowth")),
        earnings_growth_yoy=_safe(info.get("earningsGrowth")),
        profit_margin=_safe(info.get("profitMargins")),
        # yfinance returns debtToEquity pre-multiplied by 100 (e.g. 170 = 1.70x).
        # Divide by 100 to normalize to standard ratio convention.
        # Cap at 50x and floor at -50x — values outside that range indicate
        # negative or near-zero equity (e.g. Boeing), where D/E is not meaningful.
        debt_to_equity=_de_ratio(info.get("debtToEquity")),
        beta=_safe(info.get("beta") or getattr(fast, "three_month_average_volume", None) and None),

        analyst_target_price=target,
        analyst_upside_pct=upside,
        analyst_recommendation=rec_str,
        analyst_count=info.get("numberOfAnalystOpinions"),

        revenue_history=revenue_history,
        earnings_history=earnings_history,
        gross_margin_history=gross_margin_history,

        opportunity_score=score_row.opportunity_score if score_row else None,
        risk_score=score_row.risk_score               if score_row else None,
        crowding_score=score_row.crowding_score        if score_row else None,

        fetched_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Discovered Tickers (ML stock discovery)
# ---------------------------------------------------------------------------

@router.get("/discovered", response_model=List[models.DiscoveredTickerOut])
def list_discovered_tickers(
    min_confidence: float = Query(0.45, ge=0.0, le=1.0),
    promoted_only: bool = Query(False),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_db),
):
    """
    List stocks discovered dynamically from article text.
    These are tickers beyond the hardcoded universe that were mentioned
    in news articles and validated via yfinance.
    """
    from services.stock_discovery import get_discovered_tickers
    results = get_discovered_tickers(session, min_confidence=min_confidence, limit=limit)
    if promoted_only:
        results = [r for r in results if r.is_promoted]
    return results


# ---------------------------------------------------------------------------
# ML Model (training + inference)
# ---------------------------------------------------------------------------

@router.get("/signals")
def get_stock_signals(
    min_confidence: float = Query(0.0, ge=0, le=1, description="Filter by ML confidence"),
    limit: int = Query(20, le=50),
    session: Session = Depends(get_db),
):
    """
    Core intelligence output: news → stock signals.

    For each tracked stock with an active alert, returns the clearest
    directional signal derived from recent news, ranked by ML confidence.
    This is the primary decision-support output of the system.

    Each signal shows:
      - Which news event is driving the signal
      - ML predicted outcome and direction
      - Confidence level (only signals above 0.52 are meaningful)
      - Decision bucket and key scores
    """
    import json
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=7)

    # Get all recent non-dismissed alerts with ML predictions
    alerts = (
        session.query(db.Alert)
        .filter(
            db.Alert.dismissed == 0,
            db.Alert.timestamp >= cutoff,
            db.Alert.ml_predicted_outcome.isnot(None),
        )
        .order_by(db.Alert.ml_confidence.desc(), db.Alert.timestamp.desc())
        .all()
    )

    seen_tickers = set()
    signals = []

    for alert in alerts:
        if len(signals) >= limit:
            break

        conf = alert.ml_confidence or 0.0
        if conf < min_confidence:
            continue

        event = session.get(db.Event, alert.event_id)
        if not event:
            continue

        # Get top impacted tickers for this alert
        impacts = (
            session.query(db.EventTickerImpact)
            .filter_by(event_id=alert.event_id)
            .order_by(db.EventTickerImpact.impact_score.desc())
            .limit(5)
            .all()
        )

        for imp in impacts:
            if imp.ticker in seen_tickers:
                continue
            score_row = session.query(db.StockScore).filter_by(ticker=imp.ticker).first()
            ticker_row = session.get(db.Ticker, imp.ticker)
            if not score_row:
                continue

            seen_tickers.add(imp.ticker)

            outcome = alert.ml_predicted_outcome or "neutral"
            direction = alert.ml_predicted_direction or "flat"

            # Conviction label
            if conf >= 0.65:
                conviction = "High"
            elif conf >= 0.52:
                conviction = "Moderate"
            else:
                conviction = "Low"

            signals.append({
                "ticker":               imp.ticker,
                "company_name":         ticker_row.company_name if ticker_row else imp.ticker,
                "sector":               ticker_row.sector if ticker_row else None,
                # News driving the signal
                "event_title":          event.title,
                "event_type":           event.event_type,
                "event_credibility":    round(event.credibility_score, 3),
                "narrative_stage":      event.narrative_stage,
                "alert_tier":           alert.tier,
                "alert_timestamp":      alert.timestamp.isoformat(),
                "horizon":              alert.horizon,
                # ML signal
                "ml_predicted_outcome":   outcome,
                "ml_predicted_direction": direction,
                "ml_confidence":          round(conf, 3),
                "conviction":             conviction,
                # Impact direction on this ticker
                "impact_score":          round(imp.impact_score, 3),
                "impact_direction":      "positive" if imp.impact_score > 0 else "negative",
                # Stock scores
                "opportunity_score":    round(score_row.opportunity_score or 0, 3),
                "risk_score":           round(score_row.risk_score or 0, 3),
                "crowding_score":       round(score_row.crowding_score or 0, 3),
                "decision_bucket":      score_row.decision_bucket or "Watch",
            })

    return {"signals": signals, "total": len(signals)}


@router.get("/universe", tags=["Stocks"])
def get_universe(session: Session = Depends(get_db)):
    """
    Returns the full tracked stock universe grouped by source layer and sector.
    Sources: seed | etf_sweep | news_discovery | etf_delisted
    """
    try:
        from services.universe_manager import get_universe_status
        return get_universe_status(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ml/info", response_model=models.MLModelInfo)
def get_ml_info(session: Session = Depends(get_db)):
    """
    Return ML model status, training readiness, and accuracy metrics.
    The model activates automatically once 50 labeled alert outcomes accumulate.
    """
    from services.ml_predictor import get_model_info
    return get_model_info(session)


@router.post("/ml/train", status_code=200)
def trigger_ml_training(session: Session = Depends(get_db)):
    """
    Manually trigger ML model training from labeled outcome records.
    Requires at least 10 labeled outcomes. Safe to call repeatedly.
    If insufficient real outcomes exist, call /ml/bootstrap first.
    """
    from services.ml_predictor import train_model
    return train_model(session)


@router.post("/ml/bootstrap", status_code=200)
def bootstrap_ml_model(session: Session = Depends(get_db)):
    """
    Bootstrap the ML model from existing stock scores when no labeled
    alert outcomes exist yet. Uses opportunity_score as proxy labels.

    Use this to get predictions working on day 1.
    The model will auto-retrain on real outcome data as it accumulates.
    """
    from services.ml_predictor import bootstrap_train
    return bootstrap_train(session)


@router.get("/ml/predict/{alert_id}", response_model=models.MLPredictionOut)
def predict_alert_outcome(alert_id: int, session: Session = Depends(get_db)):
    """
    Run ML prediction for a specific alert.
    Returns outcome probabilities (profitable/neutral/unprofitable) and
    directional prediction (up/flat/down) with confidence score.
    """
    import json
    from services.ml_predictor import predict
    alert = session.get(db.Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    snap = json.loads(alert.feature_vector_snapshot or "{}")
    event = session.get(db.Event, alert.event_id) if alert.event_id else None
    return predict(snap, event)


@router.get("/{ticker}/ml-prediction")
def get_ticker_ml_prediction(ticker: str, session: Session = Depends(get_db)):
    """
    Return the latest ML prediction for a ticker, drawn from its most recent
    non-dismissed alert that has ML inference fields populated.

    Returns the predicted outcome, direction, confidence, and the alert context
    so the stock detail panel can show what the model currently thinks.
    """
    sym = ticker.upper()

    # Find the most recent alert for this ticker that has an ML prediction
    alert = (
        session.query(db.Alert)
        .join(db.AlertOutcome, db.Alert.id == db.AlertOutcome.alert_id)
        .filter(
            db.AlertOutcome.ticker == sym,
            db.Alert.dismissed == 0,
            db.Alert.ml_predicted_outcome.isnot(None),
        )
        .order_by(db.Alert.timestamp.desc())
        .first()
    )

    if alert is None:
        # Fall back: find any alert for this ticker's events
        impact = (
            session.query(db.EventTickerImpact)
            .filter_by(ticker=sym)
            .order_by(db.EventTickerImpact.impact_score.desc())
            .first()
        )
        if impact:
            alert = (
                session.query(db.Alert)
                .filter_by(event_id=impact.event_id, dismissed=0)
                .order_by(db.Alert.timestamp.desc())
                .first()
            )

    if alert is None:
        return {
            "ticker": sym,
            "ml_available": False,
            "ml_predicted_outcome": None,
            "ml_predicted_direction": None,
            "ml_confidence": 0.0,
            "alert_tier": None,
            "alert_timestamp": None,
            "model_note": "No alerts found for this ticker yet.",
        }

    # ── On-the-fly inference when stored prediction is missing ────────────
    # Alerts created before the ML model was trained have null ml_predicted_outcome.
    # Compute the prediction now from the stored feature snapshot, then persist
    # it so subsequent calls return instantly.
    if alert.ml_predicted_outcome is None and alert.feature_vector_snapshot:
        try:
            import json as _json
            from services.ml_predictor import predict as _predict
            _snap  = _json.loads(alert.feature_vector_snapshot)
            _event = session.get(db.Event, alert.event_id) if alert.event_id else None
            _pred  = _predict(_snap, _event)
            # Only persist predictions from the real trained model, not the
            # rule-based fallback (model_available=False), to avoid polluting
            # stored predictions with low-quality inferences.
            if _pred.get("model_available") and _pred.get("predicted_outcome"):
                alert.ml_predicted_outcome   = _pred["predicted_outcome"]
                alert.ml_predicted_direction = _pred.get("predicted_direction")
                alert.ml_confidence          = _pred.get("ml_confidence")
                try:
                    session.commit()
                except Exception:
                    session.rollback()
        except Exception:
            pass  # model not yet trained — fall through to informational response

    return {
        "ticker": sym,
        "ml_available": alert.ml_predicted_outcome is not None,
        "ml_predicted_outcome":   alert.ml_predicted_outcome,
        "ml_predicted_direction": alert.ml_predicted_direction,
        "ml_confidence":          alert.ml_confidence or 0.0,
        "alert_tier":             alert.tier,
        "alert_timestamp":        alert.timestamp.isoformat() if alert.timestamp else None,
        "confidence_score":       alert.confidence_score or 0.0,
        "horizon":                alert.horizon,
        "regime_label":           alert.regime_label,
        "model_note": (
            f"ML model prediction based on {alert.tier} alert from "
            f"{alert.timestamp.strftime('%b %d %H:%M') if alert.timestamp else 'unknown'}."
            if alert.ml_predicted_outcome else
            "No ML model trained yet — train from the Admin panel or run POST /stocks/ml/bootstrap."
        ),
    }


@router.get("/{ticker}", response_model=models.StockScoreOut)
def get_stock_scorecard(ticker: str, session: Session = Depends(get_db)):
    """Get the full scorecard for a single ticker."""
    score = session.query(db.StockScore).filter_by(ticker=ticker.upper()).first()
    if not score:
        raise HTTPException(status_code=404, detail=f"No scorecard for {ticker.upper()}")
    ticker_row = session.get(db.Ticker, ticker.upper())
    return models.StockScoreOut(
        ticker=score.ticker,
        company_name=ticker_row.company_name if ticker_row else None,
        sector=ticker_row.sector if ticker_row else None,
        opportunity_score=score.opportunity_score,
        crowding_score=score.crowding_score,
        risk_score=score.risk_score,
        exposure_score=getattr(score, "exposure_score", 0.0) or 0.0,
        impact_score=getattr(score, "impact_score", 0.0) or 0.0,
        narrative_score=getattr(score, "narrative_score", 0.0) or 0.0,
        lag_score=getattr(score, "lag_score", 0.0) or 0.0,
        asymmetry_score=getattr(score, "asymmetry_score", 0.0) or 0.0,
        decision_bucket=getattr(score, "decision_bucket", "Watch") or "Watch",
        updated_at=score.updated_at,
    )
