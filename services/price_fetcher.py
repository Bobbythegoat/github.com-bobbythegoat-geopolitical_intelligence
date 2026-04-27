"""
services/price_fetcher.py
--------------------------
Fetches real-time and historical price data to populate:
  - pre-event price drift (for expectation_gap.py)
  - ATM options implied move (for expectation_gap.py)
  - post-event price response (for narrative_inflection.py)
  - VIX level (for regime detection in learning.py)

All functions degrade gracefully to 0.0 / None on failure.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_pre_event_price_drift(ticker: str, event_time: datetime, days: int = 5) -> float:
    """Return % price drift in `days` before event_time. 0.0 on failure."""
    try:
        import yfinance as yf
        start = (event_time - timedelta(days=days + 5)).strftime("%Y-%m-%d")
        end = event_time.strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) < 2:
            return 0.0
        first, last = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        return (last - first) / first if first != 0 else 0.0
    except Exception as e:
        logger.debug("price drift fetch failed %s: %s", ticker, e)
        return 0.0


def get_implied_move(ticker: str) -> float:
    """
    Estimate ATM straddle implied move from nearest-expiry options chain.
    Returns fraction (e.g. 0.05 = 5% implied move). 0.0 on failure.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return 0.0
        chain = t.option_chain(expirations[0])
        hist = t.history(period="1d")
        if hist.empty:
            return 0.0
        spot = float(hist["Close"].iloc[-1])
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty or spot == 0:
            return 0.0
        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[:1]]
        atm_put  = puts.iloc[(puts["strike"]  - spot).abs().argsort().iloc[:1]]
        straddle = float(atm_call["lastPrice"].values[0]) + float(atm_put["lastPrice"].values[0])
        return straddle / spot
    except Exception as e:
        logger.debug("implied move fetch failed %s: %s", ticker, e)
        return 0.0


def get_post_event_return(ticker: str, event_time: datetime, days: int = 1) -> float:
    """Return actual price return `days` after event_time. 0.0 on failure."""
    try:
        import yfinance as yf
        start = event_time.strftime("%Y-%m-%d")
        end   = (event_time + timedelta(days=days + 3)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) < 2:
            return 0.0
        first, last = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        return (last - first) / first if first != 0 else 0.0
    except Exception as e:
        logger.debug("post-event return fetch failed %s: %s", ticker, e)
        return 0.0


def get_vix() -> Optional[float]:
    """Fetch latest VIX close. Returns None on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^VIX").history(period="5d")
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)
        return None


def fetch_and_update_event_prices(event_id: int, session) -> Dict[str, dict]:
    """
    For the top-5 impacted tickers of an event, fetch pre-event drift and
    implied move. Stores results and re-runs expectation gap computation.
    Returns dict: ticker -> {drift, implied_move}.
    """
    import database as db
    event = session.get(db.Event, event_id)
    if not event:
        return {}

    impacts = (
        session.query(db.EventTickerImpact)
        .filter_by(event_id=event_id)
        .order_by(db.EventTickerImpact.impact_score.desc())
        .limit(5)
        .all()
    )
    if not impacts:
        return {}

    results: Dict[str, dict] = {}
    for imp in impacts:
        ticker = imp.ticker
        drift   = get_pre_event_price_drift(ticker, event.timestamp)
        implied = get_implied_move(ticker)
        results[ticker] = {"drift": drift, "implied_move": implied}

    # Re-run expectation gap with real price data
    try:
        from services.expectation_gap import compute_event_expectation_gaps
        compute_event_expectation_gaps(event_id, session, price_data=results)
    except Exception as e:
        logger.debug("expectation gap recompute failed: %s", e)

    return results
