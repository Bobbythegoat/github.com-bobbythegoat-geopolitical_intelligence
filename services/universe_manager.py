"""
services/universe_manager.py
-----------------------------
Manages the three-layer ticker universe:
  Layer 1 — Curated seed (defined in seed_data.py, always present)
  Layer 2 — ETF sweep: pulls SOXX/SMH/ITA/XLE holdings weekly
  Layer 3 — News discovery: auto-promotes DiscoveredTicker rows
              when article_count >= 3 AND avg credibility >= 0.50

Called from:
  main.py lifespan — sweep_etf_holdings() at startup
  signal_scheduler.py — sweep_etf_holdings() weekly
  services/stock_discovery.py — check_and_promote_discovered() on each discovery
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

import database as db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback ETF holdings (used when yfinance ETF holdings API fails)
# These are the current top-10 holdings of each ETF.
# ---------------------------------------------------------------------------

_FALLBACK_HOLDINGS: Dict[str, List[str]] = {
    "SOXX": ["NVDA", "AVGO", "AMD", "QCOM", "AMAT", "LRCX", "KLAC", "INTC", "MCHP", "ON"],
    "SMH":  ["NVDA", "TSM", "ASML", "AVGO", "QCOM", "AMD", "AMAT", "TXN", "KLAC", "MU"],
    "ITA":  ["RTX", "LMT", "GD", "NOC", "BA", "LHX", "HII", "LDOS", "TDG", "HEI"],
    "XLE":  ["XOM", "CVX", "COP", "EOG", "MPC", "PSX", "SLB", "OXY", "HAL", "DVN"],
}

# Minimum market cap threshold for ETF sweep additions ($2B)
_MIN_MARKET_CAP_ETF = 2_000_000_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_market_cap(symbol: str) -> Optional[float]:
    """
    Fetch market cap for a symbol using yfinance fast_info.
    Returns None if unavailable or on error.
    """
    try:
        import yfinance as yf
        fi = yf.Ticker(symbol).fast_info
        # fast_info exposes market_cap as an attribute
        mc = getattr(fi, "market_cap", None)
        if mc is None:
            # Some versions use marketCap
            mc = getattr(fi, "marketCap", None)
        return float(mc) if mc else None
    except Exception as exc:
        logger.warning("universe_manager: could not fetch market cap for %s: %s", symbol, exc)
        return None


def _upsert_ticker(
    session: Session,
    symbol: str,
    company_name: Optional[str],
    sector: Optional[str],
    source: str,
) -> bool:
    """
    Insert ticker into Ticker table if not already present.
    Returns True if a new row was inserted, False if already existed.
    New columns (source, enrolled_at, market_cap) may not yet exist in DB —
    we use setattr so the ORM handles gracefully; SQLAlchemy will simply ignore
    unknown columns if they are not yet in the schema.
    """
    existing = session.get(db.Ticker, symbol)
    if existing is not None:
        return False

    ticker_row = db.Ticker(
        ticker=symbol,
        company_name=company_name or symbol,
        sector=sector or "Unknown",
    )
    # Gracefully set new columns that may not be migrated yet
    try:
        setattr(ticker_row, "source", source)
    except Exception:
        pass
    try:
        setattr(ticker_row, "enrolled_at", datetime.now(timezone.utc))
    except Exception:
        pass

    session.add(ticker_row)
    try:
        session.commit()
        logger.info("universe_manager: added new ticker %s (source=%s)", symbol, source)
        return True
    except Exception as exc:
        session.rollback()
        logger.warning("universe_manager: failed to insert ticker %s: %s", symbol, exc)
        return False


# ---------------------------------------------------------------------------
# Layer 2: ETF Holdings Sweep
# ---------------------------------------------------------------------------

def sweep_etf_holdings(session: Session) -> int:
    """
    Pull holdings from SOXX, SMH, ITA, XLE via yfinance, filter to >= $2B
    market cap, and upsert any new tickers into the Ticker table.

    Returns the count of newly added tickers.
    """
    try:
        import yfinance as yf  # noqa: F401 — imported here so module loads without yfinance
    except ImportError:
        logger.warning("universe_manager: yfinance not installed; skipping ETF sweep")
        return 0

    etf_symbols = ["SOXX", "SMH", "ITA", "XLE"]
    new_count = 0

    for etf_symbol in etf_symbols:
        logger.info("universe_manager: sweeping ETF %s", etf_symbol)
        holding_symbols: List[str] = []

        # Attempt to fetch live holdings from yfinance
        try:
            import yfinance as yf
            etf = yf.Ticker(etf_symbol)
            try:
                funds_data = etf.funds_data
                holdings_df = funds_data.top_holdings
                # holdings_df has columns including 'Symbol' and 'Market Value' or 'holdingPercent'
                symbols = holdings_df.index.tolist() if holdings_df is not None and not holdings_df.empty else []
            except Exception:
                symbols = _FALLBACK_HOLDINGS.get(etf_symbol, [])
            if not symbols:
                raise ValueError("empty holdings returned")
            holding_symbols = [str(s) for s in symbols]
        except Exception as exc:
            logger.warning(
                "universe_manager: yfinance holdings failed for %s (%s); using fallback",
                etf_symbol, exc,
            )
            holding_symbols = list(_FALLBACK_HOLDINGS.get(etf_symbol, []))

        if not holding_symbols:
            logger.warning("universe_manager: no holdings found for ETF %s", etf_symbol)
            continue

        for symbol in holding_symbols:
            symbol = symbol.strip().upper()
            if not symbol or len(symbol) > 10:
                continue

            # Check if already in the universe (fast path — skip market cap fetch)
            existing = session.get(db.Ticker, symbol)
            if existing is not None:
                continue

            # Validate market cap — use fast_info to stay within time budget
            market_cap = _get_market_cap(symbol)
            if market_cap is None:
                logger.warning(
                    "universe_manager: skipping %s — could not fetch market cap", symbol
                )
                continue
            if market_cap < _MIN_MARKET_CAP_ETF:
                logger.info(
                    "universe_manager: skipping %s — market cap $%.0fM below $2B threshold",
                    symbol, market_cap / 1e6,
                )
                continue

            added = _upsert_ticker(
                session=session,
                symbol=symbol,
                company_name=None,
                sector=None,
                source="etf_sweep",
            )
            if added:
                new_count += 1

    logger.info(
        "universe_manager: ETF sweep complete — %d new tickers added", new_count
    )
    return new_count


# ---------------------------------------------------------------------------
# Layer 3: Auto-promotion check
# ---------------------------------------------------------------------------

def check_and_promote_discovered(ticker: str, session: Session) -> bool:
    """
    Check if a DiscoveredTicker meets promotion criteria and, if so,
    promote it into the main Ticker table.

    Criteria:
      - article_count >= 3
      - average credibility >= 0.50 (proxied by confidence_score when
        per-article credibility is not tracked separately)

    Returns True if the ticker was promoted in this call, False otherwise.
    """
    discovered: Optional[db.DiscoveredTicker] = (
        session.query(db.DiscoveredTicker)
        .filter(db.DiscoveredTicker.ticker == ticker)
        .first()
    )

    if discovered is None:
        logger.warning(
            "universe_manager: check_and_promote_discovered called for unknown ticker %s",
            ticker,
        )
        return False

    # Already promoted — nothing to do
    if discovered.is_promoted:
        return False

    article_count = discovered.article_count or 0
    confidence = discovered.confidence_score or 0.0

    if article_count < 3:
        return False
    if confidence < 0.50:
        return False

    # Criteria met — promote to main Ticker table
    added = _upsert_ticker(
        session=session,
        symbol=discovered.ticker,
        company_name=discovered.company_name,
        sector=discovered.sector,
        source="news_discovery",
    )

    # Mark as promoted regardless of whether _upsert_ticker inserted a new row
    # (it might already exist from another source — still mark promoted)
    try:
        discovered.is_promoted = 1
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning(
            "universe_manager: failed to mark %s as promoted: %s", ticker, exc
        )
        return False

    if added:
        logger.info(
            "universe_manager: promoted discovered ticker %s to main universe "
            "(article_count=%d, confidence=%.2f)",
            ticker, article_count, confidence,
        )
    else:
        logger.info(
            "universe_manager: %s already in main universe; marked DiscoveredTicker as promoted",
            ticker,
        )

    return True


# ---------------------------------------------------------------------------
# Universe status
# ---------------------------------------------------------------------------

def get_universe_status(session: Session) -> Dict[str, Any]:
    """
    Return a summary dict describing the current ticker universe.

    Uses getattr() with defaults for new columns (source, enrolled_at,
    market_cap) that may not yet be present in the DB schema.
    """
    tickers = session.query(db.Ticker).all()

    by_source: Dict[str, int] = {
        "seed": 0,
        "etf_sweep": 0,
        "news_discovery": 0,
        "etf_delisted": 0,
        "unknown": 0,
    }
    by_sector: Dict[str, int] = {}
    ticker_list = []

    for t in tickers:
        source = getattr(t, "source", None) or "seed"
        # Normalise to known buckets
        if source not in by_source:
            source = "unknown"
        by_source[source] = by_source.get(source, 0) + 1

        sector = t.sector or "Unknown"
        by_sector[sector] = by_sector.get(sector, 0) + 1

        enrolled_at = getattr(t, "enrolled_at", None)
        market_cap = getattr(t, "market_cap", None)

        ticker_list.append(
            {
                "ticker": t.ticker,
                "company_name": t.company_name or "",
                "sector": sector,
                "source": source,
                "enrolled_at": enrolled_at.isoformat() if enrolled_at else None,
                "market_cap": float(market_cap) if market_cap is not None else None,
            }
        )

    return {
        "total": len(tickers),
        "by_source": by_source,
        "by_sector": by_sector,
        "tickers": ticker_list,
    }


# ---------------------------------------------------------------------------
# Layer 4: SEC EDGAR full-company universe bootstrap
# ---------------------------------------------------------------------------

def load_sec_universe(
    session: Session,
    min_market_cap: float = 2_000_000_000,
    limit: int = 500,
) -> int:
    """
    Bootstrap the ticker universe from SEC EDGAR's full company list.

    Downloads company_tickers.json (all SEC-registered companies with tickers)
    and validates each against yfinance for market cap >= min_market_cap.

    This is the path to scaling from 65 tickers to 6000+.
    Runs conservatively: limit controls how many are processed per call.
    Returns count of newly added tickers.

    NOTE: This is designed to be called incrementally. Call multiple times
    (increasing offset via an offset param or by running daily) to build up
    the full universe over time without hammering the APIs.
    """
    try:
        import yfinance as yf  # noqa: F401 — validate import before proceeding
    except ImportError:
        logger.warning("universe_manager: yfinance not installed; skipping SEC universe load")
        return 0

    try:
        from services.sec_filing_analyzer import load_sec_company_registry
    except ImportError:
        logger.warning("universe_manager: sec_filing_analyzer not available; skipping SEC universe load")
        return 0

    # Step 1: Load (or use cached) CIK → ticker registry
    registry = load_sec_company_registry(max_companies=max(limit * 4, 6000))
    if not registry:
        logger.warning("universe_manager: SEC company registry is empty; nothing to load")
        return 0

    # registry maps cik → ticker; we need unique tickers with a company name lookup
    # Rebuild a deduplicated ticker → (cik, company_name) mapping from the registry
    # The registry may contain duplicate tickers (one entry per CIK representation),
    # so collect unique ticker symbols.
    seen_tickers: set = set()
    ticker_list_from_registry: List[str] = []
    for _cik, tkr in registry.items():
        if tkr and tkr not in seen_tickers:
            seen_tickers.add(tkr)
            ticker_list_from_registry.append(tkr)
        if len(ticker_list_from_registry) >= limit * 2:
            break

    new_count = 0
    processed = 0

    for symbol in ticker_list_from_registry:
        if processed >= limit:
            break

        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 10:
            continue

        # Fast path: skip if already in universe
        existing = session.get(db.Ticker, symbol)
        if existing is not None:
            continue

        # Validate with yfinance
        try:
            import yfinance as yf
            fi = yf.Ticker(symbol).fast_info
            market_cap = getattr(fi, "market_cap", None) or getattr(fi, "marketCap", None)
            market_cap = float(market_cap) if market_cap else None
        except Exception as exc:
            logger.debug("universe_manager: yfinance error for %s: %s", symbol, exc)
            time.sleep(0.05)
            processed += 1
            continue

        time.sleep(0.05)
        processed += 1

        if not market_cap or market_cap < min_market_cap:
            logger.debug(
                "universe_manager: skipping %s — market cap $%.0fM below threshold",
                symbol, (market_cap or 0) / 1e6,
            )
            continue

        added = _upsert_ticker(
            session=session,
            symbol=symbol,
            company_name=None,
            sector=None,
            source="sec_edgar",
        )
        if added:
            new_count += 1

    logger.info(
        "universe_manager: SEC universe load complete — %d new tickers added (processed=%d)",
        new_count, processed,
    )
    return new_count
