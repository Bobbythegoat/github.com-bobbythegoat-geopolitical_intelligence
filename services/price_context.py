"""
services/price_context.py
--------------------------
Fetches historical price context for a ticker using yfinance.
These features give the ML model genuine market history —
understanding whether a stock is overbought, in a downtrend,
or near multi-year highs changes whether an alert is likely to succeed.

Features returned
-----------------
rsi_14              : 14-day RSI (0–100). >70 = overbought, <30 = oversold.
pct_from_52w_high   : (price - 52w_high) / 52w_high. Negative = below high.
momentum_30d        : 30-day price return. Positive = uptrend.
momentum_90d        : 90-day price return. Longer-term trend direction.
rel_strength_90d    : Stock 90d return minus SPY 90d return (relative strength).
vol_regime          : 30d realised volatility / 90d realised volatility.
                      >1.2 = elevated vol regime; <0.8 = calm regime.

Caching
-------
Results are cached in _PRICE_CACHE for CACHE_TTL_SECONDS to avoid
hammering yfinance on every alert creation. The cache is per-process
(not persisted across restarts).
"""

import math
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Cache TTL: 4 hours (price context doesn't need real-time precision)
CACHE_TTL_SECONDS = 4 * 3600

_PRICE_CACHE: Dict[str, tuple] = {}   # ticker → (timestamp, features_dict)

NEUTRAL_CONTEXT = {
    "rsi_14":            50.0,
    "pct_from_52w_high": -0.10,
    "momentum_30d":       0.0,
    "momentum_90d":       0.0,
    "rel_strength_90d":   0.0,
    "vol_regime":         1.0,
}


def _compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.001
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var) or 0.001


def fetch_price_context(ticker: str) -> dict:
    """
    Return a dict of historical price features for the given ticker.
    Uses a 4-hour in-memory cache to avoid redundant yfinance fetches.
    Returns NEUTRAL_CONTEXT if yfinance is unavailable or fetch fails.
    """
    # Cache hit
    now = time.time()
    if ticker in _PRICE_CACHE:
        ts, features = _PRICE_CACHE[ticker]
        if now - ts < CACHE_TTL_SECONDS:
            return features

    try:
        import yfinance as yf
    except ImportError:
        return NEUTRAL_CONTEXT.copy()

    try:
        # Fetch 1 year of daily data — enough for 52w high, 90d momentum, RSI
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 20:
            return NEUTRAL_CONTEXT.copy()

        closes = hist["Close"].tolist()
        current = closes[-1]

        # RSI (14-day)
        rsi = _compute_rsi(closes, 14)

        # 52-week high position
        high_52w = max(closes)
        pct_from_high = round((current - high_52w) / high_52w, 4)

        # 30-day and 90-day momentum
        mom_30d = round((current - closes[-min(30, len(closes))]) / closes[-min(30, len(closes))], 4)
        mom_90d = round((current - closes[-min(90, len(closes))]) / closes[-min(90, len(closes))], 4)

        # Relative strength vs SPY (90d)
        rel_strength = 0.0
        try:
            spy_hist = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=True)
            if not spy_hist.empty and len(spy_hist) >= 90:
                spy_closes = spy_hist["Close"].tolist()
                spy_90d = (spy_closes[-1] - spy_closes[-min(90, len(spy_closes))]) / spy_closes[-min(90, len(spy_closes))]
                rel_strength = round(mom_90d - spy_90d, 4)
        except Exception:
            pass

        # Volatility regime: recent 30-day vol / trailing 90-day vol
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        vol_30  = _std(returns[-30:]) if len(returns) >= 30 else _std(returns)
        vol_90  = _std(returns[-90:]) if len(returns) >= 90 else _std(returns)
        vol_regime = round(vol_30 / vol_90, 3) if vol_90 > 0 else 1.0

        features = {
            "rsi_14":            rsi,
            "pct_from_52w_high": pct_from_high,
            "momentum_30d":      mom_30d,
            "momentum_90d":      mom_90d,
            "rel_strength_90d":  rel_strength,
            "vol_regime":        vol_regime,
        }
        _PRICE_CACHE[ticker] = (now, features)
        return features

    except Exception as e:
        logger.debug("Price context fetch failed for %s: %s", ticker, e)
        return NEUTRAL_CONTEXT.copy()
