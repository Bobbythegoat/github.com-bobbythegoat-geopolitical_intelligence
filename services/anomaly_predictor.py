"""
services/anomaly_predictor.py
------------------------------
Pre-breakout anomaly prediction engine.

Detects BEFORE a price anomaly happens by combining:
  1. Bollinger Band Squeeze     — low volatility before explosive move
  2. Volume Accumulation        — unusual volume building without price move
  3. RSI Divergence             — price making new high/low but RSI not confirming
  4. Event-Lag Signal           — credible high-impact event + no price reaction yet

Outputs a prediction score [0, 1] and direction estimate for each ticker.
The prediction is surfaced as a "pre_anomaly" event in the pipeline so
users see it BEFORE the move, not after.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level prediction cache: ticker -> (result_dict, expiry_timestamp)
# TTL = 2 hours to avoid spamming yfinance
# ---------------------------------------------------------------------------
_PREDICTION_CACHE: Dict[str, Tuple[dict, float]] = {}
_CACHE_TTL_SECONDS = 7200  # 2 hours


def _cache_get(ticker: str) -> Optional[dict]:
    """Return cached prediction if still valid, else None."""
    entry = _PREDICTION_CACHE.get(ticker)
    if entry is None:
        return None
    result, expiry = entry
    if time.monotonic() > expiry:
        del _PREDICTION_CACHE[ticker]
        return None
    return result


def _cache_set(ticker: str, result: dict) -> None:
    """Store prediction result in cache with TTL."""
    _PREDICTION_CACHE[ticker] = (result, time.monotonic() + _CACHE_TTL_SECONDS)


# ---------------------------------------------------------------------------
# Signal 1: Bollinger Band Squeeze
# ---------------------------------------------------------------------------

def _signal_bollinger_squeeze(closes: list) -> dict:
    """
    Detect low-volatility squeeze before an explosive breakout.

    Logic:
      - 20-day SMA and 20-day std dev from daily closes
      - Squeeze ratio = current band width / average band width over last 90 days
      - Squeeze = squeeze_ratio < 0.5  (bands are abnormally narrow)
      - Score = (1.0 - squeeze_ratio) * 0.8 when squeezed, else 0
      - Direction = +1 if price > SMA (bullish setup), -1 if below

    Returns dict: {score, direction, detail}
    """
    default = {"score": 0.0, "direction": 0.0, "detail": "insufficient data"}

    if len(closes) < 90:
        return default

    import statistics

    try:
        # Current 20-day SMA and standard deviation
        window_20 = closes[-20:]
        sma_20 = statistics.mean(window_20)
        std_20 = statistics.pstdev(window_20)
        current_band_width = 4 * std_20  # upper - lower = 2*2*std

        if current_band_width <= 0:
            return {**default, "detail": "zero band width (flat price)"}

        # Historical average band width over last 90 days using rolling 20-day windows
        band_widths = []
        for i in range(20, len(closes) + 1):
            w = closes[i - 20:i]
            bw = 4 * statistics.pstdev(w)
            band_widths.append(bw)

        if not band_widths:
            return default

        avg_band_width = statistics.mean(band_widths)

        if avg_band_width <= 0:
            return {**default, "detail": "zero historical band width"}

        squeeze_ratio = current_band_width / avg_band_width

        # Direction: is price above or below the SMA?
        current_price = closes[-1]
        direction = 1.0 if current_price > sma_20 else -1.0

        if squeeze_ratio < 0.5:
            score = (1.0 - squeeze_ratio) * 0.8
            score = min(score, 1.0)
            detail = (
                f"Squeeze active: band width {current_band_width:.3f} is "
                f"{squeeze_ratio:.2f}x historical avg ({avg_band_width:.3f}). "
                f"Price {'above' if direction > 0 else 'below'} SMA-20 "
                f"({'bullish' if direction > 0 else 'bearish'} setup)."
            )
        else:
            score = 0.0
            detail = (
                f"No squeeze: band width ratio {squeeze_ratio:.2f} (threshold 0.5). "
                f"Price {'above' if direction > 0 else 'below'} SMA-20."
            )

        return {"score": score, "direction": direction, "detail": detail}

    except Exception as e:
        logger.debug("BB squeeze calculation error: %s", e)
        return {**default, "detail": f"calculation error: {e}"}


# ---------------------------------------------------------------------------
# Signal 2: Volume Accumulation Divergence
# ---------------------------------------------------------------------------

def _signal_volume_accumulation(closes: list, volumes: list, highs: list, lows: list) -> dict:
    """
    Detect unusual volume building without a corresponding price move.

    Logic:
      - Compare 5-day avg volume to 20-day avg volume
      - Accumulation = vol_5d_avg / vol_20d_avg > 1.5 (50% more volume)
      - Price divergence = price change over last 5 days < 1.5%
      - Score = (vol_ratio - 1.0) * 0.6 when accumulating + no price move
      - Direction = detect buying (close near high) vs selling (close near low)

    Returns dict: {score, direction, detail}
    """
    default = {"score": 0.0, "direction": 0.0, "detail": "insufficient data"}

    if len(closes) < 20 or len(volumes) < 20 or len(highs) < 5 or len(lows) < 5:
        return default

    try:
        vol_5d_avg = sum(volumes[-5:]) / 5.0
        vol_20d_avg = sum(volumes[-20:]) / 20.0

        if vol_20d_avg <= 0:
            return {**default, "detail": "zero avg volume"}

        vol_ratio = vol_5d_avg / vol_20d_avg

        # Price change over last 5 days
        price_5d_change = abs(closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 1.0

        if vol_ratio > 1.5 and price_5d_change < 0.015:
            score = (vol_ratio - 1.0) * 0.6
            score = min(score, 1.0)

            # Estimate buying vs selling pressure using close position within day's range
            # Average (close - low) / (high - low) over last 5 days
            # > 0.5 = closes near high = buying pressure
            # < 0.5 = closes near low = selling pressure
            close_position_sum = 0.0
            valid_days = 0
            for i in range(-5, 0):
                day_range = highs[i] - lows[i]
                if day_range > 0:
                    pos = (closes[i] - lows[i]) / day_range
                    close_position_sum += pos
                    valid_days += 1

            if valid_days > 0:
                avg_close_pos = close_position_sum / valid_days
                direction = 1.0 if avg_close_pos >= 0.5 else -1.0
                pressure_word = "buying" if direction > 0 else "selling"
            else:
                direction = 0.0
                pressure_word = "unclear"

            detail = (
                f"Volume accumulation: 5d avg {vol_ratio:.2f}x vs 20d avg, "
                f"price 5d change only {price_5d_change*100:.2f}%. "
                f"Pressure type: {pressure_word}."
            )
        else:
            score = 0.0
            if vol_ratio <= 1.5:
                detail = f"No volume surge: 5d/20d ratio {vol_ratio:.2f} (need >1.5)."
            else:
                detail = (
                    f"Volume up ({vol_ratio:.2f}x) but price already moved "
                    f"{price_5d_change*100:.2f}% (need <1.5%)."
                )
            direction = 0.0

        return {"score": score, "direction": direction, "detail": detail}

    except Exception as e:
        logger.debug("Volume accumulation calculation error: %s", e)
        return {**default, "detail": f"calculation error: {e}"}


# ---------------------------------------------------------------------------
# Signal 3: RSI Divergence
# ---------------------------------------------------------------------------

def _compute_rsi(closes: list, period: int = 14) -> list:
    """
    Compute RSI values for a list of closes.
    Returns a list the same length as closes (first `period` values are None).
    """
    if len(closes) <= period:
        return [None] * len(closes)

    rsi_values = [None] * period
    gains = []
    losses = []

    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def _rsi_from_avgs(ag, al):
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    rsi_values.append(_rsi_from_avgs(avg_gain, avg_loss))

    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rsi_values.append(_rsi_from_avgs(avg_gain, avg_loss))

    return rsi_values


def _signal_rsi_divergence(closes: list) -> dict:
    """
    Detect RSI divergence — price and momentum moving in opposite directions.

    Logic:
      - Compute 14-day RSI
      - Look back 10 bars for divergence
      - Bullish: price lower low but RSI higher low  → upcoming upside
      - Bearish: price higher high but RSI lower high → upcoming downside
      - Score = 0.65 if divergence found, else 0

    Returns dict: {score, direction, detail}
    """
    default = {"score": 0.0, "direction": 0.0, "detail": "insufficient data"}

    if len(closes) < 30:
        return default

    try:
        rsi_values = _compute_rsi(closes)
        # Filter to last 10 bars where RSI is valid
        lookback = 10
        recent_closes = closes[-lookback:]
        recent_rsi = rsi_values[-lookback:]

        # Remove pairs where RSI is None
        valid_pairs = [(c, r) for c, r in zip(recent_closes, recent_rsi) if r is not None]

        if len(valid_pairs) < 4:
            return {**default, "detail": "not enough RSI values computed"}

        valid_closes = [p[0] for p in valid_pairs]
        valid_rsi = [p[1] for p in valid_pairs]

        current_close = valid_closes[-1]
        current_rsi = valid_rsi[-1]
        prior_closes = valid_closes[:-1]
        prior_rsi = valid_rsi[:-1]

        # Bullish divergence: price at lower low, RSI at higher low
        min_prior_close = min(prior_closes)
        min_prior_rsi_at_close_low = valid_rsi[valid_closes.index(min_prior_close)]

        bullish_divergence = (
            current_close < min_prior_close          # price making lower low
            and current_rsi > min_prior_rsi_at_close_low  # RSI making higher low
        )

        # Bearish divergence: price at higher high, RSI at lower high
        max_prior_close = max(prior_closes)
        max_prior_rsi_at_close_high = valid_rsi[valid_closes.index(max_prior_close)]

        bearish_divergence = (
            current_close > max_prior_close          # price making higher high
            and current_rsi < max_prior_rsi_at_close_high  # RSI making lower high
        )

        if bullish_divergence:
            return {
                "score": 0.65,
                "direction": 1.0,
                "detail": (
                    f"Bullish RSI divergence: price at lower low ({current_close:.2f} < "
                    f"{min_prior_close:.2f}) but RSI at higher low "
                    f"({current_rsi:.1f} > {min_prior_rsi_at_close_low:.1f})."
                ),
            }
        elif bearish_divergence:
            return {
                "score": 0.65,
                "direction": -1.0,
                "detail": (
                    f"Bearish RSI divergence: price at higher high ({current_close:.2f} > "
                    f"{max_prior_close:.2f}) but RSI at lower high "
                    f"({current_rsi:.1f} < {max_prior_rsi_at_close_high:.1f})."
                ),
            }
        else:
            return {
                "score": 0.0,
                "direction": 0.0,
                "detail": (
                    f"No RSI divergence detected over last {len(valid_pairs)} bars "
                    f"(current RSI {current_rsi:.1f}, price {current_close:.2f})."
                ),
            }

    except Exception as e:
        logger.debug("RSI divergence calculation error: %s", e)
        return {**default, "detail": f"calculation error: {e}"}


# ---------------------------------------------------------------------------
# Signal 4: Event-Lag Signal (uses DB session)
# ---------------------------------------------------------------------------

def _signal_event_lag(ticker: str, session, closes: list) -> dict:
    """
    Detect credible high-impact events with no corresponding price reaction yet.

    Logic:
      - Query EventTickerImpact for this ticker from the last 3 days
      - Qualifying event: credibility > 0.6 AND |impact_score| > 0.5
                          AND narrative_stage in ("emerging", "developing")
      - Price lag check: |momentum_5d| < 0.02 (stock hasn't reacted yet)
      - Score = credibility * impact_score * (1 - |momentum_5d| / 0.05) clamped [0, 1]
      - Direction = sign of impact_score

    Returns dict: {score, direction, detail}
    """
    import database as db

    default = {"score": 0.0, "direction": 0.0, "detail": "no qualifying events found"}

    try:
        cutoff_3d = datetime.utcnow() - timedelta(days=3)

        # Query recent EventTickerImpact rows joined to qualifying Events
        impacts = (
            session.query(db.EventTickerImpact, db.Event)
            .join(db.Event, db.Event.event_id == db.EventTickerImpact.event_id)
            .filter(
                db.EventTickerImpact.ticker == ticker,
                db.Event.timestamp >= cutoff_3d,
                db.Event.credibility_score > 0.6,
                db.Event.narrative_stage.in_(["emerging", "developing"]),
            )
            .all()
        )

        # Filter by impact_score threshold
        qualifying = [
            (impact, event)
            for impact, event in impacts
            if abs(impact.impact_score or 0.0) > 0.5
        ]

        if not qualifying:
            return default

        # Compute 5-day price momentum
        if len(closes) >= 5 and closes[-5] > 0:
            momentum_5d = (closes[-1] - closes[-5]) / closes[-5]
        else:
            momentum_5d = 0.0

        # Price hasn't reacted significantly yet
        if abs(momentum_5d) >= 0.02:
            return {
                "score": 0.0,
                "direction": 0.0,
                "detail": (
                    f"Event found but price already moved {momentum_5d*100:.2f}% "
                    f"(lag threshold: <2%)."
                ),
            }

        # Pick the qualifying event with the highest combined score
        best_impact, best_event = max(
            qualifying,
            key=lambda pair: (pair[1].credibility_score or 0.0) * abs(pair[0].impact_score or 0.0),
        )

        credibility = best_event.credibility_score or 0.0
        raw_impact = best_impact.impact_score or 0.0
        direction = 1.0 if raw_impact > 0 else -1.0

        # Score formula: credibility * |impact| * (1 - |momentum_5d| / 0.05)
        lag_factor = 1.0 - min(abs(momentum_5d) / 0.05, 1.0)
        score = credibility * abs(raw_impact) * lag_factor
        score = max(0.0, min(score, 1.0))

        detail = (
            f"Event lag detected: '{best_event.title}' "
            f"(credibility={credibility:.2f}, impact={raw_impact:+.2f}, "
            f"stage={best_event.narrative_stage}). "
            f"Price 5d momentum only {momentum_5d*100:.2f}% — event not yet priced in."
        )

        return {"score": score, "direction": direction, "detail": detail}

    except Exception as e:
        logger.debug("Event-lag calculation error for %s: %s", ticker, e)
        return {**default, "detail": f"calculation error: {e}"}


# ---------------------------------------------------------------------------
# Composite prediction
# ---------------------------------------------------------------------------

# Signal weights must sum to 1.0
_SIGNAL_WEIGHTS = {
    "bb_squeeze":    0.25,
    "volume_accum":  0.30,
    "rsi_divergence": 0.20,
    "event_lag":     0.25,
}


def predict_anomaly(ticker: str, session) -> dict:
    """
    Run all four pre-breakout signals for a single ticker and return a
    composite prediction dict.

    Returns:
    {
        "ticker": str,
        "prediction_score": float,   # 0-1, probability of upcoming anomaly
        "direction": float,           # +1 = likely up, -1 = likely down, 0 = uncertain
        "signals": {
            "bb_squeeze":    {"score": float, "direction": float, "detail": str},
            "volume_accum":  {"score": float, "direction": float, "detail": str},
            "rsi_divergence": {"score": float, "direction": float, "detail": str},
            "event_lag":     {"score": float, "direction": float, "detail": str},
        },
        "reasoning": str,             # human-readable summary
        "confidence": str,            # "high" / "moderate" / "low"
    }

    Returns None if prediction_score < 0.25 (noise filter).
    """
    # Check cache first
    cached = _cache_get(ticker)
    if cached is not None:
        logger.debug("Returning cached prediction for %s", ticker)
        return cached

    # Fetch price history from yfinance (wrapped to degrade gracefully)
    closes: list = []
    volumes: list = []
    highs: list = []
    lows: list = []

    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
        if hist is not None and len(hist) >= 20:
            closes = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()
            highs = hist["High"].tolist()
            lows = hist["Low"].tolist()
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        # All signals will return score=0 with insufficient data

    # Run each signal
    sig_bb = _signal_bollinger_squeeze(closes)
    sig_vol = _signal_volume_accumulation(closes, volumes, highs, lows)
    sig_rsi = _signal_rsi_divergence(closes)
    sig_lag = _signal_event_lag(ticker, session, closes)

    signals = {
        "bb_squeeze":    sig_bb,
        "volume_accum":  sig_vol,
        "rsi_divergence": sig_rsi,
        "event_lag":     sig_lag,
    }

    # Weighted composite score
    prediction_score = sum(
        _SIGNAL_WEIGHTS[name] * signals[name]["score"]
        for name in _SIGNAL_WEIGHTS
    )
    prediction_score = round(max(0.0, min(prediction_score, 1.0)), 4)

    # Weighted direction: use each signal's score as its direction weight
    direction_numerator = sum(
        signals[name]["score"] * signals[name]["direction"]
        for name in _SIGNAL_WEIGHTS
    )
    total_signal_score = sum(signals[name]["score"] for name in _SIGNAL_WEIGHTS)

    if total_signal_score > 0:
        raw_direction = direction_numerator / total_signal_score
    else:
        raw_direction = 0.0

    # Snap to +1 / -1 / 0 with a dead-zone around 0
    if raw_direction > 0.15:
        direction = 1.0
    elif raw_direction < -0.15:
        direction = -1.0
    else:
        direction = 0.0

    # Confidence thresholds
    if prediction_score >= 0.55:
        confidence = "high"
    elif prediction_score >= 0.35:
        confidence = "moderate"
    else:
        confidence = "low"

    # Human-readable reasoning
    active_signals = [
        name for name in _SIGNAL_WEIGHTS if signals[name]["score"] > 0
    ]
    direction_word = "upside" if direction > 0 else ("downside" if direction < 0 else "neutral")

    if active_signals:
        signal_summary = ", ".join(
            f"{name.replace('_', ' ')} ({signals[name]['score']:.2f})"
            for name in active_signals
        )
        reasoning = (
            f"{ticker} shows {confidence}-confidence pre-breakout setup "
            f"({direction_word} bias, score={prediction_score:.2f}). "
            f"Active signals: {signal_summary}."
        )
    else:
        reasoning = (
            f"{ticker} shows no significant pre-breakout signals "
            f"(composite score={prediction_score:.2f})."
        )

    result = {
        "ticker": ticker,
        "prediction_score": prediction_score,
        "direction": direction,
        "signals": signals,
        "reasoning": reasoning,
        "confidence": confidence,
    }

    # Apply noise filter: don't cache or return low-signal results
    if prediction_score < 0.25:
        # Still cache to avoid re-running yfinance for clearly weak tickers
        _cache_set(ticker, result)
        return result

    _cache_set(ticker, result)
    return result


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def batch_predict_anomalies(tickers: List[str], session) -> List[dict]:
    """
    Run predict_anomaly for all tickers.

    Returns list of prediction dicts where prediction_score >= 0.25,
    sorted by prediction_score descending.
    """
    results = []
    for ticker in tickers:
        try:
            pred = predict_anomaly(ticker, session)
            if pred["prediction_score"] >= 0.25:
                results.append(pred)
        except Exception as e:
            logger.warning("batch_predict_anomalies failed for %s: %s", ticker, e)

    results.sort(key=lambda p: p["prediction_score"], reverse=True)
    logger.info(
        "batch_predict_anomalies: %d/%d tickers above threshold (>=0.25)",
        len(results),
        len(tickers),
    )
    return results
