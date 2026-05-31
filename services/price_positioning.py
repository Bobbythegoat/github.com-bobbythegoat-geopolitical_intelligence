"""
services/price_positioning.py
------------------------------
"Has this stock already exploded?" detection.

The opportunity edge of this system is catching events *before* the price has
fully absorbed them.  A name that has already run 40% (e.g. DELL after an AI
re-rating) offers a worse fresh-entry risk/reward than the same story in a name
that hasn't moved yet — even if the news flow is identical.

This module turns the raw technical features from ``price_context.py`` into:

  extension_score   (0–1)  How much the stock has ALREADY moved / how stretched.
                           High  = overbought, at 52w highs, big recent run.
                           Low   = quiet, mid-range, or pulled back.
  freshness_score   (0–1)  Inverse of extension — "room left to run" before the
                           crowd is fully in.  Used as the price-reaction-lag /
                           underreaction proxy in the opportunity score.
  price_status      (str)  Human label: fresh | early_move | extended |
                           overextended | pulled_back.
  is_extended       (bool) True once the name is meaningfully stretched.
  penalty_mult      (0–1)  Multiplier to dampen a recommendation score for names
                           that already ran (1.0 = no penalty, ~0.4 = heavy).
  note              (str)  One-line plain-English explanation, with the numbers.

Design intent (per user feedback): do NOT silently hide extended names.  Surface
the label + the underlying numbers so the user can make their own inference and
research further.  The score is merely down-weighted, not zeroed.

Everything here is pure / deterministic given a price-context dict, so it is
trivially unit-testable and never makes a network call itself.
"""

from typing import Dict


# Thresholds (tunable).  Chosen to be deliberately conservative so we flag the
# clearly-already-moved names, not every stock in a mild uptrend.
_OVEREXTENDED = 0.72
_EXTENDED     = 0.52
_EARLY_MOVE   = 0.34

# How much of the recommendation score an overextended name can lose.
_MAX_PENALTY  = 0.60


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _scale(value: float, full: float) -> float:
    """Map a positive ``value`` onto [0,1] where ``full`` maps to 1.0.

    Negative values map to 0.0 (a falling stock is not 'extended').
    """
    if full <= 0:
        return 0.0
    return _clamp01(value / full)


def compute_extension(ctx: Dict) -> float:
    """Composite 'how much has this already moved' score in [0,1].

    Components (each mapped to 0–1) and their weight:
      RSI(14)            0.28  — momentum/overbought oscillator
      proximity to 52w   0.24  — buying at the highs has worse asymmetry
      30-day momentum    0.24  — the recent explosive move itself
      90-day momentum    0.12  — the broader run-up
      relative strength  0.12  — already outperformed the market

    A neutral / mid-range stock lands near 0; a name that is overbought, sitting
    on its 52-week high after a sharp run lands near 1.
    """
    rsi      = ctx.get("rsi_14", 50.0)
    pct_high = ctx.get("pct_from_52w_high", -0.10)   # 0 = at high, -0.30 = 30% below
    mom_30   = ctx.get("momentum_30d", 0.0)
    mom_90   = ctx.get("momentum_90d", 0.0)
    rel_str  = ctx.get("rel_strength_90d", 0.0)

    # RSI: 50 -> 0, 90 -> 1 (overbought territory drives the score)
    rsi_c = _clamp01((rsi - 50.0) / 40.0)

    # Proximity to 52w high: at the high (0) -> 1, 25%+ below -> 0
    near_high_c = _clamp01(1.0 + (pct_high / 0.25))

    # Recent run: +30% in 30d -> 1 ; +60% in 90d -> 1 ; +30% rel vs SPY -> 1
    mom30_c   = _scale(mom_30,  0.30)
    mom90_c   = _scale(mom_90,  0.60)
    rel_c     = _scale(rel_str, 0.30)

    extension = (
        0.28 * rsi_c
        + 0.24 * near_high_c
        + 0.24 * mom30_c
        + 0.12 * mom90_c
        + 0.12 * rel_c
    )
    return round(_clamp01(extension), 4)


def classify(ctx: Dict) -> Dict:
    """Return the full price-positioning summary for a price-context dict.

    See module docstring for field meanings.
    """
    extension = compute_extension(ctx)
    freshness = round(1.0 - extension, 4)

    rsi      = ctx.get("rsi_14", 50.0)
    pct_high = ctx.get("pct_from_52w_high", -0.10)
    mom_30   = ctx.get("momentum_30d", 0.0)

    # Status label.  "pulled_back" is a special low-extension case where the
    # stock has actively sold off — often the *best* fresh-entry setup on a
    # still-credible event, so we name it distinctly rather than lumping it in
    # with generic "fresh".
    if extension >= _OVEREXTENDED:
        status = "overextended"
    elif extension >= _EXTENDED:
        status = "extended"
    elif extension >= _EARLY_MOVE:
        status = "early_move"
    elif mom_30 <= -0.08 and rsi <= 45:
        status = "pulled_back"
    else:
        status = "fresh"

    is_extended = extension >= _EXTENDED

    # Penalty multiplier: no penalty until 'early_move', then ramp linearly to
    # _MAX_PENALTY at full extension.  Keeps fresh names untouched.
    if extension <= _EARLY_MOVE:
        penalty_mult = 1.0
    else:
        ramp = (extension - _EARLY_MOVE) / (1.0 - _EARLY_MOVE)
        penalty_mult = round(1.0 - _MAX_PENALTY * _clamp01(ramp), 4)

    note = _build_note(status, rsi, pct_high, mom_30)

    return {
        "extension_score":   extension,
        "freshness_score":   freshness,
        "price_status":      status,
        "is_extended":       is_extended,
        "penalty_mult":      penalty_mult,
        "rsi_14":            round(rsi, 1),
        "pct_from_52w_high": round(pct_high, 4),
        "momentum_30d":      round(mom_30, 4),
        "note":              note,
    }


_STATUS_LABELS = {
    "fresh":        "Fresh — has not run yet",
    "early_move":   "Early move — starting to run",
    "extended":     "Extended — already moved a lot",
    "overextended": "Overextended — likely already priced in",
    "pulled_back":  "Pulled back — sold off, possible re-entry",
}


def status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status)


def _build_note(status: str, rsi: float, pct_high: float, mom_30: float) -> str:
    pct_high_txt = f"{pct_high * 100:+.0f}% vs 52w high"
    mom_txt      = f"{mom_30 * 100:+.0f}% in 30d"
    rsi_txt      = f"RSI {rsi:.0f}"

    if status == "overextended":
        head = "Already exploded — fresh-entry opportunity is lower."
    elif status == "extended":
        head = "Already moved a lot — late to the trade, weigh risk/reward."
    elif status == "early_move":
        head = "Move underway but not crowded yet."
    elif status == "pulled_back":
        head = "Sold off recently — potential re-entry if the thesis holds."
    else:
        head = "Hasn't run yet — better fresh-entry asymmetry."

    return f"{head} ({rsi_txt}, {pct_high_txt}, {mom_txt})"


def summarize_ticker(ticker: str) -> Dict:
    """Convenience wrapper: fetch live price context and classify.

    Falls back to a neutral classification if price data is unavailable, so
    callers never have to handle a network failure themselves.
    """
    try:
        from services.price_context import fetch_price_context
        ctx = fetch_price_context(ticker)
    except Exception:
        ctx = {}
    result = classify(ctx)
    result["ticker"] = ticker
    return result
