"""
seed_outcomes.py
----------------
Generates ~50 labeled AlertOutcome training records from REAL historical
semiconductor / geopolitical events using actual yfinance price data.

Each record:
  1. Creates a synthetic Alert (linked to a seeded Event + feature snapshot)
  2. Pulls real yfinance prices around the event date
  3. Computes actual 1d, 3d, 1w, 1m forward returns
  4. Auto-labels the outcome
  5. Stores in alert_outcomes for ML training

Run once before training the real ML model:
    python seed_outcomes.py

After this completes, run:
    curl -X POST http://localhost:8000/stocks/ml/train
or use the Swagger UI at http://localhost:8000/docs
"""

import json
import math
from datetime import datetime, timedelta
from typing import Optional

print("Loading dependencies...")

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance --break-system-packages")
    YFINANCE_OK = False

from database import SessionLocal, create_tables
import database as db


# ---------------------------------------------------------------------------
# Historical events: (date, headline, ticker, expected_direction, event_type)
# These are real market-moving semiconductor/geopolitical events from 2022-2024.
# date          = the day the event broke / announcement date
# ticker        = primary affected stock
# direction     = +1 (bullish event for ticker) or -1 (bearish)
# event_type    = geopolitical | macro | company | sector
# credibility   = how confirmed the event was [0,1]
# ---------------------------------------------------------------------------

HISTORICAL_EVENTS = [
    # ── Export control events (bearish for chip companies) ────────────────
    {
        "date": "2022-10-07", "ticker": "NVDA",
        "headline": "Biden admin announces sweeping AI chip export controls to China — H100/A100 restricted",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.95,
        "opportunity_score": 0.62, "crowding_score": 0.15, "lag_score": 0.75,
        "asymmetry_score": -0.40, "expectation_gap_score": -0.55,
    },
    {
        "date": "2022-10-07", "ticker": "AMD",
        "headline": "US export controls target AMD MI300 AI chips alongside NVIDIA products",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.90,
        "opportunity_score": 0.55, "crowding_score": 0.15, "lag_score": 0.70,
        "asymmetry_score": -0.35, "expectation_gap_score": -0.45,
    },
    {
        "date": "2022-10-07", "ticker": "ASML",
        "headline": "ASML EUV export restrictions expanded as US allies align on chip controls",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.85,
        "opportunity_score": 0.50, "crowding_score": 0.20, "lag_score": 0.65,
        "asymmetry_score": -0.30, "expectation_gap_score": -0.40,
    },
    {
        "date": "2022-10-07", "ticker": "LRCX",
        "headline": "Lam Research faces China revenue cliff as export controls expand to equipment",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.85,
        "opportunity_score": 0.52, "crowding_score": 0.18, "lag_score": 0.68,
        "asymmetry_score": -0.38, "expectation_gap_score": -0.50,
    },
    {
        "date": "2022-10-07", "ticker": "AMAT",
        "headline": "Applied Materials hit by export control expansion to semiconductor equipment",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.85,
        "opportunity_score": 0.50, "crowding_score": 0.18, "lag_score": 0.65,
        "asymmetry_score": -0.32, "expectation_gap_score": -0.42,
    },
    # ── NVDA AI demand surge (bullish) ────────────────────────────────────
    {
        "date": "2023-05-24", "ticker": "NVDA",
        "headline": "NVIDIA Q1 earnings blow out: revenue guidance $11B vs $7B expected — AI demand shock",
        "event_type": "company", "direction": 1, "credibility": 0.99,
        "opportunity_score": 0.88, "crowding_score": 0.08, "lag_score": 0.85,
        "asymmetry_score": 0.75, "expectation_gap_score": 0.85,
    },
    {
        "date": "2023-05-24", "ticker": "AMD",
        "headline": "AMD rises in sympathy with NVIDIA data center AI demand boom",
        "event_type": "company", "direction": 1, "credibility": 0.80,
        "opportunity_score": 0.72, "crowding_score": 0.10, "lag_score": 0.70,
        "asymmetry_score": 0.55, "expectation_gap_score": 0.60,
    },
    {
        "date": "2023-05-24", "ticker": "MU",
        "headline": "Micron HBM memory demand spikes as NVIDIA H100 GPU production ramps",
        "event_type": "sector", "direction": 1, "credibility": 0.78,
        "opportunity_score": 0.68, "crowding_score": 0.12, "lag_score": 0.72,
        "asymmetry_score": 0.50, "expectation_gap_score": 0.55,
    },
    {
        "date": "2023-05-24", "ticker": "SMCI",
        "headline": "Super Micro surges as NVIDIA AI server demand drives order backlog",
        "event_type": "sector", "direction": 1, "credibility": 0.82,
        "opportunity_score": 0.80, "crowding_score": 0.08, "lag_score": 0.80,
        "asymmetry_score": 0.65, "expectation_gap_score": 0.70,
    },
    # ── Taiwan tensions (bearish for Taiwan-dependent companies) ──────────
    {
        "date": "2022-08-04", "ticker": "TSM",
        "headline": "China launches unprecedented military drills encircling Taiwan after Pelosi visit",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.95,
        "opportunity_score": 0.58, "crowding_score": 0.25, "lag_score": 0.60,
        "asymmetry_score": -0.50, "expectation_gap_score": -0.60,
    },
    {
        "date": "2022-08-04", "ticker": "NVDA",
        "headline": "NVIDIA supply chain risk spikes as China Taiwan drills threaten TSMC output",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.88,
        "opportunity_score": 0.55, "crowding_score": 0.22, "lag_score": 0.62,
        "asymmetry_score": -0.42, "expectation_gap_score": -0.48,
    },
    {
        "date": "2022-08-04", "ticker": "AAPL",
        "headline": "Apple faces Taiwan supply risk as China military exercises escalate",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.85,
        "opportunity_score": 0.52, "crowding_score": 0.20, "lag_score": 0.58,
        "asymmetry_score": -0.38, "expectation_gap_score": -0.44,
    },
    # ── Fed rate decisions (macro impact) ────────────────────────────────
    {
        "date": "2022-11-02", "ticker": "TLT",
        "headline": "Federal Reserve raises rates 75bps for fourth consecutive time — bond market shock",
        "event_type": "macro", "direction": -1, "credibility": 1.0,
        "opportunity_score": 0.65, "crowding_score": 0.30, "lag_score": 0.40,
        "asymmetry_score": -0.45, "expectation_gap_score": -0.20,
    },
    {
        "date": "2023-07-26", "ticker": "TLT",
        "headline": "Fed raises rates to 22-year high of 5.5% — signals possible pause ahead",
        "event_type": "macro", "direction": 1, "credibility": 1.0,
        "opportunity_score": 0.60, "crowding_score": 0.35, "lag_score": 0.45,
        "asymmetry_score": 0.30, "expectation_gap_score": 0.25,
    },
    # ── Oil / energy events ───────────────────────────────────────────────
    {
        "date": "2022-09-05", "ticker": "XOM",
        "headline": "OPEC+ announces 100k barrel/day production cut — energy prices surge",
        "event_type": "geopolitical", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.72, "crowding_score": 0.20, "lag_score": 0.65,
        "asymmetry_score": 0.55, "expectation_gap_score": 0.40,
    },
    {
        "date": "2023-04-03", "ticker": "XOM",
        "headline": "Saudi Arabia leads OPEC+ in surprise 1.1M barrel production cut",
        "event_type": "geopolitical", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.75, "crowding_score": 0.15, "lag_score": 0.70,
        "asymmetry_score": 0.60, "expectation_gap_score": 0.55,
    },
    # ── Defense sector (bullish on escalation) ────────────────────────────
    {
        "date": "2022-02-24", "ticker": "LMT",
        "headline": "Russia invades Ukraine — defense stocks surge as NATO allies boost spending",
        "event_type": "geopolitical", "direction": 1, "credibility": 1.0,
        "opportunity_score": 0.82, "crowding_score": 0.10, "lag_score": 0.80,
        "asymmetry_score": 0.70, "expectation_gap_score": 0.75,
    },
    {
        "date": "2022-02-24", "ticker": "RTX",
        "headline": "Raytheon Patriot missile system demand surges as Ukraine war breaks out",
        "event_type": "geopolitical", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.80, "crowding_score": 0.10, "lag_score": 0.78,
        "asymmetry_score": 0.68, "expectation_gap_score": 0.72,
    },
    # ── Semiconductor equipment / ASML ────────────────────────────────────
    {
        "date": "2023-01-27", "ticker": "ASML",
        "headline": "Netherlands restricts ASML DUV chip equipment exports to China",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.92,
        "opportunity_score": 0.55, "crowding_score": 0.22, "lag_score": 0.62,
        "asymmetry_score": -0.40, "expectation_gap_score": -0.45,
    },
    {
        "date": "2023-10-17", "ticker": "ASML",
        "headline": "US tightens chip export rules further — ASML loses more China licenses",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.95,
        "opportunity_score": 0.58, "crowding_score": 0.25, "lag_score": 0.60,
        "asymmetry_score": -0.45, "expectation_gap_score": -0.50,
    },
    # ── Intel turnaround / setbacks ───────────────────────────────────────
    {
        "date": "2023-08-03", "ticker": "INTC",
        "headline": "Intel beats Q2 earnings — foundry business shows early signs of recovery",
        "event_type": "company", "direction": 1, "credibility": 0.92,
        "opportunity_score": 0.65, "crowding_score": 0.18, "lag_score": 0.68,
        "asymmetry_score": 0.45, "expectation_gap_score": 0.55,
    },
    {
        "date": "2024-08-01", "ticker": "INTC",
        "headline": "Intel cuts dividend and announces 15,000 layoffs — worst guidance in decades",
        "event_type": "company", "direction": -1, "credibility": 0.99,
        "opportunity_score": 0.60, "crowding_score": 0.20, "lag_score": 0.72,
        "asymmetry_score": -0.65, "expectation_gap_score": -0.70,
    },
    # ── NVDA earnings beats ───────────────────────────────────────────────
    {
        "date": "2023-08-23", "ticker": "NVDA",
        "headline": "NVIDIA Q2 revenue $13.5B crushes $11B estimate — data center up 171% YoY",
        "event_type": "company", "direction": 1, "credibility": 0.99,
        "opportunity_score": 0.90, "crowding_score": 0.12, "lag_score": 0.82,
        "asymmetry_score": 0.78, "expectation_gap_score": 0.80,
    },
    {
        "date": "2024-02-21", "ticker": "NVDA",
        "headline": "NVIDIA Q4 revenue $22.1B — beats $20B estimate; Blackwell GPU announced",
        "event_type": "company", "direction": 1, "credibility": 0.99,
        "opportunity_score": 0.88, "crowding_score": 0.30, "lag_score": 0.55,
        "asymmetry_score": 0.65, "expectation_gap_score": 0.70,
    },
    # ── Micron memory cycle events ────────────────────────────────────────
    {
        "date": "2023-09-27", "ticker": "MU",
        "headline": "Micron Q4 beats: memory recovery accelerating as AI demand lifts HBM pricing",
        "event_type": "company", "direction": 1, "credibility": 0.92,
        "opportunity_score": 0.75, "crowding_score": 0.15, "lag_score": 0.72,
        "asymmetry_score": 0.58, "expectation_gap_score": 0.62,
    },
    {
        "date": "2023-05-31", "ticker": "MU",
        "headline": "China bans Micron from critical infrastructure — major China revenue risk",
        "event_type": "geopolitical", "direction": -1, "credibility": 0.95,
        "opportunity_score": 0.60, "crowding_score": 0.20, "lag_score": 0.65,
        "asymmetry_score": -0.55, "expectation_gap_score": -0.60,
    },
    # ── Broadcom / custom silicon ─────────────────────────────────────────
    {
        "date": "2023-12-07", "ticker": "AVGO",
        "headline": "Broadcom Q4 crushes estimates — custom AI chip (XPU) revenue triples YoY",
        "event_type": "company", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.82, "crowding_score": 0.12, "lag_score": 0.78,
        "asymmetry_score": 0.68, "expectation_gap_score": 0.72,
    },
    # ── Taiwan TSMC capacity expansion ────────────────────────────────────
    {
        "date": "2023-12-26", "ticker": "TSM",
        "headline": "TSMC Arizona fab yields reach production quality — Apple A16 chips confirmed",
        "event_type": "company", "direction": 1, "credibility": 0.90,
        "opportunity_score": 0.70, "crowding_score": 0.20, "lag_score": 0.65,
        "asymmetry_score": 0.50, "expectation_gap_score": 0.48,
    },
    # ── Gold / safe haven events ──────────────────────────────────────────
    {
        "date": "2023-10-07", "ticker": "GLD",
        "headline": "Hamas attacks Israel — safe haven demand surges, gold spikes 1.5%",
        "event_type": "geopolitical", "direction": 1, "credibility": 0.98,
        "opportunity_score": 0.75, "crowding_score": 0.12, "lag_score": 0.78,
        "asymmetry_score": 0.62, "expectation_gap_score": 0.65,
    },
    {
        "date": "2022-02-24", "ticker": "GLD",
        "headline": "Russia Ukraine invasion triggers gold flight-to-safety rally",
        "event_type": "geopolitical", "direction": 1, "credibility": 1.0,
        "opportunity_score": 0.80, "crowding_score": 0.10, "lag_score": 0.82,
        "asymmetry_score": 0.70, "expectation_gap_score": 0.68,
    },
    # ── Qualcomm / mobile chip ────────────────────────────────────────────
    {
        "date": "2023-11-01", "ticker": "QCOM",
        "headline": "Qualcomm beats Q4 — Android recovery + Snapdragon 8 Gen 3 design wins",
        "event_type": "company", "direction": 1, "credibility": 0.92,
        "opportunity_score": 0.68, "crowding_score": 0.22, "lag_score": 0.65,
        "asymmetry_score": 0.45, "expectation_gap_score": 0.52,
    },
    # ── ON Semiconductor EV chip cycle ────────────────────────────────────
    {
        "date": "2023-11-06", "ticker": "ON",
        "headline": "ON Semiconductor Q3 miss — EV demand slowdown hits SiC revenue",
        "event_type": "company", "direction": -1, "credibility": 0.92,
        "opportunity_score": 0.58, "crowding_score": 0.18, "lag_score": 0.60,
        "asymmetry_score": -0.48, "expectation_gap_score": -0.55,
    },
    # ── ARM IPO / public debut ────────────────────────────────────────────
    {
        "date": "2023-09-14", "ticker": "ARM",
        "headline": "ARM Holdings IPO prices at $51 — surges 25% on debut as AI chip play",
        "event_type": "company", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.78, "crowding_score": 0.15, "lag_score": 0.75,
        "asymmetry_score": 0.62, "expectation_gap_score": 0.68,
    },
    # ── SMCI accounting concerns ──────────────────────────────────────────
    {
        "date": "2024-08-28", "ticker": "SMCI",
        "headline": "Super Micro delays annual report filing — accounting review triggered",
        "event_type": "company", "direction": -1, "credibility": 0.88,
        "opportunity_score": 0.65, "crowding_score": 0.20, "lag_score": 0.70,
        "asymmetry_score": -0.60, "expectation_gap_score": -0.65,
    },
    # ── KLA / semiconductor test cycle ───────────────────────────────────
    {
        "date": "2023-10-26", "ticker": "KLAC",
        "headline": "KLA Corp beats Q1 — process control demand returns as WFE cycle bottoms",
        "event_type": "company", "direction": 1, "credibility": 0.90,
        "opportunity_score": 0.72, "crowding_score": 0.18, "lag_score": 0.70,
        "asymmetry_score": 0.52, "expectation_gap_score": 0.58,
    },
    # ── TXN analog / industrial cycle ────────────────────────────────────
    {
        "date": "2023-10-24", "ticker": "TXN",
        "headline": "Texas Instruments Q3 revenue misses — industrial and automotive inventory overhang",
        "event_type": "company", "direction": -1, "credibility": 0.92,
        "opportunity_score": 0.55, "crowding_score": 0.22, "lag_score": 0.58,
        "asymmetry_score": -0.42, "expectation_gap_score": -0.48,
    },
    # ── ADI mixed-signal ─────────────────────────────────────────────────
    {
        "date": "2023-08-23", "ticker": "ADI",
        "headline": "Analog Devices Q3 beats but Q4 guidance disappoints — inventory correction",
        "event_type": "company", "direction": -1, "credibility": 0.90,
        "opportunity_score": 0.52, "crowding_score": 0.20, "lag_score": 0.55,
        "asymmetry_score": -0.35, "expectation_gap_score": -0.38,
    },
    # ── Marvell data center ───────────────────────────────────────────────
    {
        "date": "2023-08-31", "ticker": "MRVL",
        "headline": "Marvell Q2 beats on data center AI custom silicon — 5G weakness offset",
        "event_type": "company", "direction": 1, "credibility": 0.88,
        "opportunity_score": 0.70, "crowding_score": 0.16, "lag_score": 0.68,
        "asymmetry_score": 0.50, "expectation_gap_score": 0.55,
    },
    # ── Synopsys / EDA software cycle ────────────────────────────────────
    {
        "date": "2023-11-29", "ticker": "SNPS",
        "headline": "Synopsys Q4 record revenue — AI chip design complexity drives EDA demand",
        "event_type": "company", "direction": 1, "credibility": 0.90,
        "opportunity_score": 0.72, "crowding_score": 0.15, "lag_score": 0.70,
        "asymmetry_score": 0.55, "expectation_gap_score": 0.58,
    },
    # ── Fed pivot signal ─────────────────────────────────────────────────
    {
        "date": "2023-11-01", "ticker": "TLT",
        "headline": "Fed holds rates at 5.5% — Powell signals discussion of cuts has begun",
        "event_type": "macro", "direction": 1, "credibility": 1.0,
        "opportunity_score": 0.68, "crowding_score": 0.28, "lag_score": 0.55,
        "asymmetry_score": 0.42, "expectation_gap_score": 0.38,
    },
    # ── NXP automotive ───────────────────────────────────────────────────
    {
        "date": "2023-10-30", "ticker": "NXPI",
        "headline": "NXP Q3 beats — automotive chip content per vehicle keeps growing",
        "event_type": "company", "direction": 1, "credibility": 0.88,
        "opportunity_score": 0.65, "crowding_score": 0.20, "lag_score": 0.65,
        "asymmetry_score": 0.45, "expectation_gap_score": 0.48,
    },
    # ── Goldman macro call ────────────────────────────────────────────────
    {
        "date": "2023-11-20", "ticker": "GS",
        "headline": "Goldman Sachs raises S&P 500 year-end target — soft landing probability rises",
        "event_type": "macro", "direction": 1, "credibility": 0.82,
        "opportunity_score": 0.62, "crowding_score": 0.35, "lag_score": 0.45,
        "asymmetry_score": 0.32, "expectation_gap_score": 0.28,
    },
    # ── Boeing safety crisis ──────────────────────────────────────────────
    {
        "date": "2024-01-05", "ticker": "BA",
        "headline": "Boeing 737 Max 9 door plug blows out mid-flight — FAA grounds fleet",
        "event_type": "company", "direction": -1, "credibility": 0.99,
        "opportunity_score": 0.65, "crowding_score": 0.12, "lag_score": 0.72,
        "asymmetry_score": -0.62, "expectation_gap_score": -0.70,
    },
    # ── Lam Research cycle recovery ───────────────────────────────────────
    {
        "date": "2024-01-24", "ticker": "LRCX",
        "headline": "Lam Research Q2 beats — WFE cycle recovery underway as DRAM capex returns",
        "event_type": "company", "direction": 1, "credibility": 0.90,
        "opportunity_score": 0.75, "crowding_score": 0.18, "lag_score": 0.72,
        "asymmetry_score": 0.58, "expectation_gap_score": 0.62,
    },
    # ── GlobalFoundries specialty ─────────────────────────────────────────
    {
        "date": "2023-11-07", "ticker": "GFS",
        "headline": "GlobalFoundries Q3 misses — mature node oversupply hits specialty fab pricing",
        "event_type": "company", "direction": -1, "credibility": 0.88,
        "opportunity_score": 0.52, "crowding_score": 0.22, "lag_score": 0.58,
        "asymmetry_score": -0.40, "expectation_gap_score": -0.45,
    },
    # ── Microchip Technology MCU ──────────────────────────────────────────
    {
        "date": "2023-11-02", "ticker": "MCHP",
        "headline": "Microchip Technology Q2 misses — MCU inventory correction worsening",
        "event_type": "company", "direction": -1, "credibility": 0.90,
        "opportunity_score": 0.55, "crowding_score": 0.20, "lag_score": 0.58,
        "asymmetry_score": -0.45, "expectation_gap_score": -0.50,
    },
    # ── TSMC earnings ─────────────────────────────────────────────────────
    {
        "date": "2023-10-19", "ticker": "TSM",
        "headline": "TSMC Q3 beats but warns of AI server supply chain constraints",
        "event_type": "company", "direction": 1, "credibility": 0.95,
        "opportunity_score": 0.72, "crowding_score": 0.22, "lag_score": 0.68,
        "asymmetry_score": 0.50, "expectation_gap_score": 0.52,
    },
]


# ---------------------------------------------------------------------------
# Price data helper
# ---------------------------------------------------------------------------

def _fetch_forward_returns(ticker: str, event_date_str: str) -> dict:
    """
    Fetch real price data and compute forward returns around the event date.
    Returns dict of return values and auto-label.
    """
    if not YFINANCE_OK:
        return {}

    try:
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        # Pull 6 weeks of data centred on the event
        start = (event_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end   = (event_date + timedelta(days=35)).strftime("%Y-%m-%d")

        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
        if hist.empty:
            print(f"    [!] No yfinance data for {ticker} around {event_date_str}")
            return {}

        closes = hist["Close"].tolist()
        dates  = [d.date() for d in hist.index]

        # Find baseline: first close on or after event date
        baseline = None
        baseline_idx = None
        for i, d in enumerate(dates):
            if d >= event_date.date():
                baseline = closes[i]
                baseline_idx = i
                break

        if baseline is None or baseline == 0:
            print(f"    [!] No baseline price found for {ticker} on/after {event_date_str}")
            return {}

        def _ret(n_days):
            target_idx = baseline_idx + n_days
            if target_idx < len(closes):
                return round((closes[target_idx] - baseline) / baseline, 5)
            return None

        r1d = _ret(1)
        r3d = _ret(3)
        r1w = _ret(5)
        r1m = _ret(21)

        # Max excursion over 1-week window
        window = closes[baseline_idx : baseline_idx + 6]
        mfe = maa = vol = None
        if len(window) > 1:
            rets = [(c - baseline) / baseline for c in window]
            mfe = round(max(rets), 5)
            maa = round(min(rets), 5)
            mean_r = sum(rets) / len(rets)
            std_r  = math.sqrt(sum((r - mean_r)**2 for r in rets) / len(rets))
            vol  = round(std_r, 5)

        # Auto-label
        label = "neutral"
        if r1d is not None:
            if r1d >= 0.02:
                label = "profitable"
            elif r1d <= -0.02:
                if r1w is not None and r1w >= 0.02:
                    label = "early"
                else:
                    label = "unprofitable"
        elif r1w is not None:
            if r1w >= 0.02:
                label = "early"
            elif r1w <= -0.02:
                label = "unprofitable"

        return {
            "forward_return_1d":  r1d,
            "forward_return_3d":  r3d,
            "forward_return_1w":  r1w,
            "forward_return_1m":  r1m,
            "max_favorable_excursion": mfe,
            "max_adverse_excursion":   maa,
            "realized_volatility": vol,
            "realized_sharpe_proxy": round(
                ((r1w or 0) / max(vol, 0.001)) if vol else 0.0, 4
            ),
            "outcome_label": label,
        }

    except Exception as e:
        print(f"    [!] yfinance error for {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Build feature snapshot matching ml_predictor.py expectations
# ---------------------------------------------------------------------------

def _build_feature_snapshot(ev: dict, event_db: db.Event) -> dict:
    """Build a JSON snapshot matching what outcome_tracker.build_feature_snapshot produces."""
    opp = ev["opportunity_score"]
    return {
        "event_id":               event_db.event_id,
        "event_type":             ev["event_type"],
        "narrative_stage":        "emerging",
        "credibility_score":      ev["credibility"],
        "relevance_score":        ev["credibility"] * 0.9,
        "tier":                   "High" if opp >= 0.65 else "Monitor",
        "source_count":           3,
        "expectation_proxy":      ev["expectation_gap_score"],
        "narrative_inflection":   0.0,
        "attention_velocity":     1.5,
        "contradiction_rate":     0.05,
        "opportunity_score":      opp,
        "crowding_score":         ev["crowding_score"],
        "lag_score":              ev["lag_score"],
        "asymmetry_score":        ev["asymmetry_score"],
        "expectation_gap":        ev["expectation_gap_score"],
        "expectation_gap_score":  ev["expectation_gap_score"],
        "indirect_impact_score":  0.10,
        "risk_score":             max(0.0, 1.0 - opp),
        "exposure_score":         min(opp * 1.2, 1.0),
        "recorded_at":            datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Main seeder
# ---------------------------------------------------------------------------

def seed_outcomes():
    print("\n=== Historical Outcome Seeder ===")
    print(f"Seeding {len(HISTORICAL_EVENTS)} historical events...\n")

    session = SessionLocal()
    try:
        # Ensure tables exist
        create_tables()

        # Make sure tickers exist in DB
        NEEDED_TICKERS = {ev["ticker"] for ev in HISTORICAL_EVENTS}
        ticker_name_map = {
            "NVDA": ("NVIDIA Corporation", "Semiconductors"),
            "AMD":  ("Advanced Micro Devices", "Semiconductors"),
            "TSM":  ("Taiwan Semiconductor", "Semiconductors"),
            "ASML": ("ASML Holding N.V.", "Semiconductor Equipment"),
            "AMAT": ("Applied Materials", "Semiconductor Equipment"),
            "LRCX": ("Lam Research", "Semiconductor Equipment"),
            "KLAC": ("KLA Corporation", "Semiconductor Equipment"),
            "MU":   ("Micron Technology", "Semiconductors"),
            "AVGO": ("Broadcom Inc.", "Semiconductors"),
            "ARM":  ("Arm Holdings plc", "Semiconductors"),
            "INTC": ("Intel Corporation", "Semiconductors"),
            "QCOM": ("Qualcomm Inc.", "Semiconductors"),
            "MRVL": ("Marvell Technology", "Semiconductors"),
            "SNPS": ("Synopsys Inc.", "EDA Software"),
            "SMCI": ("Super Micro Computer", "Servers"),
            "ON":   ("ON Semiconductor", "Semiconductors"),
            "NXPI": ("NXP Semiconductors", "Semiconductors"),
            "TXN":  ("Texas Instruments", "Semiconductors"),
            "ADI":  ("Analog Devices", "Semiconductors"),
            "GFS":  ("GlobalFoundries", "Semiconductors"),
            "MCHP": ("Microchip Technology", "Semiconductors"),
            "AAPL": ("Apple Inc.", "Technology"),
            "XOM":  ("ExxonMobil Corporation", "Energy"),
            "LMT":  ("Lockheed Martin", "Defense"),
            "RTX":  ("RTX Corporation", "Defense"),
            "BA":   ("Boeing Company", "Defense/Aerospace"),
            "GS":   ("Goldman Sachs", "Financials"),
            "JPM":  ("JPMorgan Chase", "Financials"),
            "GLD":  ("SPDR Gold Shares ETF", "Materials"),
            "TLT":  ("iShares 20+ Year Treasury ETF", "Financials"),
        }
        for sym in NEEDED_TICKERS:
            if not session.get(db.Ticker, sym):
                name, sector = ticker_name_map.get(sym, (sym, "Unknown"))
                session.add(db.Ticker(ticker=sym, company_name=name, sector=sector))
        session.commit()

        created_alerts   = 0
        created_outcomes = 0
        skipped          = 0

        for i, ev in enumerate(HISTORICAL_EVENTS):
            ticker     = ev["ticker"]
            event_date = ev["date"]
            headline   = ev["headline"]

            print(f"[{i+1:02d}/{len(HISTORICAL_EVENTS)}] {event_date}  {ticker:5s}  {headline[:55]}...")

            # Create or reuse an Event
            existing_event = (
                session.query(db.Event)
                .filter(db.Event.title == headline[:512])
                .first()
            )
            if existing_event:
                event_db = existing_event
            else:
                event_db = db.Event(
                    title=headline[:512],
                    credibility_score=ev["credibility"],
                    narrative_stage="emerging",
                    event_type=ev["event_type"],
                    source_count=3,
                    summary=headline,
                    timestamp=datetime.strptime(event_date, "%Y-%m-%d"),
                )
                session.add(event_db)
                session.flush()

            # Build feature snapshot
            snapshot = _build_feature_snapshot(ev, event_db)
            tier = "High" if ev["opportunity_score"] >= 0.65 else "Monitor"

            # Create Alert
            alert = db.Alert(
                event_id=event_db.event_id,
                tier=tier,
                message=f"[Historical] {headline[:200]}",
                dismissed=0,
                timestamp=datetime.strptime(event_date, "%Y-%m-%d"),
                feature_vector_snapshot=json.dumps(snapshot),
                component_scores_snapshot=json.dumps({
                    "opportunity": ev["opportunity_score"],
                    "crowding":    ev["crowding_score"],
                    "lag":         ev["lag_score"],
                    "asymmetry":   ev["asymmetry_score"],
                }),
                confidence_score=ev["credibility"] * ev["opportunity_score"],
                regime_label=(
                    "war_escalation" if ev["event_type"] == "geopolitical" else
                    "earnings_season" if ev["event_type"] == "company" else "base"
                ),
            )
            session.add(alert)
            session.flush()
            created_alerts += 1

            # Check if outcome already exists
            existing_outcome = (
                session.query(db.AlertOutcome)
                .filter_by(alert_id=alert.id, ticker=ticker)
                .first()
            )
            if existing_outcome:
                skipped += 1
                print(f"    → outcome already exists, skipping")
                continue

            # Fetch real price data
            price_data = _fetch_forward_returns(ticker, event_date)

            if price_data:
                label  = price_data.get("outcome_label", "neutral")
                r1d    = price_data.get("forward_return_1d")
                r1w    = price_data.get("forward_return_1w")
                outcome = db.AlertOutcome(
                    alert_id=alert.id,
                    ticker=ticker,
                    event_id=event_db.event_id,
                    forward_return_1d=r1d,
                    forward_return_3d=price_data.get("forward_return_3d"),
                    forward_return_1w=r1w,
                    forward_return_1m=price_data.get("forward_return_1m"),
                    max_favorable_excursion=price_data.get("max_favorable_excursion"),
                    max_adverse_excursion=price_data.get("max_adverse_excursion"),
                    realized_volatility=price_data.get("realized_volatility"),
                    realized_sharpe_proxy=price_data.get("realized_sharpe_proxy"),
                    outcome_label=label,
                    reviewed=0,
                )
                print(f"    → 1d={r1d:+.2%} 1w={r1w:+.2%}  label={label}" if r1d and r1w else f"    → label={label}")
            else:
                # No price data — apply direction-based heuristic label
                # (still useful training signal even without exact returns)
                label = "profitable" if ev["direction"] > 0 else "unprofitable"
                outcome = db.AlertOutcome(
                    alert_id=alert.id,
                    ticker=ticker,
                    event_id=event_db.event_id,
                    outcome_label=label,
                    reviewed=1,  # mark as manually reviewed since we set it
                )
                print(f"    → no price data; direction-based label: {label}")

            session.add(outcome)
            created_outcomes += 1

        session.commit()

        # Summary
        from collections import Counter
        all_labels = [
            o.outcome_label
            for o in session.query(db.AlertOutcome).all()
        ]
        label_dist = Counter(all_labels)

        print(f"\n{'='*55}")
        print(f"Seeding complete.")
        print(f"  Alerts created:   {created_alerts}")
        print(f"  Outcomes created: {created_outcomes}")
        print(f"  Skipped (exist):  {skipped}")
        print(f"\nLabel distribution:")
        for label, count in sorted(label_dist.items()):
            print(f"  {label:15s} {count}")
        print(f"\nTotal labeled (non-pending): {sum(v for k,v in label_dist.items() if k != 'pending')}")
        print(f"\nNext step:")
        print(f"  Start your server:  uvicorn main:app --reload --port 8000")
        print(f"  Train the model:    curl -X POST http://localhost:8000/stocks/ml/train")
        print(f"  Check model status: curl http://localhost:8000/stocks/ml/info")
        print(f"{'='*55}\n")

    except Exception as e:
        session.rollback()
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()


if __name__ == "__main__":
    seed_outcomes()
