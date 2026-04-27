# Dynamic Stock Universe Expansion — Design Spec
**Date:** 2026-04-26  
**Status:** Approved  
**Scope:** Expand the geopolitical intelligence system from a 36-stock hardcoded universe to a dynamic, multi-sector, continuously-updated stock universe with consistent signal generation. Seed target: ~65 unique tickers (36 existing + ~29 new across AI infrastructure, defense, energy).

---

## 1. Problem Statement

The current system tracks 36 hardcoded stocks (24 semiconductors + 12 macro-context names). Signals only fire when new articles arrive and clear credibility thresholds. Three gaps:

1. **Universe breadth** — only semiconductors; AI infrastructure, defense, and energy are untracked despite being directly coupled to geopolitical events.
2. **Company event coverage** — ingestion feeds are almost entirely geopolitical/macro; earnings, lawsuits, SEC filings, and M&A are poorly covered.
3. **Signal consistency** — when news is slow the dashboard goes quiet; no mechanism re-scores stocks using fresh price data between news events.

---

## 2. Goals

- Expand tracked universe to ~65 seed stocks across semiconductors, AI infrastructure, defense, and energy (36 existing kept; ~29 net-new names added).
- Auto-promote any validated stock ($500M+ market cap, 3+ credible mentions) directly into the full scoring + ML pipeline.
- Supplement seed with ETF holdings (SOXX, SMH, ITA, XLE) refreshed weekly.
- Classify every ingested article by company event type (earnings, legal, regulatory, filing, product launch, executive, M&A, geopolitical).
- Add company-specific ingestion feeds: SEC EDGAR 8-K, earnings/PR wire, sector trade media.
- Re-score all tickers every 30 minutes using fresh price data.
- Run a daily price/volume anomaly scan that surfaces unusual moves as synthetic events even without news.

---

## 3. Architecture Overview

### New Services

| File | Purpose |
|------|---------|
| `services/universe_manager.py` | Manages the 3-layer ticker universe (seed + ETF sweep + news discovery) |
| `services/company_event_classifier.py` | Labels every article by event class before causal engine runs |
| `services/signal_scheduler.py` | 30-min re-score loop + daily price/volume anomaly scan |

### Modified Files

| File | Change |
|------|--------|
| `services/ingestion.py` | Adds SEC EDGAR, PR wire, and sector trade media feeds; calls classifier |
| `services/stock_discovery.py` | Removes confidence gate; auto-promotes when `DiscoveredTicker.article_count ≥ 3` AND cumulative avg credibility ≥ 0.50 |
| `seed_data.py` | TICKERS expanded from 36 → ~65 across four sectors; L3Harris corrected to ticker `LHX` |
| `main.py` | Adds `_migrate_db` columns; starts two new scheduler threads |
| `routers/stocks.py` | Adds `GET /stocks/universe` endpoint |
| `routers/admin.py` | Adds `POST /admin/universe-sweep` endpoint |

### Data Flow

```
Ingest article
    ↓
[company_event_classifier] → sets article.event_class
    ↓
[universe_manager] — checks if article mentions auto-promotable ticker
    ↓
[clustering → causal engine → scoring → ML] — unchanged
    ↓
[signal_scheduler]
    ├── every 30 min: re-score all Ticker table rows with fresh price data
    └── every 24 hrs: price/volume anomaly scan → synthetic events → alert pipeline
```

---

## 4. Three-Layer Universe

### Layer 1 — Curated Seed (~80 stocks)

Defined statically in `seed_data.py`. Always available on cold start. Never removed automatically.

**Semiconductors (24, existing):** NVDA, AMD, TSM, ASML, AMAT, LRCX, KLAC, MU, AVGO, ARM, INTC, QCOM, MRVL, SNPS, CDNS, TER, ON, NXPI, TXN, ADI, SMCI, GFS, UMC, MCHP

**AI Infrastructure (12, new):** MSFT, GOOGL, META, AMZN, ORCL, DELL, HPE, VRT, PLTR, NET, SNOW, CRWD

**Defense (net-new names):** NOC, GD, HII, LHX, LDOS, KTOS, RCAT *(LMT, RTX, BA already in existing seed — reclassified from "macro context" to "Defense")*

**Energy (net-new names):** COP, EOG, SLB, MPC, PSX, OXY, HAL, BP *(XOM, CVX already in existing seed — reclassified from "macro context" to "Energy")*

### Layer 2 — ETF Sweep (weekly + startup)

`universe_manager.py` pulls holdings from four ETFs via yfinance and upserts qualifying names into `Ticker` table:

- **SOXX** — iShares Semiconductor ETF
- **SMH** — VanEck Semiconductor ETF
- **ITA** — iShares US Aerospace & Defense ETF
- **XLE** — Energy Select Sector SPDR

**Filter:** Market cap ≥ $2B. Holdings below this floor are skipped (illiquid names generate garbage signals).

**Delisting behavior:** A stock removed from all ETFs keeps its `Ticker` row but `source` is updated to `etf_delisted`. It remains in scoring history but is deprioritized in ranking.

**Schedule:** Once at startup (after seed upsert), then weekly via `signal_scheduler.py`.

### Layer 3 — News-Driven Discovery (continuous)

Auto-promotion rule in `stock_discovery.py`:

> Any stock passing yfinance validation ($500M+ market cap, valid US ticker format) whose `DiscoveredTicker.article_count` reaches **3 or more** (cumulative across all ingestion cycles) with **cumulative average credibility ≥ 0.50** is immediately upserted into the `Ticker` table with `source = 'news_discovery'` and enters the full scoring pipeline on the next 30-minute cycle. The `article_count` field on the existing `DiscoveredTicker` row is incremented on each new mention rather than creating duplicate rows.

The `DiscoveredTicker` table remains as a staging area for stocks that haven't yet hit the promotion threshold (useful for the `/stocks/emerging` endpoint).

---

## 5. Company Event Classifier

**File:** `services/company_event_classifier.py`  
**Called:** Inside `ingestion.py` immediately after relevance filter passes, before clustering.  
**Output:** Stored in `articles.event_class` (new column).

### Eight Event Classes (priority order)

| Priority | Class | Trigger keywords |
|----------|-------|-----------------|
| 1 | `regulatory` | export control, ban, sanction, restriction, CFIUS, FCC, FDA approval |
| 2 | `legal` | lawsuit, sued, DOJ, SEC investigation, fraud, settlement, antitrust, indictment |
| 3 | `filing` | 8-K, 10-K, 10-Q, proxy, DEF 14A, Form 4, insider buying, insider selling |
| 4 | `ma` | acquires, merger, acquisition, takeover, buys, deal closed, buyout |
| 5 | `earnings` | beat, miss, EPS, revenue, quarterly results, guidance, outlook, raised guidance |
| 6 | `executive` | CEO, CFO, appointed, resigned, steps down, named president, promoted to |
| 7 | `product_launch` | announces, unveils, launches, new chip, new model, next-gen, partnership |
| 8 | `geopolitical` | war, sanction, tariff, Taiwan, export ban, military, invasion, missile |

Priority ordering matters: if an article matches both `earnings` and `regulatory` (e.g., "guidance cut due to export ban"), `regulatory` wins as it is higher-impact.

### Integration with Causal Engine

The causal engine's 15-class event type mapping reads `article.event_class` as a pre-classification hint:

- `earnings` → maps to `company_earnings_guidance`
- `regulatory` → maps to `export_restriction_*` or `geopolitical_sanctions`
- `ma` → maps to `merger_acquisition_tech`
- `legal` → maps to `regulatory_antitrust`

This eliminates redundant classification work and improves accuracy for company-specific events where the causal engine previously defaulted to generic sector/macro buckets.

---

## 6. Feed Expansion

### SEC EDGAR (company-specific filings)

```
SEC EDGAR Semiconductors   — 8-K filings mentioning "semiconductor"
SEC EDGAR AI Infrastructure — 8-K filings mentioning "artificial intelligence" + "data center"
SEC EDGAR Defense           — 8-K filings mentioning "defense" + "contract"
SEC EDGAR Energy            — 8-K filings mentioning "oil" OR "energy" + "production"
```

All EDGAR feeds: `is_primary: 1`, `strict_filter: False` (official source, no noise filter needed).

### Earnings & Corporate News Wire

```
Yahoo Finance Earnings RSS  — per-ticker RSS for all seed tickers
PR Newswire Technology      — tech sector press releases
BusinessWire Semiconductors — semiconductor/electronics press releases
```

All PR wire feeds: `is_primary: 0`, `strict_filter: True` (must match `COMPANY_RELEVANCE_ANCHORS`).

### Sector Trade Media

```
EE Times        — electronics/semiconductor trade
Defense News    — aerospace/defense trade
Oil Price News  — energy markets
The Information — AI/tech industry (premium, public feed only)
```

All trade media: `is_primary: 0`, `strict_filter: True`.

### `COMPANY_RELEVANCE_ANCHORS` Expansion

New anchor terms added for the three new sectors:

- **AI Infrastructure:** microsoft azure, google cloud, aws, meta ai, oracle cloud, palantir, cloudflare, snowflake, crowdstrike, vertiv, dell server, hpe greenlake
- **Defense:** lockheed, raytheon, northrop, general dynamics, huntington ingalls, l3harris, leidos, kratos, pentagon contract, defense budget, dod award
- **Energy:** exxonmobil, chevron production, conocophillips, eog resources, schlumberger, marathon petroleum, phillips 66, occidental, halliburton, oil output, lng export

---

## 7. Signal Scheduler

**File:** `services/signal_scheduler.py`

### Job 1 — 30-Minute Re-score

```
Every 30 minutes (guarded by _INGEST_LOCK — skips if ingestion running):
  1. Fetch price context for all Ticker rows (yfinance, 4h cache)
  2. For each ticker:
     a. Pull active EventTickerImpacts (last 7 days)
     b. Re-run calculate_opportunity() with current adaptive weights
     c. Upsert StockScore row
  3. Run alert sweep (existing try_trigger_alert)
  4. Invalidate stock_recommender cache
```

### Job 2 — Daily Price/Volume Anomaly Scan

```
Once per day at 16:30 ET (or configurable via DAILY_SCAN_HOUR env var):
  For each ticker in Ticker table:
    1. Fetch 20-day OHLCV history via yfinance
    2. Compute:
       vol_zscore   = (today_volume - avg_20d_vol) / std_20d_vol
       price_zscore = (today_return - avg_20d_return) / std_20d_return
       atr_multiple = today_range / avg_20d_atr
    3. If vol_zscore > 2.0 OR |price_zscore| > 2.0:
       → Create Event(title="Price/Volume Anomaly: {ticker}", event_type="price_anomaly")
       → Create EventTickerImpact(direction=sign(price_zscore), strength=min(|price_zscore|/3, 1.0))
       → Credibility = min(vol_zscore / 4.0, 0.85)  [vol confirms price move]
       → Run try_trigger_alert() — flows through existing alert pipeline unchanged
```

**Why synthetic events work:** The existing alert pipeline processes any `Event` row regardless of origin. A `price_anomaly` event flows through credibility check → stock scoring → ML prediction → alert thresholding identically to a news event. No special casing in downstream code.

### `main.py` Thread Changes

```python
# Three daemon threads replace the current single scheduled thread:
threading.Thread(target=_background_ingest,               daemon=True).start()
threading.Thread(target=_scheduled_ingest, args=(30,),    daemon=True).start()
threading.Thread(target=signal_scheduler.start_rescoring_loop,  daemon=True).start()
threading.Thread(target=signal_scheduler.start_daily_scan_loop, daemon=True).start()
```

---

## 8. Data Model Changes

### New Column: `articles.event_class`

```sql
ALTER TABLE articles ADD COLUMN event_class VARCHAR(32) DEFAULT 'geopolitical';
```

Values: `earnings | legal | regulatory | filing | product_launch | executive | ma | geopolitical | price_anomaly`

### New Columns: `tickers.source`, `tickers.enrolled_at`, `tickers.market_cap`

```sql
ALTER TABLE tickers ADD COLUMN source VARCHAR(32) DEFAULT 'seed';
ALTER TABLE tickers ADD COLUMN enrolled_at DATETIME;
ALTER TABLE tickers ADD COLUMN market_cap FLOAT;
```

`source` values: `seed | etf_sweep | news_discovery | etf_delisted`

All added via `_migrate_db()` in `main.py` — idempotent, safe on existing databases.

---

## 9. API Changes

### New Endpoints

**`GET /stocks/universe`**
Returns all tickers grouped by source layer and sector. Response:
```json
{
  "total": 95,
  "by_source": {"seed": 80, "etf_sweep": 12, "news_discovery": 3},
  "by_sector": {"Semiconductors": 24, "AI Infrastructure": 14, ...},
  "tickers": [{"ticker": "NVDA", "source": "seed", "sector": "Semiconductors", ...}]
}
```

**`POST /admin/universe-sweep`**
Manually triggers ETF holdings sweep. Returns count of new tickers enrolled.

### Modified Endpoints

**`GET /stocks/`** — adds optional `?source=seed|etf_sweep|news_discovery` filter  
**`GET /admin/status`** — extends response with universe breakdown by source layer

---

## 10. Promotion Rule Summary

| Layer | Trigger | Market Cap Floor | Confidence Gate | Pipeline Entry |
|-------|---------|-----------------|-----------------|----------------|
| Seed | Static (deploy time) | None | None | Immediate |
| ETF Sweep | Weekly + startup | $2B | None | Next 30-min cycle |
| News Discovery | 3+ mentions, avg cred ≥ 0.50 | $500M | None (removed) | Next 30-min cycle |

---

## 11. Out of Scope

- Mobile app or native desktop notifications (Phase B per CLAUDE.md)
- Multi-user authentication
- Portfolio integration or position tracking
- Backtesting framework
- Sectors beyond semiconductors, AI infrastructure, defense, and energy
- Non-US exchanges (ADRs like TSM and ASML are included; pure foreign-listed names are not)

---

## 12. Success Criteria

- Universe grows from 36 → 65+ tickers immediately on restart (seed expansion); further grows via ETF sweep and news discovery
- At least one signal generated per day even during quiet news periods (via daily scan)
- Company-specific events (earnings, filings, legal) appear in daily brief with correct `event_class` label
- Auto-promoted tickers from news discovery appear in `/stocks/signals` ranked alongside seed tickers
- Re-score cycle completes in < 60 seconds for 100-ticker universe
