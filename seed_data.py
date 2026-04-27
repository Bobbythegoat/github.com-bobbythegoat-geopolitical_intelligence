"""
seed_data.py
------------
Populates the database with realistic sample data for testing / demo.
Covers all three phases of the blueprint.
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Session

import database as db
from services.alerts import try_trigger_alert, generate_daily_brief


def seed(session: Session):
    _seed_tickers(session)
    _seed_articles_and_events(session)
    _seed_stock_scores(session)
    run_alerts_and_brief(session)


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

TICKERS = [
    # ── Primary semiconductor universe (CLAUDE.md primary focus) ──────────
    ("NVDA", "NVIDIA Corporation",                    "Semiconductors"),
    ("AMD",  "Advanced Micro Devices Inc.",            "Semiconductors"),
    ("TSM",  "Taiwan Semiconductor Manufacturing",    "Semiconductors"),
    ("ASML", "ASML Holding N.V.",                     "Semiconductor Equipment"),
    ("AMAT", "Applied Materials Inc.",                "Semiconductor Equipment"),
    ("LRCX", "Lam Research Corporation",              "Semiconductor Equipment"),
    ("KLAC", "KLA Corporation",                       "Semiconductor Equipment"),
    ("MU",   "Micron Technology Inc.",                "Semiconductors"),
    ("AVGO", "Broadcom Inc.",                         "Semiconductors"),
    ("ARM",  "Arm Holdings plc",                      "Semiconductors"),
    ("INTC", "Intel Corporation",                     "Semiconductors"),
    ("QCOM", "Qualcomm Inc.",                         "Semiconductors"),
    ("MRVL", "Marvell Technology Inc.",               "Semiconductors"),
    ("SNPS", "Synopsys Inc.",                         "EDA Software"),
    ("CDNS", "Cadence Design Systems Inc.",           "EDA Software"),
    ("TER",  "Teradyne Inc.",                         "Semiconductor Test"),
    ("ON",   "ON Semiconductor Corp.",                "Semiconductors"),
    ("NXPI", "NXP Semiconductors N.V.",               "Semiconductors"),
    ("TXN",  "Texas Instruments Inc.",                "Semiconductors"),
    ("ADI",  "Analog Devices Inc.",                   "Semiconductors"),
    ("SMCI", "Super Micro Computer Inc.",             "Servers / AI Infrastructure"),
    ("GFS",  "GlobalFoundries Inc.",                  "Semiconductors"),
    ("UMC",  "United Microelectronics Corporation",   "Semiconductors"),
    ("MCHP", "Microchip Technology Inc.",             "Semiconductors"),
    # ── Supporting universe (macro / geopolitical context) ────────────────
    ("AAPL", "Apple Inc.",                            "Technology"),
    ("XOM",  "ExxonMobil Corporation",                "Energy"),
    ("CVX",  "Chevron Corporation",                   "Energy"),
    ("LMT",  "Lockheed Martin Corporation",           "Defense"),
    ("RTX",  "RTX Corporation (Raytheon)",            "Defense"),
    ("BA",   "Boeing Company",                        "Defense/Aerospace"),
    ("GS",   "Goldman Sachs Group",                   "Financials"),
    ("JPM",  "JPMorgan Chase & Co.",                  "Financials"),
    ("GLD",  "SPDR Gold Shares ETF",                  "Materials"),
    ("TLT",  "iShares 20+ Year Treasury ETF",         "Financials"),
    ("UNG",  "United States Natural Gas ETF",         "Energy"),
    ("DBA",  "Invesco DB Agriculture Fund",           "Agriculture"),
    # ── AI Infrastructure (12, new) ──────────────────────────────────────────
    ("MSFT", "Microsoft Corporation",                 "AI Infrastructure"),
    ("GOOGL","Alphabet Inc.",                         "AI Infrastructure"),
    ("META", "Meta Platforms Inc.",                   "AI Infrastructure"),
    ("AMZN", "Amazon.com Inc.",                       "AI Infrastructure"),
    ("ORCL", "Oracle Corporation",                    "AI Infrastructure"),
    ("DELL", "Dell Technologies Inc.",                "AI Infrastructure"),
    ("HPE",  "Hewlett Packard Enterprise Co.",        "AI Infrastructure"),
    ("VRT",  "Vertiv Holdings Co.",                   "AI Infrastructure"),
    ("PLTR", "Palantir Technologies Inc.",            "AI Infrastructure"),
    ("NET",  "Cloudflare Inc.",                       "AI Infrastructure"),
    ("SNOW", "Snowflake Inc.",                        "AI Infrastructure"),
    ("CRWD", "CrowdStrike Holdings Inc.",             "AI Infrastructure"),
    # ── Defense — net-new names (LMT, RTX, BA already above) ─────────────────
    ("NOC",  "Northrop Grumman Corporation",          "Defense"),
    ("GD",   "General Dynamics Corporation",          "Defense"),
    ("HII",  "Huntington Ingalls Industries Inc.",    "Defense"),
    ("LHX",  "L3Harris Technologies Inc.",            "Defense"),
    ("LDOS", "Leidos Holdings Inc.",                  "Defense"),
    ("KTOS", "Kratos Defense & Security Solutions",   "Defense"),
    ("RCAT", "Red Cat Holdings Inc.",                 "Defense"),
    # ── Energy — net-new names (XOM, CVX already above) ──────────────────────
    ("COP",  "ConocoPhillips",                        "Energy"),
    ("EOG",  "EOG Resources Inc.",                    "Energy"),
    ("SLB",  "SLB (Schlumberger)",                    "Energy"),
    ("MPC",  "Marathon Petroleum Corporation",        "Energy"),
    ("PSX",  "Phillips 66",                           "Energy"),
    ("OXY",  "Occidental Petroleum Corporation",      "Energy"),
    ("HAL",  "Halliburton Company",                   "Energy"),
    ("BP",   "BP p.l.c.",                             "Energy"),
]

def _seed_tickers(session: Session):
    for ticker, name, sector in TICKERS:
        if not session.get(db.Ticker, ticker):
            session.add(db.Ticker(ticker=ticker, company_name=name, sector=sector))
    session.commit()


# ---------------------------------------------------------------------------
# Articles → Events
# ---------------------------------------------------------------------------

SAMPLE_ARTICLES = [
    # --- Event cluster 1: AI chip export controls (primary semiconductor focus) ---
    {
        "source": "Reuters",
        "headline": "US expands AI chip export restrictions targeting NVIDIA H100 and A100 to new countries",
        "content": (
            "The Biden administration announced new export control rules that expand restrictions "
            "on advanced AI semiconductors including NVIDIA's H100 and A100 chips to additional "
            "countries beyond China. NVIDIA shares fell 6% on the announcement. "
            "AMD's MI300X accelerator is also affected. The rules require new licenses for "
            "data centers in over 40 countries. ASML and AMAT also fell on fears of "
            "second-order restrictions on equipment."
        ),
        "is_primary": 1, "official": 1,
        "hours_ago": 5,
    },
    {
        "source": "Bloomberg",
        "headline": "NVIDIA scrambles to develop export-compliant AI chips for China market",
        "content": (
            "NVIDIA is racing to develop modified versions of its H20 chip that comply with "
            "US export control rules for China. The company generated $10 billion annually from "
            "China before restrictions. AMD is pursuing a similar strategy with its MI300 series. "
            "TSMC, which manufactures both companies' chips, faces complex compliance requirements. "
            "Analysts estimate revenue impact of $5-8 billion annually for NVIDIA."
        ),
        "is_primary": 0, "official": 0,
        "hours_ago": 4,
    },
    # --- Event cluster 2: Taiwan geopolitical risk / TSMC supply chain ---
    {
        "source": "Financial Times",
        "headline": "China conducts largest-ever military drills around Taiwan targeting semiconductor supply chain",
        "content": (
            "China's PLA launched unprecedented military exercises encircling Taiwan, "
            "deploying aircraft carriers, submarines, and ballistic missiles in zones that "
            "would blockade TSMC's chip shipments. TSMC shares dropped 8% in Taipei trading. "
            "Apple, NVIDIA, AMD, Qualcomm and Broadcom — all heavily reliant on TSMC — fell in sympathy. "
            "ASML, Lam Research and Applied Materials also declined as supply disruption risk surged."
        ),
        "is_primary": 1, "official": 0,
        "hours_ago": 20,
    },
    {
        "source": "Wall Street Journal",
        "headline": "TSMC accelerates Arizona fab ramp to reduce Taiwan concentration risk",
        "content": (
            "Taiwan Semiconductor Manufacturing Company announced it is accelerating its "
            "Arizona fabrication plant ramp to 3nm production ahead of schedule, partially "
            "driven by geopolitical risk diversification requests from Apple and NVIDIA. "
            "The plant is expected to produce 600,000 wafers annually by 2026. "
            "SMCI stands to benefit from increased US-based AI server production."
        ),
        "is_primary": 1, "official": 0,
        "hours_ago": 16,
    },
    # --- Event cluster 3: ASML/semiconductor equipment restrictions ---
    {
        "source": "Reuters",
        "headline": "Netherlands expands ASML export restrictions on DUV chip equipment to China",
        "content": (
            "The Dutch government has expanded restrictions on ASML's DUV lithography machine "
            "exports to China, following earlier EUV bans. ASML shares fell 4%. "
            "Lam Research and Applied Materials face similar pressure as the US pushes allies "
            "to align on semiconductor equipment controls. KLA Corporation also affected. "
            "China's SMIC foundry faces production constraints as a result."
        ),
        "is_primary": 1, "official": 1,
        "hours_ago": 30,
    },
    # --- Event cluster 4: AI infrastructure capex surge (bullish semiconductor signal) ---
    {
        "source": "Bloomberg",
        "headline": "Microsoft, Google, Amazon commit $300B in AI infrastructure capex — NVIDIA primary beneficiary",
        "content": (
            "Major hyperscalers announced combined capital expenditure plans exceeding $300 billion "
            "for AI infrastructure over the next three years. NVIDIA is the primary beneficiary "
            "with H100 and Blackwell GPU demand surging. Micron's HBM memory, used in NVIDIA GPUs, "
            "is also in tight supply. Super Micro Computer (SMCI) surged 12% on server demand. "
            "Broadcom's custom ASIC business is seen as a long-term beneficiary."
        ),
        "is_primary": 0, "official": 0,
        "hours_ago": 48,
    },
    # --- Event cluster 5: Middle East oil disruption (retained for macro context) ---
    {
        "source": "Reuters",
        "headline": "Iran threatens to block Strait of Hormuz amid US sanctions escalation",
        "content": (
            "Iran's Revolutionary Guard warned it could close the Strait of Hormuz "
            "to international shipping if new US sanctions targeting its oil exports take effect. "
            "The move would disrupt roughly 20% of global oil supply. "
            "OPEC members Saudi Arabia and UAE condemned the threat as destabilising. "
            "Crude oil prices surged 4% on the news."
        ),
        "is_primary": 1, "official": 0,
        "hours_ago": 10,
    },
    {
        "source": "Bloomberg",
        "headline": "Oil prices spike as Hormuz closure threat rattles energy markets",
        "content": (
            "Brent crude rose above $95 per barrel as markets priced in the risk of a "
            "Strait of Hormuz closure. Energy analysts warn XOM and CVX face supply "
            "chain disruptions while natural gas LNG prices also spiked. "
            "US officials called the threat 'destabilising and irresponsible'."
        ),
        "is_primary": 0, "official": 0,
        "hours_ago": 9,
    },
    {
        "source": "AP",
        "headline": "Pentagon deploys carrier strike group to Persian Gulf",
        "content": (
            "The US Department of Defense confirmed it is repositioning a carrier strike "
            "group to the Persian Gulf in response to escalating tensions with Iran. "
            "Defense contractors LMT and RTX shares rose on the announcement. "
            "The deployment is the largest US naval presence in the region since 2020."
        ),
        "is_primary": 1, "official": 1,
        "hours_ago": 7,
    },

    # --- Event cluster 2: Taiwan semiconductor tensions ---
    {
        "source": "Financial Times",
        "headline": "China conducts largest-ever military drills around Taiwan",
        "content": (
            "China's People's Liberation Army launched unprecedented military exercises "
            "encircling Taiwan, involving aircraft carriers, submarines, and ballistic missiles. "
            "Taiwan Semiconductor Manufacturing (TSMC) shares dropped 6% in pre-market trading. "
            "Apple and NVIDIA, both heavily reliant on TSMC chip production, fell in sympathy. "
            "The US called the exercises 'provocative and destabilising'."
        ),
        "is_primary": 1, "official": 0,
        "hours_ago": 20,
    },
    {
        "source": "Reuters",
        "headline": "US imposes new export controls on advanced AI chips bound for China",
        "content": (
            "The Biden administration announced sweeping new restrictions on the export of "
            "advanced AI semiconductors to China, targeting NVIDIA's H100 and A100 chips. "
            "NVIDIA stock fell 8%. The move is part of a broader effort to limit China's "
            "access to technology that could be used for military applications."
        ),
        "is_primary": 1, "official": 1,
        "hours_ago": 15,
    },

    # --- Event cluster 3: Fed rate decision ---
    {
        "source": "Federal Reserve",
        "headline": "Federal Reserve signals potential rate pause amid inflation concerns",
        "content": (
            "Federal Reserve Chair Jerome Powell indicated the central bank may pause "
            "further rate hikes at the next meeting, citing mixed economic signals. "
            "Treasury yields fell sharply. TLT ETF surged 3%. "
            "Goldman Sachs and JPMorgan analysts revised their rate forecasts lower. "
            "The S&P 500 rallied 1.5% on the news."
        ),
        "is_primary": 1, "official": 1,
        "hours_ago": 30,
    },

    # --- Event cluster 4: Ukraine/Russia grain corridor ---
    {
        "source": "BBC",
        "headline": "Russia withdraws from Black Sea grain deal, threatening global food supply",
        "content": (
            "Russia announced it is suspending participation in the Black Sea Grain Initiative, "
            "blocking Ukrainian wheat and sunflower oil exports. "
            "Global wheat futures jumped 7%. Agriculture ETF DBA surged. "
            "The UN warned of severe food security consequences for North Africa and the Middle East. "
            "Gold prices rose as investors sought safe-haven assets."
        ),
        "is_primary": 1, "official": 0,
        "hours_ago": 48,
    },
    {
        "source": "Reuters",
        "headline": "Ukraine wheat exports halt as Black Sea corridor collapses",
        "content": (
            "Ukrainian grain shipments have ground to a halt following Russia's withdrawal "
            "from the Black Sea deal. Egypt and Tunisia, heavily dependent on Ukrainian wheat, "
            "face acute shortages. DBA agriculture ETF hit a 3-month high. "
            "Gold (GLD) also advanced as geopolitical risk premium climbed."
        ),
        "is_primary": 0, "official": 0,
        "hours_ago": 46,
    },
]

def _seed_articles_and_events(session: Session):
    from services.clustering import cluster_article
    from services.processing import process_article

    now = datetime.utcnow()
    for art_data in SAMPLE_ARTICLES:
        article = db.Article(
            source=art_data["source"],
            headline=art_data["headline"],
            content=art_data["content"],
            timestamp=now - timedelta(hours=art_data["hours_ago"]),
            is_primary_source=art_data["is_primary"],
            has_official_confirm=art_data["official"],
        )
        session.add(article)
        session.flush()
        cluster_article(article.id, session)
        process_article(article.id, session)

    session.commit()


# ---------------------------------------------------------------------------
# Stock Scores (ensure rows exist for all tickers)
# ---------------------------------------------------------------------------

def _seed_stock_scores(session: Session):
    """Make sure every ticker has a StockScore row."""
    from services.processing import _update_stock_scores
    _update_stock_scores(session)

    tickers = session.query(db.Ticker).all()
    for t in tickers:
        if not session.query(db.StockScore).filter_by(ticker=t.ticker).first():
            session.add(db.StockScore(
                ticker=t.ticker,
                opportunity_score=0.05,
                crowding_score=0.10,
                risk_score=0.05,
            ))
    session.commit()


def run_alerts_and_brief(session: Session):
    from services.alerts import run_alert_sweep
    run_alert_sweep(session)
    generate_daily_brief(session)


if __name__ == "__main__":
    from database import create_tables, SessionLocal
    create_tables()
    s = SessionLocal()
    seed(s)
    s.close()
    print("Seed complete.")
