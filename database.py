"""
database.py — SQLAlchemy models & session management
Implements all core tables including Phase 2+ upgrade schema:
  - AlertOutcome  : post-alert forward-return and excursion logging
  - FactorWeight  : adaptive model weights per factor per regime
  - RegimeState   : current and historical market regime labels
  - RelationshipEdge : second-order transmission graph edges
  - NarrativeState : per-event inflection metrics over time
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship
from datetime import datetime, timezone

DATABASE_URL = "sqlite:///./geoint.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

from sqlalchemy import event as _sa_event

@_sa_event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Event(Base):
    """
    A clustered event derived from one or more articles.
    narrative_stage: 'emerging' | 'developing' | 'peak' | 'declining'
    event_type: 'geopolitical' | 'macro' | 'company' | 'sector'

    Phase 2+ additions:
      expectation_proxy      — estimated market expectation before the event
      narrative_shift_score  — how much the narrative changed vs prior state
      contradiction_rate     — fraction of articles contradicting the cluster
      source_breadth         — number of distinct sources covering the event
      narrative_inflection   — turning-point score (positive = building,
                               negative = exhausting)
      attention_velocity     — rate of mention growth (Mentions_t / Mentions_{t-1})
      price_response         — normalised absolute return relative to realised vol
    """
    __tablename__ = "events"

    event_id              = Column(Integer, primary_key=True, index=True)
    title                 = Column(String(512), nullable=False)
    timestamp             = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    credibility_score     = Column(Float, default=0.0)
    narrative_stage       = Column(String(32), default="emerging")
    event_type            = Column(String(32), default="sector")
    source_count          = Column(Integer, default=1)
    geography_tags        = Column(String(512), nullable=True)
    summary               = Column(Text, nullable=True)
    # Phase 2+ expectation & inflection fields
    expectation_proxy     = Column(Float, nullable=True)          # market consensus proxy
    narrative_shift_score = Column(Float, default=0.0)            # delta vs prior stage
    contradiction_rate    = Column(Float, default=0.0)            # [0,1]
    source_breadth        = Column(Integer, default=1)            # distinct source count
    narrative_inflection  = Column(Float, default=0.0)            # turning-point score
    attention_velocity    = Column(Float, default=1.0)            # Mentions_t / Mentions_{t-1}
    price_response        = Column(Float, nullable=True)          # |Return_t| / RealVol_t

    articles = relationship("Article", back_populates="event", cascade="all, delete")
    impacts  = relationship("EventTickerImpact", back_populates="event", cascade="all, delete")
    alerts   = relationship("Alert", back_populates="event", cascade="all, delete")


class Article(Base):
    """
    Raw ingested article that may be linked to a clustered Event.
    """
    __tablename__ = "articles"

    id        = Column(Integer, primary_key=True, index=True)
    source    = Column(String(256))
    url       = Column(String(1024), nullable=True)
    headline  = Column(String(512))
    content   = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    event_id  = Column(Integer, ForeignKey("events.event_id"), nullable=True)
    # Processing metadata
    is_primary_source    = Column(Integer, default=0)   # 1 = primary
    has_official_confirm = Column(Integer, default=0)   # 1 = confirmed
    contradiction_flag   = Column(Integer, default=0)   # 1 = contradicts cluster

    event = relationship("Event", back_populates="articles")


class Ticker(Base):
    """
    Master list of tracked stocks/instruments.
    """
    __tablename__ = "tickers"

    ticker       = Column(String(16), primary_key=True)
    company_name = Column(String(256))
    sector       = Column(String(128))

    impacts = relationship("EventTickerImpact", back_populates="ticker_obj")
    scores  = relationship("StockScore", back_populates="ticker_obj", uselist=False)


class EventTickerImpact(Base):
    """
    Maps a geopolitical event to the tickers it affects and by how much.
    impact_score in [-1, 1]: negative = bearish, positive = bullish
    """
    __tablename__ = "event_ticker_impacts"

    id           = Column(Integer, primary_key=True, index=True)
    event_id     = Column(Integer, ForeignKey("events.event_id"))
    ticker       = Column(String(16), ForeignKey("tickers.ticker"))
    impact_score = Column(Float, default=0.0)

    event      = relationship("Event", back_populates="impacts")
    ticker_obj = relationship("Ticker", back_populates="impacts")


class StockScore(Base):
    """
    Composite scorecard for each tracked ticker.
    All scores in [0, 1] unless noted.

    Phase 2+ additions:
      expectation_gap_score  — surprise relative to prior consensus
      indirect_impact_score  — second-order transmission benefit/damage
    """
    __tablename__ = "stock_scores"

    id                    = Column(Integer, primary_key=True, index=True)
    ticker                = Column(String(16), ForeignKey("tickers.ticker"), unique=True)
    opportunity_score     = Column(Float, default=0.0)
    crowding_score        = Column(Float, default=0.0)
    risk_score            = Column(Float, default=0.0)
    exposure_score        = Column(Float, default=0.0)
    impact_score          = Column(Float, default=0.0)
    narrative_score       = Column(Float, default=0.0)
    lag_score             = Column(Float, default=0.0)
    asymmetry_score       = Column(Float, default=0.0)
    expectation_gap_score = Column(Float, default=0.0)   # Phase 2+
    indirect_impact_score = Column(Float, default=0.0)   # Phase 2+
    decision_bucket       = Column(String(64), default="Watch")
    updated_at            = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    ticker_obj = relationship("Ticker", back_populates="scores")


class Alert(Base):
    """
    Triggered notification when credibility/relevance thresholds are exceeded.
    tier: 'Critical' | 'High' | 'Monitor'

    Phase 2+ additions:
      horizon                 — intended signal time frame
      expectation_gap_score   — gap at alert time
      feature_vector_snapshot — JSON snapshot of all input features
      component_scores_snapshot — JSON snapshot of sub-scores
      confidence_score        — calibrated confidence at issuance
      regime_label            — market regime at issuance
    """
    __tablename__ = "alerts"

    id                        = Column(Integer, primary_key=True, index=True)
    event_id                  = Column(Integer, ForeignKey("events.event_id"))
    tier                      = Column(String(16))
    message                   = Column(Text)
    dismissed                 = Column(Integer, default=0)
    timestamp                 = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # Phase 2+ fields
    horizon                   = Column(String(32), default="short_swing")   # intraday | short_swing | structural
    expectation_gap_score     = Column(Float, default=0.0)
    feature_vector_snapshot   = Column(Text, nullable=True)                 # JSON
    component_scores_snapshot = Column(Text, nullable=True)                 # JSON
    confidence_score          = Column(Float, default=0.5)
    regime_label              = Column(String(64), nullable=True)
    # ML inference fields (populated at alert creation time)
    ml_predicted_outcome      = Column(String(32), nullable=True)           # profitable | neutral | unprofitable
    ml_predicted_direction    = Column(String(16), nullable=True)           # up | flat | down
    ml_confidence             = Column(Float, default=0.0)

    event    = relationship("Event", back_populates="alerts")
    outcomes = relationship("AlertOutcome", back_populates="alert", cascade="all, delete")


class DailyBrief(Base):
    """
    Auto-generated end-of-day narrative summary.
    """
    __tablename__ = "daily_briefs"

    id           = Column(Integer, primary_key=True, index=True)
    summary_text = Column(Text)
    timestamp    = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class FeedbackEntry(Base):
    """
    User feedback on event quality (drives scoring weight refinement).
    """
    __tablename__ = "feedback"

    id        = Column(Integer, primary_key=True, index=True)
    event_id  = Column(Integer, ForeignKey("events.event_id"), nullable=True)
    ticker    = Column(String(16), nullable=True)
    rating    = Column(Integer)          # 1–5
    comment   = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Phase 2+ Tables
# ---------------------------------------------------------------------------

class AlertOutcome(Base):
    """
    Post-alert forward-return and excursion tracking.
    Populated asynchronously as time passes after an alert is fired.
    outcome_label: 'profitable' | 'unprofitable' | 'early' | 'late' |
                   'neutral' | 'invalidated' | 'pending'
    """
    __tablename__ = "alert_outcomes"

    id                    = Column(Integer, primary_key=True, index=True)
    alert_id              = Column(Integer, ForeignKey("alerts.id"))
    ticker                = Column(String(16))
    event_id              = Column(Integer, ForeignKey("events.event_id"), nullable=True)
    timestamp             = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    # Forward returns at multiple horizons
    forward_return_15m    = Column(Float, nullable=True)
    forward_return_1h     = Column(Float, nullable=True)
    forward_return_1d     = Column(Float, nullable=True)
    forward_return_3d     = Column(Float, nullable=True)
    forward_return_1w     = Column(Float, nullable=True)
    forward_return_1m     = Column(Float, nullable=True)
    # Risk-adjusted metrics
    max_favorable_excursion = Column(Float, nullable=True)
    max_adverse_excursion   = Column(Float, nullable=True)
    realized_volatility     = Column(Float, nullable=True)
    realized_sharpe_proxy   = Column(Float, nullable=True)
    # Classification
    outcome_label           = Column(String(32), default="pending")
    user_override_label     = Column(String(32), nullable=True)
    user_comment            = Column(Text, nullable=True)
    reviewed                = Column(Integer, default=0)   # 0=no, 1=yes

    alert = relationship("Alert", back_populates="outcomes")


class FactorWeight(Base):
    """
    Adaptive model weights for each scoring factor, per regime.
    Updated via the learning service using the online weight update rule:
      w_{t+1} = (1 - λ)·w_t + η·(y_t - ŷ_t)·x_t
    """
    __tablename__ = "factor_weights"

    id           = Column(Integer, primary_key=True, index=True)
    factor_name  = Column(String(64), nullable=False)
    weight_value = Column(Float, default=0.0)
    regime_label = Column(String(64), default="base")   # base | risk_on | risk_off | ...
    hit_rate     = Column(Float, nullable=True)          # calibrated hit rate for this factor
    sample_count = Column(Integer, default=0)
    updated_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RegimeState(Base):
    """
    Market regime tracking.
    regime_label: 'base' | 'risk_on' | 'risk_off' | 'rate_sensitive' |
                  'war_escalation' | 'earnings_season' | 'policy_shock'
    """
    __tablename__ = "regime_states"

    id          = Column(Integer, primary_key=True, index=True)
    label       = Column(String(64), nullable=False)
    started_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at    = Column(DateTime, nullable=True)
    is_active   = Column(Integer, default=1)   # 1 = current regime


class RelationshipEdge(Base):
    """
    Directed relationship edge for the second-order transmission graph.
    from_entity → to_entity with a weight representing historical
    transmission strength.

    Examples:
      TSMC → NVDA (supplier → customer)
      Iran oil sanctions → XOM (policy → stock)
      Taiwan conflict → AAPL (geography → supply-chain dependent)
    """
    __tablename__ = "relationship_edges"

    id               = Column(Integer, primary_key=True, index=True)
    from_entity      = Column(String(128), nullable=False, index=True)
    to_entity        = Column(String(128), nullable=False, index=True)
    relationship_type = Column(String(64))   # supplier | customer | competitor | policy | geography | sector
    weight           = Column(Float, default=0.5)    # historical transmission strength [0,1]
    time_decay_days  = Column(Integer, default=5)    # how fast the effect decays
    direction        = Column(String(8), default="positive")  # positive | negative | uncertain
    updated_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Commodity(Base):
    """
    Strategic commodity / industrial input being tracked.

    Tiers (from product spec):
      tier 1 — high importance, easy to track (copper, nat gas, uranium,
               aluminum, lithium, nickel) — LME data, COT positioning,
               liquid futures.
      tier 2 — strategic choke points (gallium, germanium, indium, rare
               earths, graphite) — policy-sensitive, processing concentration.
      tier 3 — semiconductor hidden bottlenecks (neon, argon, krypton,
               xenon, photoresists, etchants, CMP slurries, ultra-pure
               water) — tracked via supplier earnings, fab capex,
               geopolitical disruption.

    `exposed_tickers` is a comma-separated list of tickers that have material
    revenue/cost exposure to this commodity. RelationshipEdge rows are seeded
    from this list so commodity shocks propagate through the existing
    second-order transmission engine.
    """
    __tablename__ = "commodities"

    id                       = Column(Integer, primary_key=True, index=True)
    symbol                   = Column(String(32), unique=True, nullable=False, index=True)
    name                     = Column(String(128), nullable=False)
    tier                     = Column(Integer, nullable=False)        # 1 | 2 | 3
    category                 = Column(String(64))                     # metal | energy | gas | chemical | rare_earth
    unit                     = Column(String(32))                     # USD/lb, USD/mmBtu, USD/kg ...
    proxy_ticker             = Column(String(16), nullable=True)      # ETF/futures ETF proxy
    supplier_concentration   = Column(Float, default=0.0)             # HHI-like [0,1]; 1 = single-country choke
    policy_sensitivity       = Column(Float, default=0.0)             # [0,1]; export-control exposure
    ai_demand_linkage        = Column(Float, default=0.0)             # [0,1]; how tightly tied to AI/data-center scaling
    primary_geographies      = Column(String(512))                    # CSV: Chile,Indonesia,China,DRC,Russia ...
    exposed_tickers          = Column(String(1024))                   # CSV of tickers with material exposure
    keywords                 = Column(Text)                           # CSV / JSON list of detection keywords
    notes                    = Column(Text)
    updated_at               = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                                                 onupdate=lambda: datetime.now(timezone.utc))


class CommoditySignal(Base):
    """
    A detected commodity-relevant signal extracted from an article/event.

    signal_type:
      supply_disruption | export_control | stockpiling | capacity_expansion
      | price_move | policy_action | geopolitical_risk | demand_shock
    direction: +1 (bullish for commodity price), -1 (bearish), 0 (uncertain)
    severity:  [0,1] — how material the signal is
    """
    __tablename__ = "commodity_signals"

    id              = Column(Integer, primary_key=True, index=True)
    commodity_symbol= Column(String(32), ForeignKey("commodities.symbol"), index=True)
    event_id        = Column(Integer, ForeignKey("events.event_id"), nullable=True, index=True)
    article_id      = Column(Integer, ForeignKey("articles.id"), nullable=True)
    signal_type     = Column(String(48), nullable=False)
    direction       = Column(Integer, default=0)       # +1 / -1 / 0
    severity        = Column(Float, default=0.0)       # [0,1]
    confidence      = Column(Float, default=0.5)       # [0,1]
    summary         = Column(Text)
    timestamp       = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class DiscoveredTicker(Base):
    """
    Tickers discovered dynamically from article text (beyond hardcoded universe).
    High-confidence discoveries are auto-promoted to the main Ticker table.
    """
    __tablename__ = "discovered_tickers"

    id               = Column(Integer, primary_key=True)
    ticker           = Column(String(16), unique=True, nullable=False, index=True)
    company_name     = Column(String(256))
    sector           = Column(String(128))
    market_cap       = Column(Float)
    first_seen_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    discovery_event_id = Column(Integer, ForeignKey("events.event_id"))
    confidence_score = Column(Float, default=0.0)
    article_count    = Column(Integer, default=1)
    is_promoted      = Column(Integer, default=0)   # 1 = added to main Ticker table
    notes            = Column(Text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db():
    """FastAPI dependency — yields a DB session then closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables and performance indexes (idempotent)."""
    from sqlalchemy import text
    Base.metadata.create_all(bind=engine)

    # Performance indexes for high-frequency queries
    _indexes = [
        "CREATE INDEX IF NOT EXISTS idx_article_timestamp    ON articles(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_article_event_id     ON articles(event_id)",
        "CREATE INDEX IF NOT EXISTS idx_event_timestamp      ON events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_event_credibility    ON events(credibility_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_alert_timestamp      ON alerts(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_alert_event_id       ON alerts(event_id)",
        "CREATE INDEX IF NOT EXISTS idx_stock_opportunity    ON stock_scores(opportunity_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_outcome_alert_id     ON alert_outcomes(alert_id)",
        "CREATE INDEX IF NOT EXISTS idx_outcome_label        ON alert_outcomes(outcome_label)",
        "CREATE INDEX IF NOT EXISTS idx_regime_active        ON regime_states(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_disc_confidence      ON discovered_tickers(confidence_score DESC)",
    ]
    with engine.connect() as conn:
        for ddl in _indexes:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception:
                pass  # index may already exist or table not yet created
