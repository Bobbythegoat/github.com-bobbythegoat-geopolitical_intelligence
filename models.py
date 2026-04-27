"""
models.py — Pydantic request/response schemas
Phase 2+ additions:
  - ExpectationGapOut     : Expectation Gap Engine output
  - NarrativeInflectionOut: Narrative inflection metrics
  - SecondOrderImpactOut  : Indirect impact mapping
  - AlertOutcomeIn/Out    : Post-alert outcome recording
  - FactorWeightOut       : Adaptive weight inspection
  - OutcomeReviewIn       : User outcome review submission
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------

class ArticleBase(BaseModel):
    source: str
    url: Optional[str] = None
    headline: str
    content: str
    is_primary_source: int = 0
    has_official_confirm: int = 0

class ArticleOut(ArticleBase):
    id: int
    timestamp: datetime
    event_id: Optional[int] = None
    contradiction_flag: int = 0

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventBase(BaseModel):
    title: str
    narrative_stage: str = "emerging"
    event_type: Optional[str] = "sector"
    summary: Optional[str] = None

class EventCreate(EventBase):
    pass

class EventOut(EventBase):
    event_id: int
    timestamp: datetime
    credibility_score: float
    event_type: Optional[str] = "sector"
    source_count: Optional[int] = 1
    geography_tags: Optional[str] = None
    articles: List[ArticleOut] = []

    model_config = {"from_attributes": True}

class EventSummary(BaseModel):
    """Lightweight event listing (no articles)."""
    event_id: int
    title: str
    timestamp: datetime
    credibility_score: float
    narrative_stage: str
    event_type: Optional[str] = "sector"
    source_count: Optional[int] = 1
    geography_tags: Optional[str] = None
    summary: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

class TickerBase(BaseModel):
    ticker: str
    company_name: str
    sector: str

class TickerOut(TickerBase):
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Event–Ticker Impact
# ---------------------------------------------------------------------------

class ImpactOut(BaseModel):
    ticker: str
    impact_score: float
    company_name: Optional[str] = None
    sector: Optional[str] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Stock Scores
# ---------------------------------------------------------------------------

class StockScoreOut(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    opportunity_score: float
    crowding_score: float
    risk_score: float
    exposure_score: Optional[float] = 0.0
    impact_score: Optional[float] = 0.0
    narrative_score: Optional[float] = 0.0
    lag_score: Optional[float] = 0.0
    asymmetry_score: Optional[float] = 0.0
    expectation_gap_score: Optional[float] = 0.0    # Phase 2+
    indirect_impact_score: Optional[float] = 0.0    # Phase 2+
    decision_bucket: Optional[str] = "Watch"
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class AlertOut(BaseModel):
    id: int
    event_id: int
    tier: str
    message: str
    dismissed: int
    timestamp: datetime
    event_title: Optional[str] = None
    horizon: Optional[str] = "short_swing"              # Phase 2+
    expectation_gap_score: Optional[float] = 0.0        # Phase 2+
    confidence_score: Optional[float] = 0.5             # Phase 2+
    regime_label: Optional[str] = None                  # Phase 2+
    ml_predicted_outcome: Optional[str] = None          # ML inference
    ml_predicted_direction: Optional[str] = None        # ML inference
    ml_confidence: Optional[float] = 0.0               # ML inference

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Daily Brief
# ---------------------------------------------------------------------------

class DailyBriefOut(BaseModel):
    id: int
    summary_text: str
    timestamp: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackIn(BaseModel):
    event_id: Optional[int] = None
    ticker: Optional[str] = None
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class FeedbackOut(FeedbackIn):
    id: int
    timestamp: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Scoring internals (used by services)
# ---------------------------------------------------------------------------

class CredibilityFactors(BaseModel):
    source_quality: float = Field(0.5, ge=0, le=1)
    corroboration_count: int = 1
    is_primary_source: bool = False
    contradiction_penalty: float = Field(0.0, ge=0, le=1)
    official_confirmation: bool = False

class OpportunityFactors(BaseModel):
    exposure: float = Field(0.5, ge=0, le=1)
    credibility: float = Field(0.5, ge=0, le=1)
    narrative_stage: str = "emerging"       # emerging > developing > peak > declining
    crowding: float = Field(0.0, ge=0, le=1)
    price_reaction_lag: float = Field(0.5, ge=0, le=1)
    risk: float = Field(0.3, ge=0, le=1)
    # Phase 2+ additions
    expectation_gap: float = Field(0.0, ge=-1, le=1)   # signed: positive = bullish surprise
    indirect_impact: float = Field(0.0, ge=-1, le=1)   # signed: second-order net effect
    asymmetry: float = Field(0.0, ge=-1, le=1)         # upside vs downside asymmetry


# ---------------------------------------------------------------------------
# Stock Fundamentals (live data via yfinance)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2+: Expectation Gap
# ---------------------------------------------------------------------------

class ExpectationGapOut(BaseModel):
    event_id: int
    ticker: Optional[str] = None
    gap_score: float                          # signed: positive = bullish surprise
    fundamental_surprise: Optional[float] = None
    narrative_shift: Optional[float] = None
    pre_event_drift_adj: Optional[float] = None
    implied_move_residual: Optional[float] = None
    interpretation: str = ""

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Phase 2+: Narrative Inflection
# ---------------------------------------------------------------------------

class NarrativeInflectionOut(BaseModel):
    event_id: int
    attention_velocity: float
    price_response: Optional[float] = None
    contradiction_rate: float
    inflection_score: float
    signal: str   # 'accumulation' | 'building' | 'neutral' | 'peaking' | 'exhausting'

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Phase 2+: Second-Order Impact
# ---------------------------------------------------------------------------

class IndirectImpactItem(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    indirect_impact_score: float
    relationship_type: str
    from_entity: str
    confidence: float
    time_horizon_days: int

class SecondOrderImpactOut(BaseModel):
    event_id: int
    direct_impacts: List[str] = []
    indirect_impacts: List[IndirectImpactItem] = []


# ---------------------------------------------------------------------------
# Phase 2+: Alert Outcomes
# ---------------------------------------------------------------------------

class AlertOutcomeIn(BaseModel):
    alert_id: int
    ticker: str
    forward_return_15m: Optional[float] = None
    forward_return_1h: Optional[float] = None
    forward_return_1d: Optional[float] = None
    forward_return_3d: Optional[float] = None
    forward_return_1w: Optional[float] = None
    forward_return_1m: Optional[float] = None
    max_favorable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    realized_volatility: Optional[float] = None
    realized_sharpe_proxy: Optional[float] = None
    outcome_label: Optional[str] = "pending"

class AlertOutcomeOut(AlertOutcomeIn):
    id: int
    timestamp: datetime
    user_override_label: Optional[str] = None
    user_comment: Optional[str] = None
    reviewed: int = 0

    model_config = {"from_attributes": True}


class OutcomeReviewIn(BaseModel):
    outcome_id: int
    user_override_label: str   # profitable | unprofitable | early | late | neutral | invalidated
    user_comment: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase 2+: Factor Weights (adaptive learning inspection)
# ---------------------------------------------------------------------------

class FactorWeightOut(BaseModel):
    id: int
    factor_name: str
    weight_value: float
    regime_label: str
    hit_rate: Optional[float] = None
    sample_count: int
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Phase 2+: Regime
# ---------------------------------------------------------------------------

class RegimeOut(BaseModel):
    id: int
    label: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    is_active: int

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Stock Fundamentals (live data via yfinance)
# ---------------------------------------------------------------------------

class StockFundamentalsOut(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    # Price & valuation
    current_price: Optional[float] = None
    price_change_1d_pct: Optional[float] = None
    price_change_5d_pct: Optional[float] = None
    price_change_1m_pct: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    week_52_pct_from_high: Optional[float] = None  # how far from 52-week high (negative = below)

    # Valuation multiples
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None

    # Momentum & technicals
    rsi_14: Optional[float] = None             # 14-day RSI
    above_sma_50: Optional[bool] = None        # price > 50-day SMA
    above_sma_200: Optional[bool] = None       # price > 200-day SMA
    volume_ratio: Optional[float] = None       # today vol / 30d avg vol

    # Quality / growth
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    beta: Optional[float] = None

    # Analyst consensus
    analyst_target_price: Optional[float] = None
    analyst_upside_pct: Optional[float] = None
    analyst_recommendation: Optional[str] = None  # buy / hold / sell

    # Geopolitical score from our system
    opportunity_score: Optional[float] = None
    risk_score: Optional[float] = None
    crowding_score: Optional[float] = None

    # Historical financials (quarterly)
    company_description: Optional[str] = None
    revenue_history: Optional[List[Dict[str, Any]]] = None      # [{quarter, revenue}]
    earnings_history: Optional[List[Dict[str, Any]]] = None     # [{quarter, net_income}]
    gross_margin_history: Optional[List[Dict[str, Any]]] = None # [{quarter, gross_margin_pct}]
    analyst_count: Optional[int] = None

    fetched_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Discovered Tickers
# ---------------------------------------------------------------------------

class DiscoveredTickerOut(BaseModel):
    id:                   int
    ticker:               str
    company_name:         Optional[str]
    sector:               Optional[str]
    market_cap:           Optional[float]
    first_seen_at:        Optional[datetime]
    discovery_event_id:   Optional[int]
    confidence_score:     float
    article_count:        int
    is_promoted:          int

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# ML Model Info & Predictions
# ---------------------------------------------------------------------------

class MLModelInfo(BaseModel):
    model_available:          bool
    trained_at:               Optional[str]
    n_training_samples:       int
    outcome_cv_accuracy:      Optional[float]
    direction_cv_accuracy:    Optional[float]
    is_bootstrap_model:       bool = False
    total_outcome_records:    int
    labeled_outcome_records:  int
    samples_until_training:   int
    ready_to_train:           bool


class MLPredictionOut(BaseModel):
    outcome:             Dict[str, float]   # {"profitable": p, "neutral": p, "unprofitable": p}
    direction:           Dict[str, float]   # {"up": p, "flat": p, "down": p}
    predicted_outcome:   str
    predicted_direction: str
    ml_confidence:       float
    model_available:     bool
