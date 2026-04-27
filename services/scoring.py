"""
services/scoring.py
-------------------
Credibility Score and revised Opportunity Score formulas.

Phase 2+ composite formula (Upgrade Spec §5):
    OpportunityScore = b1·Exposure
                     + b2·Credibility
                     + b3·ExpectationGap
                     + b4·IndirectImpact
                     + b5·Lag
                     + b6·Asymmetry
                     - b7·Crowding
                     - b8·Risk

Design rule: keep every component visible in the interface.
The user should see why a stock ranks highly instead of trusting a black box.

All scores are clamped to [0.0, 1.0] for the opportunity score.
Individual components may be signed in [-1, 1] (expectation gap, indirect impact).
"""

from models import CredibilityFactors, OpportunityFactors


# ---------------------------------------------------------------------------
# Credibility weights (tunable via learning engine)
# ---------------------------------------------------------------------------

CREDIBILITY_WEIGHTS = {
    "source_quality":      0.30,
    "corroboration":       0.20,
    "primary_source":      0.20,
    "contradiction":      -0.15,
    "official_confirm":    0.15,
}

# ---------------------------------------------------------------------------
# Opportunity weights — revised composite (Upgrade Spec §5)
# ---------------------------------------------------------------------------
# These are the base (un-tuned) weights.  The adaptive learning engine in
# services/learning.py will maintain regime-specific versions that are used
# at runtime once sufficient outcome data exists.

OPPORTUNITY_WEIGHTS = {
    # Academic basis: Gu/Kelly/Xiu (2020) — hybrid theory+ML models benefit from
    # explicit economic structure.  Crowding and risk penalties are tightened
    # per Novy-Marx/Velikov evidence that anomaly returns weaken sharply under
    # crowded, high-cost conditions (and per professor feedback to tighten metrics).
    "exposure":          0.15,   # b1 — causal pathway strength to this ticker
    "credibility":       0.20,   # b2 — credibility-weighted impact
    "expectation_gap":   0.18,   # b3 — surprise vs consensus (reduced slightly)
    "indirect_impact":   0.10,   # b4 — second-order transmission
    "lag":               0.15,   # b5 — price-reaction underreaction proxy
    "asymmetry":         0.10,   # b6 — upside vs downside asymmetry
    "crowding":         -0.20,   # b7 penalty (raised: -0.15 → -0.20; crowding
                                 #             destroys returns per the literature)
    "risk":             -0.08,   # b8 penalty (raised: -0.05 → -0.08; risk
                                 #             control is non-decorative per doc)
    # Note: weights sum to 0.60 positive + (-0.28) negative = net +0.32 at full
    # exposure.  The narrative_stage multiplier (0.15–1.0) then scales the final
    # score so early-stage events rank far above late-stage events at equal raw score.
}

# Narrative stage multipliers (early > late)
NARRATIVE_STAGE_MULTIPLIERS = {
    "emerging":   1.0,
    "developing": 0.75,
    "peak":       0.40,
    "declining":  0.15,
}

# Source quality presets
SOURCE_QUALITY_MAP = {
    "reuters":               0.95,
    "bloomberg":             0.95,
    "ap":                    0.90,
    "bbc":                   0.85,
    "financial times":       0.88,
    "wall street journal":   0.88,
    "the guardian":          0.82,
    "al jazeera":            0.80,
    "sec":                   1.00,
    "federal reserve":       1.00,
    "whitehouse":            1.00,
    "dod":                   0.95,
    "defense news":          0.80,
    "politico":              0.78,
    "cnbc":                  0.75,
    "marketwatch":           0.72,
    "twitter":               0.25,
    "reddit":                0.20,
    "unknown":               0.40,
}


def get_source_quality(source: str) -> float:
    """Return a quality weight [0,1] for the given source string."""
    src_lower = source.lower()
    for key, val in SOURCE_QUALITY_MAP.items():
        if key in src_lower:
            return val
    return SOURCE_QUALITY_MAP["unknown"]


def calculate_credibility(factors: CredibilityFactors) -> float:
    """
    Credibility = weighted sum of:
      - source_quality           (0–1)
      - corroboration_count      (normalised, capped at 10)
      - primary_source_bonus
      - contradiction_penalty    (subtracted)
      - official_confirmation    (bonus)

    Returns float in [0.0, 1.0].
    """
    w = CREDIBILITY_WEIGHTS

    # Normalise corroboration: 1 article → 0.1, 10+ articles → 1.0
    corroboration_norm = min(factors.corroboration_count / 10.0, 1.0)

    score = (
        w["source_quality"]   * factors.source_quality
        + w["corroboration"]  * corroboration_norm
        + w["primary_source"] * (1.0 if factors.is_primary_source else 0.0)
        + w["contradiction"]  * factors.contradiction_penalty
        + w["official_confirm"] * (1.0 if factors.official_confirmation else 0.0)
    )
    return max(0.0, min(1.0, score))


def calculate_opportunity(
    factors: OpportunityFactors,
    weights: dict = None,
) -> float:
    """
    Revised Opportunity Score (Upgrade Spec §5):

        OpportunityScore = b1·Exposure
                         + b2·Credibility
                         + b3·ExpectationGap    (signed, converted to contribution)
                         + b4·IndirectImpact    (signed, converted to contribution)
                         + b5·Lag
                         + b6·Asymmetry         (signed, converted to contribution)
                         - b7·Crowding
                         - b8·Risk

    Signed components (expectation_gap, indirect_impact, asymmetry) are mapped
    from [-1,1] to [0,1] via (x+1)/2 before weighting, so a strongly positive
    gap contributes positively and a strongly negative gap reduces the score.

    The narrative_stage multiplier scales the entire score as a final modifier:
    an emerging story at the same raw score is worth more than a crowded peak story.

    Returns float in [0.0, 1.0].
    """
    w = weights or OPPORTUNITY_WEIGHTS

    stage_mult = NARRATIVE_STAGE_MULTIPLIERS.get(
        factors.narrative_stage.lower(), 0.5
    )

    # Convert signed factors to [0,1] contributions
    def _signed_to_contribution(val: float) -> float:
        return (max(-1.0, min(1.0, val)) + 1.0) / 2.0

    gap_contrib      = _signed_to_contribution(factors.expectation_gap)
    indirect_contrib = _signed_to_contribution(factors.indirect_impact)
    asym_contrib     = _signed_to_contribution(factors.asymmetry)

    raw = (
        w.get("exposure", 0.15)          * factors.exposure
        + w.get("credibility", 0.20)     * factors.credibility
        + w.get("expectation_gap", 0.18) * gap_contrib
        + w.get("indirect_impact", 0.10) * indirect_contrib
        + w.get("lag", 0.15)             * factors.price_reaction_lag
        + w.get("asymmetry", 0.10)       * asym_contrib
        + w.get("crowding", -0.20)       * factors.crowding   # negative weight
        + w.get("risk", -0.08)           * factors.risk        # negative weight
    )

    # Apply narrative stage multiplier (caps how high the score can be for late-stage)
    # Clamp to [0, 1] first to avoid negative raw scores from killing the signal entirely
    raw_clamped = max(0.0, min(1.0, raw))
    score = raw_clamped * stage_mult

    return round(max(0.0, min(1.0, score)), 4)


def calculate_risk(impact_score: float, credibility: float, crowding: float) -> float:
    """
    Composite risk score.
    High negative impact + high credibility + high crowding → high risk.
    """
    directional_risk = abs(impact_score)
    score = (directional_risk * 0.5) + (credibility * 0.3) + (crowding * 0.2)
    return max(0.0, min(1.0, score))


def recalculate_event_credibility(articles: list) -> float:
    """
    Given a list of Article ORM objects attached to an event,
    derive the event-level credibility score.
    """
    if not articles:
        return 0.0

    primary_count    = sum(1 for a in articles if a.is_primary_source)
    official_confirm = any(a.has_official_confirm for a in articles)
    contradictions   = sum(1 for a in articles if a.contradiction_flag)

    factors = CredibilityFactors(
        source_quality=max(get_source_quality(a.source) for a in articles),
        corroboration_count=len(articles),
        is_primary_source=(primary_count > 0),
        contradiction_penalty=min(contradictions / max(len(articles), 1), 1.0),
        official_confirmation=official_confirm,
    )
    return calculate_credibility(factors)


def explain_opportunity_score(factors: OpportunityFactors, weights: dict = None) -> dict:
    """
    Return a breakdown of each component's contribution to the opportunity score.
    Used by the dashboard to show 'why this stock ranks here'.

    Returns a dict mapping component name → contribution value.
    """
    w = weights or OPPORTUNITY_WEIGHTS

    def _s(val): return (max(-1.0, min(1.0, val)) + 1.0) / 2.0

    stage_mult = NARRATIVE_STAGE_MULTIPLIERS.get(factors.narrative_stage.lower(), 0.5)

    return {
        "exposure":          round(w.get("exposure", 0.15) * factors.exposure, 4),
        "credibility":       round(w.get("credibility", 0.20) * factors.credibility, 4),
        "expectation_gap":   round(w.get("expectation_gap", 0.18) * _s(factors.expectation_gap), 4),
        "indirect_impact":   round(w.get("indirect_impact", 0.10) * _s(factors.indirect_impact), 4),
        "lag":               round(w.get("lag", 0.15) * factors.price_reaction_lag, 4),
        "asymmetry":         round(w.get("asymmetry", 0.10) * _s(factors.asymmetry), 4),
        "crowding_penalty":  round(w.get("crowding", -0.20) * factors.crowding, 4),
        "risk_penalty":      round(w.get("risk", -0.08) * factors.risk, 4),
        "narrative_stage_multiplier": stage_mult,
    }
