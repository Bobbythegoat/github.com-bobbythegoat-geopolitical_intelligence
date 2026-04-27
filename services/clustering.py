"""
services/clustering.py
----------------------
Groups similar articles into unified Event clusters (Blueprint §3.3, Phase 2).

Strategy:
  - Represent each article as a TF-IDF vector over its headline + content.
  - Compute cosine similarity between incoming article and existing event centroids.
  - If similarity >= SIMILARITY_THRESHOLD → assign to that event.
  - Otherwise → create a new event.

Phase 3 hook: narrative_stage detection based on article volume over time.
"""

import re
import math
from collections import defaultdict, Counter
from typing import List, Optional

from sqlalchemy.orm import Session

import database as db
from services.scoring import recalculate_event_credibility
from services import semantic_clustering

SIMILARITY_THRESHOLD = 0.30   # tunable
MAX_VOCAB_SIZE       = 5_000


# ---------------------------------------------------------------------------
# Basic TF-IDF helpers (no external NLP dependency)
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","has","have","had","will","would",
    "can","could","should","may","might","that","this","these","those",
    "it","its","from","by","as","into","than","then","so","also","said",
    "says","new","s","t","he","she","they","we","you","i","us","our","their",
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def tfidf_vector(tokens: List[str], idf: dict) -> dict:
    """Return a TF-IDF sparse vector (dict: term → score)."""
    tf = Counter(tokens)
    total = max(len(tokens), 1)
    vec = {}
    for term, count in tf.items():
        if term in idf:
            vec[term] = (count / total) * idf[term]
    return vec


def cosine_similarity(v1: dict, v2: dict) -> float:
    if not v1 or not v2:
        return 0.0
    dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in v1)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def build_idf(corpus: List[List[str]]) -> dict:
    """Build IDF table from a list of token lists."""
    N = len(corpus)
    df: Counter = Counter()
    for tokens in corpus:
        df.update(set(tokens))
    return {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}


# ---------------------------------------------------------------------------
# Narrative stage detection (Phase 3)
# ---------------------------------------------------------------------------

def detect_narrative_stage(article_count: int, recent_count: int) -> str:
    """
    Heuristic: compare recent article volume to total cluster size.
    recent_count = articles added in last 6 hours.
    """
    if article_count <= 2:
        return "emerging"
    ratio = recent_count / max(article_count, 1)
    if ratio > 0.6:
        return "developing"
    if ratio > 0.3:
        return "peak"
    return "declining"


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def cluster_article(article_id: int, session: Session) -> Optional[int]:
    """
    Assign article_id to an existing event or create a new one.
    Returns the event_id the article was assigned to.
    """
    article = session.get(db.Article, article_id)
    if article is None:
        return None

    # Already clustered
    if article.event_id is not None:
        return article.event_id

    # Build corpus from all articles that are already assigned to events
    existing_articles = (
        session.query(db.Article)
        .filter(db.Article.event_id.isnot(None))
        .all()
    )

    if not existing_articles:
        # First article — create new event
        return _create_event_for(article, session)

    # Try semantic matching FIRST
    article_text = f"{article.headline} {article.content or ''}"
    active_events = (
        session.query(db.Event)
        .filter(db.Event.narrative_stage != "stale")
        .filter(db.Event.event_id.isnot(None))
        .all()
    )

    if active_events:
        try:
            event_titles = [e.title or "" for e in active_events]
            event_ids    = [e.event_id for e in active_events]
            # find_best_match returns a TUPLE (event_id, score) — unpack correctly
            sem_event_id, sem_score = semantic_clustering.find_best_match(
                article_text, event_titles, event_ids
            )
            if sem_event_id is not None:
                article.event_id = sem_event_id   # assign the int, not the tuple
                event = session.get(db.Event, sem_event_id)
                if event:
                    event.credibility_score = recalculate_event_credibility(
                        [a for a in existing_articles if a.event_id == sem_event_id] + [article]
                    )
                    from datetime import datetime, timedelta
                    recent_cutoff  = datetime.utcnow() - timedelta(hours=6)
                    cluster_arts   = [a for a in existing_articles if a.event_id == sem_event_id] + [article]
                    recent         = [a for a in cluster_arts if a.timestamp and a.timestamp >= recent_cutoff]
                    event.narrative_stage = detect_narrative_stage(len(cluster_arts), len(recent))
                session.commit()
                return sem_event_id
        except Exception:
            # Fall through to TF-IDF if semantic matching unavailable or raises
            pass

    # Gather tokens
    corpus_tokens = [
        tokenize(f"{a.headline} {a.content}") for a in existing_articles
    ]
    idf = build_idf(corpus_tokens)

    new_tokens = tokenize(f"{article.headline} {article.content}")
    new_vec    = tfidf_vector(new_tokens, idf)

    # Build centroid vectors per event
    event_articles: dict = defaultdict(list)
    for a, tokens in zip(existing_articles, corpus_tokens):
        event_articles[a.event_id].append(tfidf_vector(tokens, idf))

    best_event_id = None
    best_sim      = 0.0

    for eid, vecs in event_articles.items():
        # Centroid = average
        all_terms = set(t for v in vecs for t in v)
        centroid  = {
            term: sum(v.get(term, 0) for v in vecs) / len(vecs)
            for term in all_terms
        }
        sim = cosine_similarity(new_vec, centroid)
        if sim > best_sim:
            best_sim      = sim
            best_event_id = eid

    if best_sim >= SIMILARITY_THRESHOLD and best_event_id is not None:
        article.event_id = best_event_id
        # Update credibility for the event
        event = session.get(db.Event, best_event_id)
        if event:
            event.credibility_score = recalculate_event_credibility(
                [a for a in existing_articles if a.event_id == best_event_id] + [article]
            )
            # Update narrative stage
            from datetime import datetime, timedelta
            recent_cutoff = datetime.utcnow() - timedelta(hours=6)
            cluster_articles = [
                a for a in existing_articles if a.event_id == best_event_id
            ] + [article]
            recent = [a for a in cluster_articles if a.timestamp and a.timestamp >= recent_cutoff]
            event.narrative_stage = detect_narrative_stage(
                len(cluster_articles), len(recent)
            )
        session.commit()
        return best_event_id
    else:
        return _create_event_for(article, session)


def _create_event_for(article: db.Article, session: Session) -> int:
    """Create a new Event from a single article and link them."""
    event = db.Event(
        title=article.headline[:512],
        credibility_score=0.0,
        narrative_stage="emerging",
        summary=article.content[:500] if article.content else None,
    )
    session.add(event)
    session.flush()   # get event_id

    article.event_id = event.event_id
    from services.scoring import recalculate_event_credibility
    event.credibility_score = recalculate_event_credibility([article])
    session.commit()
    return event.event_id


def run_full_clustering(session: Session):
    """Cluster all unassigned articles. Called at startup or on demand."""
    unassigned = (
        session.query(db.Article)
        .filter(db.Article.event_id.is_(None))
        .order_by(db.Article.timestamp)
        .all()
    )
    for article in unassigned:
        cluster_article(article.id, session)
