"""
services/semantic_clustering.py
---------------------------------
Sentence-Transformer semantic similarity for event clustering.
Falls back gracefully to TF-IDF (clustering.py) if library unavailable.

Model: all-MiniLM-L6-v2 (~80 MB, CPU-friendly)
Threshold: 0.65 cosine similarity (vs 0.30 for TF-IDF)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SEMANTIC_THRESHOLD = 0.65

_model = None          # lazy-loaded
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence transformer model. Returns None if unavailable."""
    global _model
    if _model is None:
        try:
            # Pin torch to a single thread before loading the model. Combined
            # with OMP_NUM_THREADS=1 (set in main.py), this avoids the macOS
            # libomp segfault when torch's OpenMP pool is created from a
            # background thread alongside xgboost's OpenMP pool.
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Loaded sentence transformer: %s", _MODEL_NAME)
        except Exception as e:
            logger.info("sentence-transformers unavailable (%s); using TF-IDF fallback", e)
            _model = "unavailable"
    return _model if _model != "unavailable" else None


def is_available() -> bool:
    """Return True if sentence-transformers is ready to use."""
    return _get_model() is not None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def find_best_match(
    query_text: str,
    event_titles: List[str],
    event_ids: List[int],
    threshold: float = SEMANTIC_THRESHOLD,
) -> Tuple[Optional[int], float]:
    """
    Find the best matching event for a new article via semantic similarity.
    Returns (event_id, score) or (None, 0.0) if below threshold or unavailable.
    Caller should fall back to TF-IDF when (None, 0.0) is returned.
    """
    if not event_titles:
        return None, 0.0

    model = _get_model()
    if model is None:
        return None, 0.0   # signal caller to use TF-IDF

    try:
        embeddings = model.encode(
            [query_text] + event_titles,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64,
        )
        query_emb  = embeddings[0]
        event_embs = embeddings[1:]

        best_id, best_score = None, 0.0
        for eid, emb in zip(event_ids, event_embs):
            sim = _cosine(query_emb, emb)
            if sim > best_score:
                best_score, best_id = sim, eid

        return (best_id, best_score) if best_score >= threshold else (None, best_score)
    except Exception as e:
        logger.debug("Semantic matching failed: %s", e)
        return None, 0.0


def batch_encode(texts: List[str]) -> Optional[np.ndarray]:
    """Encode a list of texts. Returns ndarray or None if unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
    except Exception as e:
        logger.debug("Batch encode failed: %s", e)
        return None
