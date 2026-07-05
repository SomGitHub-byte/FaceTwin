"""Local celebrity embedding database queries."""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from typing import Any

import numpy as np

from config import CELEBRITY_DB_PATH
from mcp_server.tools.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_db() -> dict[str, Any]:
    if not CELEBRITY_DB_PATH.exists():
        raise FileNotFoundError(
            f"Celebrity database not found at {CELEBRITY_DB_PATH}. "
            "Run: python scripts/build_celebrity_db.py"
        )
    with open(CELEBRITY_DB_PATH, "rb") as f:
        return pickle.load(f)


def query_celebrity_db(vector: list[float], top_k: int = 5) -> dict[str, Any]:
    """Return top-k celebrity matches by cosine similarity."""
    if not vector:
        return {"ok": False, "error": "Empty embedding vector"}

    try:
        db = _load_db()
    except FileNotFoundError as exc:
        return {"ok": False, "error": str(exc)}

    names: list[str] = db["names"]
    embeddings: np.ndarray = db["embeddings"]

    scores = [cosine_similarity(vector, embeddings[i]) for i in range(len(names))]
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    matches = [
        {"name": names[i], "score": round(float(scores[i]), 6)}
        for i in ranked_indices
    ]

    return {
        "ok": True,
        "matches": matches,
        "all_scores": scores,
        "score_stats": db.get("score_stats", {}),
    }


def get_db_score_distribution() -> list[float]:
    """Return precomputed reference score distribution for normalization."""
    db = _load_db()
    return db.get("reference_scores", [])
