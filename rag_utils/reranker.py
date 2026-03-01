"""Cross-encoder re-ranking for retrieval results."""

from __future__ import annotations

from typing import List, Optional

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_rerank_model = None
_loaded_rerank_name: Optional[str] = None


def get_reranker_model(model_name: str = _DEFAULT_MODEL):
    """Load and cache the cross-encoder reranker."""
    global _rerank_model, _loaded_rerank_name
    if _rerank_model is None or _loaded_rerank_name != model_name:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(model_name)
        _loaded_rerank_name = model_name
    return _rerank_model


def rerank(
    query: str,
    candidates: List[dict],
    top_k: int = 5,
    model_name: str = _DEFAULT_MODEL,
) -> List[dict]:
    """
    Re-rank candidate chunks by cross-encoder score for a single query.

    candidates: list of dicts with at least "text" (and optionally "chunk_id", "score").
    Returns a new list of the same dicts, sorted by relevance, with "score" updated to
    the cross-encoder score, limited to top_k.
    """
    if not candidates:
        return []
    if top_k >= len(candidates):
        # Still re-rank to get consistent scores
        pass

    model = get_reranker_model(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Attach score to each candidate and sort descending
    scored = [
        {**c, "score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    out = scored[:top_k]
    # Re-assign rank
    for r, item in enumerate(out, 1):
        item["rank"] = r
    return out


def rerank_batch(
    queries: List[str],
    candidates_per_query: List[List[dict]],
    top_k: int = 5,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
) -> List[List[dict]]:
    """
    Re-rank candidates for multiple queries.
    candidates_per_query[i] is the list of candidate dicts for queries[i].
    """
    from tqdm import tqdm

    results: List[List[dict]] = []

    for q_idx, (query, candidates) in enumerate(
        tqdm(
            list(zip(queries, candidates_per_query)),
            desc="Reranking",
            disable=len(queries) <= 1,
        )
    ):
        results.append(
            rerank(query, candidates, top_k=top_k, model_name=model_name)
        )
    return results
