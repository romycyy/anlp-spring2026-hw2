"""Hybrid retrieval: Weighted Average Fusion and Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Score normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_scores(result_list: list[dict]) -> list[dict]:
    """Min-max normalise the 'score' field of a result list (in-place copy)."""
    if not result_list:
        return result_list
    scores = [r["score"] for r in result_list]
    min_s, max_s = min(scores), max(scores)
    denom = max_s - min_s if max_s != min_s else 1.0
    return [{**r, "score": (r["score"] - min_s) / denom} for r in result_list]


# ---------------------------------------------------------------------------
# Weighted Average Fusion
# ---------------------------------------------------------------------------

def weighted_average_fusion(
    dense_res: list[list[dict]],
    sparse_res: list[list[dict]],
    alpha: float = 0.6,
    top_k: int = 5,
) -> list[list[dict]]:
    """
    Weighted average of min-max-normalised dense and sparse scores.

    alpha controls the weight on the *dense* score (unlike the reference repo
    where alpha weights sparse; here alpha > 0.5 means "trust dense more").
    """
    results: list[list[dict]] = []
    for q_idx in tqdm(range(len(dense_res)), desc="Weighted Fusion"):
        dense_norm = _normalize_scores(dense_res[q_idx])
        sparse_norm = _normalize_scores(sparse_res[q_idx])

        dense_dict = {r["chunk_id"]: r["score"] for r in dense_norm}
        sparse_dict = {r["chunk_id"]: r["score"] for r in sparse_norm}
        text_map = {r["chunk_id"]: r["text"] for r in dense_res[q_idx] + sparse_res[q_idx]}

        all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
        combined: dict[str, float] = {
            cid: alpha * dense_dict.get(cid, 0.0) + (1.0 - alpha) * sparse_dict.get(cid, 0.0)
            for cid in all_ids
        }

        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        one_query: list[dict] = [
            {"rank": rank, "chunk_id": cid, "score": float(score), "text": text_map.get(cid, "")}
            for rank, (cid, score) in enumerate(sorted_ids, 1)
        ]
        results.append(one_query)
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_res: list[list[dict]],
    sparse_res: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
) -> list[list[dict]]:
    """
    RRF score = Σ 1 / (k + rank_i) across retriever systems.
    Uses the 'rank' field already stored in each result dict.
    """
    results: list[list[dict]] = []
    for q_idx in tqdm(range(len(dense_res)), desc="RRF Fusion"):
        rr_scores: dict[str, float] = {}
        text_map: dict[str, str] = {}

        for r in dense_res[q_idx] + sparse_res[q_idx]:
            cid = r["chunk_id"]
            rr_scores[cid] = rr_scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
            text_map.setdefault(cid, r["text"])

        sorted_ids = sorted(rr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        one_query: list[dict] = [
            {"rank": rank, "chunk_id": cid, "score": float(score), "text": text_map.get(cid, "")}
            for rank, (cid, score) in enumerate(sorted_ids, 1)
        ]
        results.append(one_query)
    return results


# ---------------------------------------------------------------------------
# Convenience single-query wrappers (used by run_rag.py)
# ---------------------------------------------------------------------------

def hybrid_search_single(
    dense_result: list[dict],
    sparse_result: list[dict],
    alpha: float = 0.6,
    top_k: int = 5,
    mode: str = "rrf",
    rrf_k: int = 60,
) -> list[dict]:
    """Single-query hybrid search; mode is 'weighted' or 'rrf'."""
    if mode == "weighted":
        return weighted_average_fusion(
            [dense_result], [sparse_result], alpha=alpha, top_k=top_k
        )[0]
    return reciprocal_rank_fusion(
        [dense_result], [sparse_result], k=rrf_k, top_k=top_k
    )[0]
