"""Hybrid retrieval: Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from tqdm import tqdm


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


def rrf_single(
    dense_result: list[dict],
    sparse_result: list[dict],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Single-query RRF fusion."""
    return reciprocal_rank_fusion(
        [dense_result], [sparse_result], k=k, top_k=top_k
    )[0]
