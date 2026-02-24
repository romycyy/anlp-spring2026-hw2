def hybrid_search(
    sparse_scores: list[tuple[int,float]],
    dense_ranks: list[int],
    alpha: float = 0.5,
    top_k: int = 5
):
    """Combine sparse + dense scores."""
    combined = {}
    # assign sparse normalized
    if sparse_scores:
        max_sparse = max(score for (_, score) in sparse_scores) or 1
        for idx, score in sparse_scores:
            combined[idx] = alpha * (score / max_sparse)

    # assign dense normalized
    max_dense = len(dense_ranks) or 1
    for rank_idx, idx in enumerate(dense_ranks):
        combined[idx] = combined.get(idx, 0) + (1 - alpha) * ((max_dense - rank_idx) / max_dense)

    # sort by combined and return top_k indices
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_docs[:top_k]]