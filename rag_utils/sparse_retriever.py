"""BM25-based sparse retrieval with regex tokenizer and structured results."""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    """Lowercase + keep alphanumerics only (matches reference repo tokenizer)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseRetriever:
    def __init__(self, texts: list[str], chunk_ids: list[str]):
        self.id_list = chunk_ids
        self.text_map = dict(zip(chunk_ids, texts))
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        print("BM25 index built successfully.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k results as structured dicts with rank/chunk_id/score/text."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, i in enumerate(top_idx, 1):
            cid = self.id_list[int(i)]
            results.append({
                "rank": rank,
                "chunk_id": cid,
                "score": float(scores[i]),
                "text": self.text_map[cid],
            })
        return results

    def search_batch(self, questions: list[str], top_k: int = 5) -> list[list[dict]]:
        """Batch search for multiple queries."""
        return [
            self.search(q, top_k=top_k)
            for q in tqdm(questions, desc="BM25 Searching")
        ]
