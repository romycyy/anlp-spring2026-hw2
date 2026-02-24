from rank_bm25 import BM25Okapi

class SparseRetriever:
    def __init__(self, documents):
        self.tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, top_k=5):
        q_tok = query.lower().split()
        scores = self.bm25.get_scores(q_tok)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]