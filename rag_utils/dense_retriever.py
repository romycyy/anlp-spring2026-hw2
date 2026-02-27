"""Dense retrieval: sentence-transformers or BAAI/bge-m3, with optional FAISS index."""
from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_BAAI_MODEL_NAME = "BAAI/bge-m3"

_dense_model = None
_loaded_model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _load_baai(model_name: str = _BAAI_MODEL_NAME):
    from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]  # pip install FlagEmbedding
    return BGEM3FlagModel(model_name, use_fp16=True)


def get_dense_model(model_name: str = _DEFAULT_MODEL_NAME):
    global _dense_model, _loaded_model_name
    if _dense_model is None or _loaded_model_name != model_name:
        if model_name == _BAAI_MODEL_NAME:
            _dense_model = _load_baai(model_name)
        else:
            _dense_model = _load_sentence_transformer(model_name)
        _loaded_model_name = model_name
    return _dense_model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _encode_batch_st(model, texts: list[str], normalize: bool = True) -> np.ndarray:
    return np.asarray(
        model.encode(texts, show_progress_bar=False, convert_to_numpy=True,
                     normalize_embeddings=normalize),
        dtype=np.float32,
    )


def _encode_batch_baai(model, texts: list[str]) -> np.ndarray:
    import faiss  # type: ignore[import]
    emb = model.encode(texts, batch_size=64)["dense_vecs"].astype("float32")
    faiss.normalize_L2(emb)
    return emb


def embed_texts(
    texts: list[str],
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    show_progress_bar: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode corpus chunks -> (N, dim) float32 array."""
    if not texts:
        raise ValueError("embed_texts received an empty list — no chunks to embed.")
    model = get_dense_model(model_name)
    all_embs: list[np.ndarray] = []
    rng = range(0, len(texts), batch_size)
    if show_progress_bar:
        from tqdm import tqdm as _tqdm
        rng = _tqdm(rng, desc="Embedding chunks")
    for i in rng:
        batch = texts[i: i + batch_size]
        if model_name == _BAAI_MODEL_NAME:
            emb = _encode_batch_baai(model, batch)
        else:
            emb = _encode_batch_st(model, batch, normalize=normalize)
        all_embs.append(emb)
    return np.vstack(all_embs).astype("float32")


def embed_query(query: str, *, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a single query -> (1, dim) float32 array."""
    model = get_dense_model(model_name)
    if model_name == _BAAI_MODEL_NAME:
        emb = _encode_batch_baai(model, [query])
    else:
        emb = _encode_batch_st(model, [query], normalize=True)
    return emb.astype(np.float32)


def embed_queries(model, questions: list[str]) -> np.ndarray:
    """Encode a list of queries using an already-loaded model."""
    if hasattr(model, "encode") and hasattr(model, "tokenize"):
        # sentence-transformers
        return np.asarray(
            model.encode(questions, show_progress_bar=False, convert_to_numpy=True,
                         normalize_embeddings=True),
            dtype=np.float32,
        )
    else:
        # BAAI
        import faiss  # type: ignore[import]
        emb = model.encode(questions, batch_size=64)["dense_vecs"].astype("float32")
        faiss.normalize_L2(emb)
        return emb


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray):
    """Build an L2-normalised FAISS flat inner-product index."""
    import faiss  # type: ignore[import]
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def dense_search(
    index,
    query_embeddings: np.ndarray,
    id_list: np.ndarray,
    texts: list[str],
    top_k: int = 5,
) -> list[list[dict]]:
    """
    Search FAISS index for each query embedding.
    Returns list[list[dict]] with keys: rank, chunk_id, score, text.
    """
    import faiss  # type: ignore[import]
    q = np.asarray(query_embeddings, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    faiss.normalize_L2(q)
    scores_mat, idx_mat = index.search(q, top_k)

    results = []
    for scores_row, idx_row in zip(scores_mat, idx_mat):
        one_query: list[dict] = []
        for rank, (i, score) in enumerate(zip(idx_row, scores_row), 1):
            if i < 0:
                continue
            idx = int(i)
            one_query.append({
                "rank": rank,
                "chunk_id": str(id_list[idx]),
                "score": float(score),
                "text": texts[idx],
            })
        results.append(one_query)
    return results


# ---------------------------------------------------------------------------
# Fallback numpy-only retriever (no FAISS required)
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


class DenseRetriever:
    """Pure-numpy dense retriever (no FAISS dependency)."""

    def __init__(self, doc_embeddings: np.ndarray, id_list: list[str], texts: list[str]):
        self.doc_embeddings = _l2_normalize(np.asarray(doc_embeddings, dtype=np.float32))
        self.id_list = id_list
        self.texts = texts

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        q = _l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        sims = (q @ self.doc_embeddings.T).ravel()
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, i in enumerate(top_idx, 1):
            idx = int(i)
            results.append({
                "rank": rank,
                "chunk_id": str(self.id_list[idx]),
                "score": float(sims[i]),
                "text": self.texts[idx],
            })
        return results

    def search_batch(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> list[list[dict]]:
        return [
            self.search(q, top_k=top_k)
            for q in tqdm(query_embeddings, desc="Dense Searching")
        ]
