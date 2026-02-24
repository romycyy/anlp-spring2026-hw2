import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_dense_model: Optional[SentenceTransformer] = None
_loaded_model_name: Optional[str] = None


def get_dense_model(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _dense_model, _loaded_model_name
    if _dense_model is None or _loaded_model_name != model_name:
        # SentenceTransformer caches models on disk after first download.
        _dense_model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
    return _dense_model


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    return x


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / denom


def embed_texts(
    texts: list[str],
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode list of text chunks -> dense vectors."""
    model = get_dense_model(model_name)
    emb = model.encode(texts, show_progress_bar=show_progress_bar)
    return np.asarray(emb, dtype=np.float32)


def embed_query(query: str, *, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a query string -> (1, dim) dense vector."""
    model = get_dense_model(model_name)
    emb = model.encode([query], show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


class DenseRetriever:
    def __init__(self, doc_embeddings: np.ndarray):
        doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
        if doc_embeddings.ndim != 2:
            raise ValueError(
                f"doc_embeddings must be a 2D array (got shape={doc_embeddings.shape})"
            )
        self.doc_embeddings = _l2_normalize(doc_embeddings, axis=1)

    def search(self, query_embedding, top_k=5):
        q = _l2_normalize(_as_2d(np.asarray(query_embedding, dtype=np.float32)), axis=1)
        sims = (q @ self.doc_embeddings.T).ravel()
        ranked = np.argsort(sims)[::-1][:top_k]
        return ranked, sims[ranked]
