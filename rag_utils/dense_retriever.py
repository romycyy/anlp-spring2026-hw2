"""Dense retrieval: sentence-transformers, BAAI/bge-m3, or NovaSearch/stella with a FAISS index."""

from __future__ import annotations

from typing import Optional

import numpy as np

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_BAAI_MODEL_NAME = "BAAI/bge-m3"

# Models that require trust_remote_code=True when loading via SentenceTransformer.
_TRUST_REMOTE_CODE_MODELS: set[str] = {"NovaSearch/stella_en_1.5B_v5"}

# Prompt name to use when encoding *queries* (not documents) for models that need it.
# Stella uses "s2p_query" (sentence-to-passage) for retrieval; documents need no prompt.
_QUERY_PROMPT_NAMES: dict[str, str] = {"NovaSearch/stella_en_1.5B_v5": "s2p_query"}

_dense_model = None
_loaded_model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _patch_qwen2_rope_theta() -> None:
    """Ensure Qwen2Config instances always expose `rope_theta` as a direct attribute.

    Newer transformers versions (≥4.47) moved rope_theta into the rope_scaling dict,
    so instances no longer have it as a standalone attribute. Stella's custom
    modeling_qwen.py (loaded via trust_remote_code) still accesses config.rope_theta
    directly, causing an AttributeError. This one-time monkey-patch restores it.
    """
    try:
        from transformers import Qwen2Config

        if hasattr(Qwen2Config(), "rope_theta"):
            return  # already present; nothing to do

        _orig_init = Qwen2Config.__init__

        def _patched_init(self, *args, rope_theta: float = 1_000_000.0, **kwargs):
            _orig_init(self, *args, **kwargs)
            if not hasattr(self, "rope_theta"):
                self.rope_theta = rope_theta

        Qwen2Config.__init__ = _patched_init  # type: ignore[method-assign]
    except Exception:
        pass  # non-fatal; let the real load attempt surface any remaining error


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    trust_remote_code = model_name in _TRUST_REMOTE_CODE_MODELS
    if trust_remote_code:
        _patch_qwen2_rope_theta()
    return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)


def get_dense_model(model_name: str = _DEFAULT_MODEL_NAME):
    """Load dense model. BAAI/bge-m3 is loaded via SentenceTransformer to avoid
    FlagEmbedding's BiTrainer import, which is incompatible with newer transformers.
    """
    global _dense_model, _loaded_model_name
    if _dense_model is None or _loaded_model_name != model_name:
        _dense_model = _load_sentence_transformer(model_name)
        _loaded_model_name = model_name
    return _dense_model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _encode_batch_st(
    model,
    texts: list[str],
    normalize: bool = True,
    prompt_name: Optional[str] = None,
) -> np.ndarray:
    kwargs: dict = dict(
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def embed_texts(
    texts: list[str],
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    show_progress_bar: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode corpus chunks -> (N, dim) float32 array. No query prompt is used."""
    if not texts:
        raise ValueError("embed_texts received an empty list — no chunks to embed.")
    model = get_dense_model(model_name)
    all_embs: list[np.ndarray] = []
    rng = range(0, len(texts), batch_size)
    if show_progress_bar:
        from tqdm import tqdm as _tqdm

        rng = _tqdm(rng, desc="Embedding chunks")
    for i in rng:
        batch = texts[i : i + batch_size]
        emb = _encode_batch_st(model, batch, normalize=normalize)
        all_embs.append(emb)
    return np.vstack(all_embs).astype("float32")


def embed_query(query: str, *, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a single query -> (1, dim) float32 array.
    Uses the model's query prompt if defined (e.g. stella's s2p_query).
    """
    model = get_dense_model(model_name)
    prompt_name = _QUERY_PROMPT_NAMES.get(model_name)
    emb = _encode_batch_st(model, [query], normalize=True, prompt_name=prompt_name)
    return emb.astype(np.float32)


def embed_queries(
    queries: list[str],
    *,
    model_name: str = _DEFAULT_MODEL_NAME,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode multiple queries -> (N, dim) float32 array. More efficient than repeated embed_query.
    Uses the model's query prompt if defined (e.g. stella's s2p_query).
    """
    if not queries:
        raise ValueError("embed_queries received an empty list.")
    model = get_dense_model(model_name)
    prompt_name = _QUERY_PROMPT_NAMES.get(model_name)
    all_embs: list[np.ndarray] = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        emb = _encode_batch_st(model, batch, normalize=True, prompt_name=prompt_name)
        all_embs.append(emb)
    return np.vstack(all_embs).astype(np.float32)


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
            one_query.append(
                {
                    "rank": rank,
                    "chunk_id": str(id_list[idx]),
                    "score": float(score),
                    "text": texts[idx],
                }
            )
        results.append(one_query)
    return results
