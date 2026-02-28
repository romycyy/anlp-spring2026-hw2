from __future__ import annotations

import re
from typing import List

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Break text into fixed-size token chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0 (got {chunk_size})")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0 (got {overlap})")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be < chunk_size (got overlap={overlap}, chunk_size={chunk_size})"
        )
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(" ".join(tokens[start:end]))
        start = end - overlap
    return chunks


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def semantic_chunk_text(
    text: str,
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    buffer_size: int = 1,
    breakpoint_percentile: int = 95,
    max_chunk_tokens: int = 500,
) -> List[str]:
    """Split text into semantically coherent chunks.

    Strategy (Greg Kamradt-style):
    1. Split into sentences.
    2. For each sentence build a context window by concatenating
       ``buffer_size`` neighbouring sentences on each side.
    3. Embed each context window.
    4. Compute cosine *distance* between consecutive windows.
    5. Any distance above ``breakpoint_percentile`` is a split point.
    6. Hard-cap chunks at ``max_chunk_tokens`` tokens by re-splitting
       oversized chunks with the fixed-size chunker.

    Parameters
    ----------
    embed_model:
        Sentence-transformers model name used to embed the context windows.
    buffer_size:
        Number of neighbouring sentences on each side included in the
        context window for computing the similarity between boundaries.
    breakpoint_percentile:
        Percentile of cosine distances used as the split threshold.
        Higher values → fewer, larger chunks.
    max_chunk_tokens:
        Hard upper limit on chunk length in whitespace-tokens.  Chunks
        that exceed this are recursively split with :func:`chunk_text`.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    # Build context-window strings for each sentence.
    combined = []
    for i, sent in enumerate(sentences):
        lo = max(0, i - buffer_size)
        hi = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[lo:hi]))

    model = SentenceTransformer(embed_model)
    embs = model.encode(combined, show_progress_bar=False, convert_to_numpy=True,
                        normalize_embeddings=True)

    # Cosine distance between consecutive windows (1 − dot since L2-normalised).
    dists = np.array([
        1.0 - float(np.dot(embs[i], embs[i + 1]))
        for i in range(len(embs) - 1)
    ])

    threshold = float(np.percentile(dists, breakpoint_percentile))

    # Group sentences into chunks at breakpoints.
    raw_chunks: List[str] = []
    current: List[str] = [sentences[0]]
    for i, dist in enumerate(dists):
        if dist > threshold:
            raw_chunks.append(" ".join(current))
            current = []
        current.append(sentences[i + 1])
    if current:
        raw_chunks.append(" ".join(current))

    # Enforce hard token cap via fixed chunker.
    final_chunks: List[str] = []
    for ch in raw_chunks:
        if len(ch.split()) > max_chunk_tokens:
            final_chunks.extend(
                chunk_text(ch, chunk_size=max_chunk_tokens, overlap=max_chunk_tokens // 10)
            )
        else:
            final_chunks.append(ch)

    return [c for c in final_chunks if c.strip()]