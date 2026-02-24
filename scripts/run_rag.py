from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python3 scripts/run_rag.py` from anywhere.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.config import CrawlConfig  # noqa: E402
from data_prep.utils import ensure_dir, read_jsonl, write_jsonl  # noqa: E402


def _load_or_build_chunks(cfg: CrawlConfig, *, chunk_size: int, overlap: int, rebuild: bool):
    from rag_utils.chunking import chunk_text

    ensure_dir(cfg.rag_dir)

    if os.path.exists(cfg.rag_chunks_path) and not rebuild:
        chunks = []
        metas = []
        for rec in read_jsonl(cfg.rag_chunks_path):
            chunks.append(rec["text"])
            metas.append({k: v for k, v in rec.items() if k != "text"})
        if chunks:
            return chunks, metas

    # Build from docs.jsonl produced by data prep.
    if not os.path.exists(cfg.parsed_docs_path):
        raise FileNotFoundError(
            f"Could not find parsed docs at {cfg.parsed_docs_path}. "
            "Run `python3 scripts/run_pipeline.py` first."
        )

    # Reset chunks output.
    if os.path.exists(cfg.rag_chunks_path):
        os.remove(cfg.rag_chunks_path)

    chunks: list[str] = []
    metas: list[dict] = []

    for doc in read_jsonl(cfg.parsed_docs_path):
        doc_id = doc.get("doc_id")
        text = doc.get("text") or ""
        if not doc_id or not text.strip():
            continue

        doc_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(doc_chunks):
            rec = {
                "chunk_id": f"{doc_id}:{i}",
                "doc_id": doc_id,
                "text": ch,
            }
            write_jsonl(cfg.rag_chunks_path, rec)
            chunks.append(ch)
            metas.append({k: v for k, v in rec.items() if k != "text"})

    if not chunks:
        raise ValueError(
            f"No chunks were produced from {cfg.parsed_docs_path}. "
            "Check that your corpus has non-empty `text` fields."
        )
    return chunks, metas


def _load_or_build_embeddings(
    cfg: CrawlConfig,
    chunks: list[str],
    *,
    rebuild: bool,
):
    import numpy as np
    from rag_utils.dense_retriever import embed_texts

    if os.path.exists(cfg.rag_embeddings_path) and not rebuild:
        emb = np.load(cfg.rag_embeddings_path)
        if emb.shape[0] == len(chunks):
            return emb

    ensure_dir(cfg.rag_dir)
    emb = embed_texts(chunks, show_progress_bar=True)
    np.save(cfg.rag_embeddings_path, emb)
    return emb


def main():
    ap = argparse.ArgumentParser(description="Run a simple RAG baseline over data/parsed/docs.jsonl")
    ap.add_argument("--question", type=str, default=None, help="Ask one question and exit.")
    ap.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight on sparse scores (0..1).")
    ap.add_argument("--top-k", type=int, default=3, help="Final number of chunks passed to reader.")
    ap.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size for sparse+dense before hybrid fusion.",
    )
    ap.add_argument("--chunk-size", type=int, default=200, help="Chunk size in whitespace tokens.")
    ap.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens.")
    ap.add_argument(
        "--reader-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model name for the reader (seq2seq or causal).",
    )
    ap.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild chunks/embeddings even if cached files exist.",
    )
    args = ap.parse_args()

    # Heavy deps (sentence-transformers/torch/transformers) are imported after argparse
    # so `--help` works even before installing optional ML packages.
    try:
        from rag_utils.dense_retriever import DenseRetriever, embed_query
        from rag_utils.hybrid_retrieval import hybrid_search
        from rag_utils.reader import answer_question
        from rag_utils.sparse_retriever import SparseRetriever
    except Exception as e:
        msg = (
            "Failed to import RAG ML dependencies.\n\n"
            f"Import error: {repr(e)}\n\n"
            "This is usually caused by incompatible package versions. In particular:\n"
            "- NumPy should be < 2 (many PyTorch wheels are built against NumPy 1.x)\n"
            "- On macOS Intel (darwin-x64), torch wheels on PyPI commonly top out at 2.2.x\n\n"
            "Fix by reinstalling with the pinned versions in requirements.txt:\n"
            "  pip install -U -r requirements.txt\n"
        )
        raise SystemExit(msg) from e

    cfg = CrawlConfig()

    chunks, metas = _load_or_build_chunks(
        cfg, chunk_size=args.chunk_size, overlap=args.chunk_overlap, rebuild=args.rebuild_index
    )
    embeddings = _load_or_build_embeddings(cfg, chunks, rebuild=args.rebuild_index)

    sparse = SparseRetriever(chunks)
    dense = DenseRetriever(embeddings)

    def answer(q: str):
        sparse_res = sparse.search(q, top_k=args.candidate_k)
        dense_ranks, _dense_scores = dense.search(embed_query(q), top_k=args.candidate_k)
        final_indices = hybrid_search(
            sparse_res, list(dense_ranks), alpha=args.alpha, top_k=args.top_k
        )

        top_chunks = [chunks[i] for i in final_indices]
        ans = answer_question(q, top_chunks, model_name=args.reader_model)

        # Print answer only (no sources for now).
        print("\nAnswer:\n", ans, "\n", sep="")

    if args.question:
        answer(args.question)
        return

    while True:
        try:
            q = input("\nAsk a question (or Ctrl-D to quit): ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        answer(q)


if __name__ == "__main__":
    main()