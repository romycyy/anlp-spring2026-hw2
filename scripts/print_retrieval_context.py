"""Print retrieved context for questions without calling the reader/LLM.

Use this to inspect what context your RAG pipeline would pass to the model.
Helps debug why RAG may underperform closed-book: irrelevant or missing
context can distract the model or add noise.

Usage:
  python3 scripts/print_retrieval_context.py --question "Your question here"
  python3 scripts/print_retrieval_context.py --queries-file path/to/queries.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.config import CrawlConfig  # noqa: E402
from data_prep.utils import read_jsonl  # noqa: E402

_EMBED_MODEL_MAP = {
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI": "BAAI/bge-m3",
}


def _resolve_paths(cfg: CrawlConfig, embed_key: str, chunking: str) -> tuple[str, str]:
    chunks_path = cfg.rag_chunks_path
    if chunking == "semantic":
        chunks_path = cfg.rag_chunks_path.replace(".jsonl", "_semantic.jsonl")
    emb_path = cfg.rag_embeddings_path.replace(".npy", f"_{embed_key}_{chunking}.npy")
    return chunks_path, emb_path


def _load_chunks(cfg: CrawlConfig, chunking: str = "fixed") -> tuple[list[str], list[str]]:
    chunks_path, _ = _resolve_paths(cfg, "", chunking)
    if not Path(chunks_path).exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            f"Run `python3 scripts/build_index.py --chunking {chunking}` first."
        )
    texts, chunk_ids = [], []
    for rec in read_jsonl(chunks_path):
        texts.append(rec["text"])
        chunk_ids.append(rec["chunk_id"])
    if not texts:
        raise ValueError(f"No chunks found in {chunks_path}.")
    return texts, chunk_ids


def _load_embeddings(cfg: CrawlConfig, n_chunks: int, *, embed_key: str, chunking: str = "fixed"):
    import numpy as np
    _, emb_path = _resolve_paths(cfg, embed_key, chunking)
    if not Path(emb_path).exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {emb_path}. "
            f"Run `python3 scripts/build_index.py --embed {embed_key} --chunking {chunking}` first."
        )
    emb = np.load(emb_path)
    if emb.shape[0] != n_chunks:
        raise ValueError(
            f"Embeddings shape {emb.shape} does not match chunk count {n_chunks}. "
            "Re-run `python3 scripts/build_index.py` to rebuild."
        )
    return emb


def _build_context(retrieved_chunks: list, *, max_chars: int = 10_000) -> str:
    """Same logic as rag_utils.reader: assemble context from retrieved chunks."""
    chunks = list(retrieved_chunks)
    if chunks and isinstance(chunks[0], dict) and "score" in chunks[0]:
        chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
    ctxs, total_len = [], 0
    for item in chunks:
        text = item["text"].strip() if isinstance(item, dict) else str(item).strip()
        if not text:
            continue
        if total_len + len(text) > max_chars:
            break
        ctxs.append(text)
        total_len += len(text)
    return "\n\n".join(ctxs)


def main():
    ap = argparse.ArgumentParser(
        description="Print retrieved context for question(s) — no LLM call."
    )
    ap.add_argument("--question", type=str, default=None, help="Single question.")
    ap.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="JSON file with list of {id, question} or {id, query}.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="rrf",
        choices=["dense", "sparse", "rrf"],
        help="Retrieval mode.",
    )
    ap.add_argument(
        "--embed",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "BAAI"],
        help="Embedding model.",
    )
    ap.add_argument(
        "--chunking",
        type=str,
        default="fixed",
        choices=["fixed", "semantic"],
        help="Chunking strategy.",
    )
    ap.add_argument("--top-k", type=int, default=5, help="Number of chunks to show.")
    ap.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size per retriever before RRF.",
    )
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant k.")
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=10_000,
        help="Max characters of context (same as reader).",
    )
    ap.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker.")
    ap.add_argument("--rerank-top-k", type=int, default=20)
    ap.add_argument(
        "--rerank-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print each chunk with score and chunk_id.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set with --queries-file, save contexts to this JSON (qid -> context).",
    )
    args = ap.parse_args()

    try:
        import numpy as np
        from rag_utils.dense_retriever import (
            build_faiss_index,
            dense_search,
            embed_query,
            embed_queries,
        )
        from rag_utils.hybrid_retrieval import rrf_single
        from rag_utils.sparse_retriever import SparseRetriever
        if args.rerank:
            from rag_utils.reranker import rerank
    except Exception as e:
        raise SystemExit(
            f"Failed to import retrieval dependencies.\n{repr(e)}\n"
            "Fix: pip install -U -r requirements.txt"
        ) from e

    cfg = CrawlConfig()
    texts, chunk_ids = _load_chunks(cfg, chunking=args.chunking)
    print(f"Loaded {len(texts)} chunks.", file=sys.stderr)

    sparse = SparseRetriever(texts, chunk_ids)
    faiss_index = None
    id_arr = None
    if args.mode in ("dense", "rrf"):
        embeddings = _load_embeddings(
            cfg, len(texts), embed_key=args.embed, chunking=args.chunking
        )
        id_arr = np.array(chunk_ids)
        faiss_index = build_faiss_index(embeddings)

    retrieve_top_k = args.rerank_top_k if args.rerank else args.top_k
    if args.mode in ("dense", "rrf"):
        candidate_k = max(args.candidate_k, retrieve_top_k) if args.mode == "rrf" else retrieve_top_k
    else:
        candidate_k = retrieve_top_k

    def retrieve(q: str, q_emb: np.ndarray | None = None) -> list[dict]:
        if args.mode == "sparse":
            out = sparse.search(q, top_k=retrieve_top_k)
        else:
            emb = q_emb if q_emb is not None else embed_query(
                q, model_name=_EMBED_MODEL_MAP[args.embed]
            )
            if args.mode == "dense":
                out = dense_search(faiss_index, emb, id_arr, texts, top_k=retrieve_top_k)[0]
            else:
                dense_res = dense_search(faiss_index, emb, id_arr, texts, top_k=candidate_k)[0]
                sparse_res = sparse.search(q, top_k=candidate_k)
                out = rrf_single(dense_res, sparse_res, top_k=retrieve_top_k, k=args.rrf_k)
        if args.rerank and out:
            out = rerank(q, out, top_k=args.top_k, model_name=args.rerank_model)
        return out

    def print_context(question: str, qid: str | None = None) -> str:
        retrieved = retrieve(question)
        context = _build_context(retrieved, max_chars=args.max_context_chars)
        header = f"[{qid}] " if qid else ""
        print(f"\n{'='*60}")
        print(f"{header}Question: {question}")
        print("=" * 60)
        if args.verbose and retrieved:
            for i, item in enumerate(retrieved):
                score = item.get("score", item.get("rank"))
                cid = item.get("chunk_id", "?")
                print(f"\n--- Chunk {i+1} (score={score}, chunk_id={cid}) ---")
                print(item.get("text", item)[:1500])
                if len(item.get("text", item)) > 1500:
                    print("...")
        else:
            print("\nRetrieved context (concatenated):\n")
            print(context if context else "(no chunks)")
        print()
        return context

    # Single question
    if args.question:
        print_context(args.question.strip())
        return

    # Batch from file
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        items = [x for x in queries if x.get("question") or x.get("query")]
        questions = [(x.get("question") or x.get("query", "")).strip() for x in items]
        query_embs = None
        if items and args.mode in ("dense", "rrf"):
            print("Batch embedding queries…", file=sys.stderr)
            query_embs = embed_queries(questions, model_name=_EMBED_MODEL_MAP[args.embed])

        results = {}
        for i, item in enumerate(items):
            qid = str(item.get("id", item.get("qid", i)))
            q = questions[i]
            q_emb = query_embs[i : i + 1] if query_embs is not None else None
            # Temporarily override retrieve to use precomputed emb for this query
            if q_emb is not None:
                retrieved = retrieve(q, q_emb=q_emb)
            else:
                retrieved = retrieve(q)
            context = _build_context(retrieved, max_chars=args.max_context_chars)
            results[qid] = {"question": q, "context": context, "num_chunks": len(retrieved)}
            print(f"\n[{qid}] {q}")
            if args.verbose:
                for j, r in enumerate(retrieved):
                    print(f"  Chunk {j+1} score={r.get('score')} id={r.get('chunk_id')}")
            print(f"  Context length: {len(context)} chars, {len(retrieved)} chunks")
            if not args.verbose:
                print(f"  Context preview: {context[:300].replace(chr(10), ' ')}...")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nSaved contexts to {args.output}", file=sys.stderr)
        return

    # Interactive
    print("Enter questions to see retrieved context (no LLM). Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except EOFError:
            break
        if not q:
            continue
        print_context(q)


if __name__ == "__main__":
    main()
