"""Run the RAG pipeline: retrieve and generate answers.

Assumes the index has been built via `python3 scripts/build_index.py`.

Supports three retrieval modes:
  dense   – FAISS dense retrieval only
  sparse  – BM25 sparse retrieval only
  rrf     – Reciprocal Rank Fusion (dense + sparse)

Supports two embedding models:
  sentence-transformers  (default)
  BAAI                   (requires: pip install FlagEmbedding)
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
from data_prep.utils import ensure_dir, read_jsonl  # noqa: E402

_EMBED_MODEL_MAP = {
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI": "BAAI/bge-m3",
}


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _resolve_paths(cfg: CrawlConfig, embed_key: str, chunking: str) -> tuple[str, str]:
    """Return (chunks_path, emb_path) for the given embed / chunking combo."""
    chunks_path = cfg.rag_chunks_path
    if chunking == "semantic":
        chunks_path = cfg.rag_chunks_path.replace(".jsonl", "_semantic.jsonl")
    emb_path = cfg.rag_embeddings_path.replace(".npy", f"_{embed_key}_{chunking}.npy")
    return chunks_path, emb_path


def _load_chunks(cfg: CrawlConfig, chunking: str = "fixed") -> tuple[list[str], list[str]]:
    """Return (texts, chunk_ids) loaded from disk; raise if file is missing."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="RAG pipeline – dense / sparse / rrf"
    )
    ap.add_argument(
        "--question", type=str, default=None, help="Ask a single question and exit."
    )
    ap.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="JSON file with list of {id, question} objects (e.g. leaderboard_queries.json).",
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
        help="Embedding model to use.",
    )
    ap.add_argument(
        "--chunking",
        type=str,
        default="fixed",
        choices=["fixed", "semantic"],
        help="Chunking strategy used when building the index.",
    )
    ap.add_argument(
        "--top-k", type=int, default=5, help="Final number of chunks passed to reader."
    )
    ap.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size per retriever before RRF fusion.",
    )
    ap.add_argument(
        "--rrf-k", type=int, default=60, help="RRF constant k (default 60)."
    )
    ap.add_argument(
        "--reader-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model name for the reader.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--max-context-chars",
        type=int,
        default=10_000,
        help="Max characters of retrieved context passed to reader.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON output (system_outputs/).",
    )
    args = ap.parse_args()

    try:
        import numpy as np
        from rag_utils.dense_retriever import build_faiss_index, dense_search, embed_query
        from rag_utils.hybrid_retrieval import rrf_single
        from rag_utils.reader import answer_question
        from rag_utils.sparse_retriever import SparseRetriever
    except Exception as e:
        raise SystemExit(
            f"Failed to import RAG ML dependencies.\nImport error: {repr(e)}\n"
            "Fix: pip install -U -r requirements.txt"
        ) from e

    cfg = CrawlConfig()
    texts, chunk_ids = _load_chunks(cfg, chunking=args.chunking)
    print(f"Loaded {len(texts)} chunks.")

    sparse = SparseRetriever(texts, chunk_ids)

    faiss_index = None
    id_arr = None
    if args.mode in ("dense", "rrf"):
        embeddings = _load_embeddings(cfg, len(texts), embed_key=args.embed,
                                      chunking=args.chunking)
        id_arr = np.array(chunk_ids)
        faiss_index = build_faiss_index(embeddings)

    print(f"\nMode={args.mode} | Embed={args.embed} | Chunking={args.chunking} | top_k={args.top_k}\n")

    def retrieve(q: str) -> list[dict]:
        """Run the selected retrieval mode for a single query."""
        if args.mode == "sparse":
            return sparse.search(q, top_k=args.top_k)

        q_emb = embed_query(q, model_name=_EMBED_MODEL_MAP[args.embed])

        if args.mode == "dense":
            return dense_search(faiss_index, q_emb, id_arr, texts, top_k=args.top_k)[0]

        # rrf – need both retrievers
        dense_res = dense_search(faiss_index, q_emb, id_arr, texts, top_k=args.candidate_k)[0]
        sparse_res = sparse.search(q, top_k=args.candidate_k)
        return rrf_single(dense_res, sparse_res, top_k=args.top_k, k=args.rrf_k)

    def run_and_print(q: str) -> str:
        retrieved = retrieve(q)
        ans = answer_question(
            q,
            retrieved,
            model_name=args.reader_model,
            max_new_tokens=args.max_new_tokens,
            max_context_chars=args.max_context_chars,
        )
        print(f"\nAnswer:\n{ans}\n")
        return ans

    # ------------------------------------------------------------------
    # Single question mode
    # ------------------------------------------------------------------
    if args.question:
        run_and_print(args.question)
        return

    # ------------------------------------------------------------------
    # Batch queries from file (e.g. leaderboard_queries.json)
    # ------------------------------------------------------------------
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)

        results: dict[str, str] = {}
        for item in queries:
            qid = str(item.get("id", item.get("qid", "")))
            question = item.get("question", item.get("query", ""))
            if not question:
                continue
            print(f"[{qid}] {question}")
            ans = run_and_print(question)
            results[qid] = ans

        output_path = args.output
        if not output_path:
            ensure_dir("system_outputs")
            output_path = (
                f"system_outputs/system_output_{args.embed}_{args.mode}"
                f"_{args.chunking}_{args.top_k}.json"
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(results)} answers to {output_path}")
        return

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------
    while True:
        try:
            q = input("\nAsk a question (Ctrl-D to quit): ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        run_and_print(q)


if __name__ == "__main__":
    main()
