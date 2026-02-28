"""Build and save chunk embeddings to disk.

Mirrors the reference repo's embeder.py, adapted for this project's layout.

Usage:
  python3 scripts/build_index.py --embed sentence-transformers
  python3 scripts/build_index.py --embed BAAI
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from data_prep.config import CrawlConfig  # noqa: E402
from data_prep.utils import ensure_dir, read_jsonl, write_jsonl  # noqa: E402
from rag_utils.chunking import chunk_text, semantic_chunk_text  # noqa: E402
from rag_utils.dense_retriever import embed_texts  # noqa: E402

_EMBED_MODEL_MAP = {
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI": "BAAI/bge-m3",
}


def main():
    ap = argparse.ArgumentParser(description="Build and save dense embeddings for RAG.")
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
        help="Chunking strategy: 'fixed' (token-based) or 'semantic' (embedding-based).",
    )
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="Max tokens per chunk (fixed) / hard cap (semantic).")
    ap.add_argument("--chunk-overlap", type=int, default=50,
                    help="Token overlap between chunks (fixed chunking only).")
    ap.add_argument("--semantic-buffer", type=int, default=1,
                    help="Sentence buffer size for context windows (semantic chunking).")
    ap.add_argument("--semantic-percentile", type=int, default=95,
                    help="Breakpoint percentile threshold (semantic chunking).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild chunks even if chunks.jsonl already exists.",
    )
    args = ap.parse_args()

    cfg = CrawlConfig()
    model_name = _EMBED_MODEL_MAP[args.embed]

    # Use distinct file names per chunking strategy so both can coexist.
    chunks_path = cfg.rag_chunks_path
    if args.chunking == "semantic":
        chunks_path = cfg.rag_chunks_path.replace(".jsonl", "_semantic.jsonl")

    emb_path = cfg.rag_embeddings_path.replace(
        ".npy", f"_{args.embed}_{args.chunking}.npy"
    )

    ensure_dir(cfg.rag_dir)

    # ------------------------------------------------------------------
    # 1. Build / load chunks
    # ------------------------------------------------------------------
    if os.path.exists(chunks_path) and not args.rebuild:
        print(f"Loading existing chunks from {chunks_path} …")
        chunk_records = list(read_jsonl(chunks_path))
    else:
        if not os.path.exists(cfg.parsed_docs_path):
            raise FileNotFoundError(
                f"Parsed docs not found at {cfg.parsed_docs_path}. "
                "Run `python3 scripts/run_pipeline.py` first."
            )
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

        print(f"Chunking docs (strategy={args.chunking}) …")
        chunk_records = []
        for idx, doc in enumerate(read_jsonl(cfg.parsed_docs_path)):
            doc_id = doc.get("doc_id") or doc.get("id") or f"doc_{idx}"
            text = doc.get("text") or ""
            if not text.strip():
                continue

            if args.chunking == "semantic":
                chunks = semantic_chunk_text(
                    text,
                    buffer_size=args.semantic_buffer,
                    breakpoint_percentile=args.semantic_percentile,
                    max_chunk_tokens=args.chunk_size,
                )
            else:
                chunks = chunk_text(
                    text, chunk_size=args.chunk_size, overlap=args.chunk_overlap
                )

            for i, ch in enumerate(chunks):
                rec = {
                    "chunk_id": f"{doc_id}:{i}",
                    "doc_id": doc_id,
                    "text": ch,
                }
                write_jsonl(chunks_path, rec)
                chunk_records.append(rec)

        print(f"Created {len(chunk_records)} chunks → {chunks_path}")

    texts = [r["text"] for r in chunk_records]
    ids = np.array([r["chunk_id"] for r in chunk_records])

    print(f"\nEncoding {len(texts)} chunks with {model_name} …")
    embs = embed_texts(
        texts, model_name=model_name, batch_size=args.batch_size, show_progress_bar=True
    )

    np.save(emb_path, embs)
    ids_path = emb_path.replace("embeddings_", "ids_")
    np.save(ids_path, ids)

    print(f"\nSaved embeddings {embs.shape} → {emb_path}")
    print(f"Saved ids        {ids.shape}  → {ids_path}")

    # Quick verification
    embs_v = np.load(emb_path)
    ids_v = np.load(ids_path, allow_pickle=True)
    print("\nVerification:")
    print(f"  embeddings shape : {embs_v.shape}")
    print(f"  ids shape        : {ids_v.shape}")


if __name__ == "__main__":
    main()
