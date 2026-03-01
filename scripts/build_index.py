"""Build and save chunk embeddings to disk.

Mirrors the reference repo's embeder.py, adapted for this project's layout.

Usage:
  python3 scripts/build_index.py --embed sentence-transformers
  python3 scripts/build_index.py --embed BAAI

Shorter, more relevant chunks (reduce length, improve precision):
  - Fixed:  --chunking fixed --chunk-size 100 --chunk-overlap 20
  - Semantic: --chunking semantic --chunk-size 100 --semantic-percentile 85
  - Sentence: --chunking sentence --sentences-per-chunk 3  (recommended for relevance)
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
from rag_utils.chunking import chunk_text, semantic_chunk_text, sentence_chunk_text  # noqa: E402
from rag_utils.dense_retriever import embed_texts  # noqa: E402

_EMBED_MODEL_MAP = {
    "sentence-transformers": "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI": "BAAI/bge-m3",
    "stella": "NovaSearch/stella_en_1.5B_v5",
}


def main():
    ap = argparse.ArgumentParser(description="Build and save dense embeddings for RAG.")
    ap.add_argument(
        "--embed",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "BAAI", "stella"],
        help="Embedding model to use.",
    )
    ap.add_argument(
        "--chunking",
        type=str,
        default="fixed",
        choices=["fixed", "semantic", "sentence"],
        help="Chunking: 'fixed' (tokens), 'semantic' (embedding boundaries), 'sentence' (N sentences).",
    )
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="Max tokens per chunk (fixed/semantic) or sentences per chunk (sentence).")
    ap.add_argument("--chunk-overlap", type=int, default=50,
                    help="Token overlap between chunks (fixed chunking only).")
    ap.add_argument("--semantic-buffer", type=int, default=1,
                    help="Sentence buffer size for context windows (semantic chunking).")
    ap.add_argument("--semantic-percentile", type=int, default=95,
                    help="Breakpoint percentile threshold (semantic chunking).")
    ap.add_argument("--sentences-per-chunk", type=int, default=3,
                    help="Sentences per chunk when --chunking sentence (short, focused chunks).")
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
    elif args.chunking == "sentence":
        chunks_path = cfg.rag_chunks_path.replace(".jsonl", "_sentence.jsonl")

    emb_path = cfg.rag_embeddings_path.replace(
        ".npy", f"_{args.embed}_{args.chunking}.npy"
    )

    ensure_dir(cfg.rag_dir)

    cleaned_dir = Path(cfg.parsed_dir) / "cleaned"

    # ------------------------------------------------------------------
    # 1. Build / load chunks
    # ------------------------------------------------------------------
    if os.path.exists(chunks_path) and not args.rebuild:
        print(f"Loading existing chunks from {chunks_path} …")
        chunk_records = list(read_jsonl(chunks_path))
    else:
        txt_files = sorted(cleaned_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No .txt files found in {cleaned_dir}. "
                "Populate data/parsed/cleaned/ first."
            )
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

        print(f"Chunking {len(txt_files)} docs from {cleaned_dir} (strategy={args.chunking}) …")
        chunk_records = []
        semantic_model = None
        if args.chunking == "semantic":
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer(model_name)
        for txt_file in txt_files:
            doc_id = txt_file.stem
            text = txt_file.read_text(encoding="utf-8")
            if not text.strip():
                continue

            if args.chunking == "semantic":
                chunks = semantic_chunk_text(
                    text,
                    model=semantic_model,
                    embed_model=model_name,
                    buffer_size=args.semantic_buffer,
                    breakpoint_percentile=args.semantic_percentile,
                    max_chunk_tokens=args.chunk_size,
                )
            elif args.chunking == "sentence":
                chunks = sentence_chunk_text(
                    text,
                    sentences_per_chunk=args.sentences_per_chunk,
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
