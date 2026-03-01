#!/usr/bin/env bash
# Run RAG experiments (A)–(E) from the report.
# Uses: build_index.py, run_rag.py. Set QUERIES_FILE or leave default.

set -e
QUERIES_FILE="${QUERIES_FILE:-leaderboard_queries.json}"
OUT_DIR="${OUT_DIR:-system_outputs}"

echo "Queries file: $QUERIES_FILE"
echo "Output dir:   $OUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# (A) Baseline RAG: fixed chunking, sentence-transformers, RRF, top-k=5, reranker
# ---------------------------------------------------------------------------
echo "=== (A) Build index: fixed chunking + sentence-transformers ==="
python3 scripts/build_index.py --chunking fixed --embed sentence-transformers

echo "=== (A) Run RAG: RRF + rerank, top-k=5 ==="
python3 scripts/run_rag.py \
  --queries-file "$QUERIES_FILE" \
  --mode rrf \
  --embed sentence-transformers \
  --chunking fixed \
  --top-k 5 \
  --rerank \
  --rerank-top-k 20 \
  --output "$OUT_DIR/system_output_A_baseline_rrf_rerank.json"

# ---------------------------------------------------------------------------
# (B) Baseline with dense-only (no BM25)
# ---------------------------------------------------------------------------
echo "=== (B) Run RAG: dense-only + rerank (same index as A) ==="
python3 scripts/run_rag.py \
  --queries-file "$QUERIES_FILE" \
  --mode dense \
  --embed sentence-transformers \
  --chunking fixed \
  --top-k 5 \
  --rerank \
  --rerank-top-k 20 \
  --output "$OUT_DIR/system_output_B_dense_only.json"

# ---------------------------------------------------------------------------
# (C) Baseline with sparse-only (no dense)
# ---------------------------------------------------------------------------
echo "=== (C) Run RAG: sparse-only + rerank (same index as A) ==="
python3 scripts/run_rag.py \
  --queries-file "$QUERIES_FILE" \
  --mode sparse \
  --embed sentence-transformers \
  --chunking fixed \
  --top-k 5 \
  --rerank \
  --rerank-top-k 20 \
  --output "$OUT_DIR/system_output_C_sparse_only.json"

# ---------------------------------------------------------------------------
# (D) Baseline with bge-m3 embedding (fixed chunking)
# ---------------------------------------------------------------------------
echo "=== (D) Build index: fixed chunking + BAAI/bge-m3 ==="
python3 scripts/build_index.py --chunking fixed --embed BAAI

echo "=== (D) Run RAG: RRF + rerank with bge-m3 ==="
python3 scripts/run_rag.py \
  --queries-file "$QUERIES_FILE" \
  --mode rrf \
  --embed BAAI \
  --chunking fixed \
  --top-k 5 \
  --rerank \
  --rerank-top-k 20 \
  --output "$OUT_DIR/system_output_D_bge_m3_fixed.json"

# ---------------------------------------------------------------------------
# (E) Baseline with semantic chunking + bge-m3 embedding
# ---------------------------------------------------------------------------
echo "=== (E) Build index: semantic chunking + BAAI/bge-m3 ==="
python3 scripts/build_index.py --chunking semantic --embed BAAI

echo "=== (E) Run RAG: RRF + rerank, semantic + bge-m3 ==="
python3 scripts/run_rag.py \
  --queries-file "$QUERIES_FILE" \
  --mode rrf \
  --embed BAAI \
  --chunking semantic \
  --top-k 5 \
  --rerank \
  --rerank-top-k 20 \
  --output "$OUT_DIR/system_output_E_semantic_bge_m3.json"

echo ""
echo "Done. Outputs in $OUT_DIR/"
