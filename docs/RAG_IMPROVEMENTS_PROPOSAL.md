# RAG Performance Improvement Proposal

This document summarizes proposed changes to improve retrieval quality, latency, and answer accuracy of the RAG pipeline. The codebase already has a solid foundation: dense (FAISS) + sparse (BM25) with RRF fusion, configurable chunking, and a Gemma-2 reader.

---

## 1. Retrieval

### 1.1 Add cross-encoder re-ranking (high impact)

**Current:** Top-k chunks from dense/sparse/RRF are sent directly to the reader.

**Proposal:** Add an optional **re-ranker** step: take a larger candidate set (e.g. 20–50) from hybrid retrieval, then score (query, chunk) pairs with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-v2-m3`) and keep the top 5. Cross-encoders consistently improve retrieval quality for RAG.

**Implementation:** New module `rag_utils/reranker.py` with a function that accepts `list[(query, chunk_text)]` and returns indices or re-ordered list; integrate in `run_rag.py` behind a `--rerank` flag and `--rerank-top-k` (e.g. 20 → 5).

---

### 1.2 Tune hybrid fusion

**Current:** RRF with fixed `k=60` and `candidate_k=20`, `top_k=5`.

**Proposal:**
- Expose and tune **RRF k** (e.g. try 40, 60, 80) and **candidate_k** (e.g. 30–50 when using re-ranking).
- Optionally implement **weighted score fusion** (normalize dense and BM25 scores to [0,1], then `α * dense + (1-α) * sparse`) as an alternative to RRF for A/B comparison.

---

### 1.3 Stronger embedding model

**Current:** Default is `all-MiniLM-L6-v2`; BAAI/bge-m3 is already supported.

**Proposal:** Prefer **BAAI/bge-m3** (or other MTEB-top models within the assignment’s constraints) for dense retrieval when quality is prioritized; keep MiniLM as a fast option. Document in README how to switch via `--embed BAAI`.

---

## 2. Chunking

### 2.1 Reuse embedding model in semantic chunking (medium impact, speed)

**Current:** `semantic_chunk_text()` in `chunking.py` instantiates a new `SentenceTransformer(embed_model)` on every call. In `build_index.py`, this is called once per document, so the model is loaded once per document — very slow for large corpora.

**Proposal:** Add an optional `model` argument to `semantic_chunk_text()`. In `build_index.py`, load the embedding model once and pass it into each `semantic_chunk_text()` call so the same model is reused for all documents.

---

### 2.2 Sentence-boundary aware fixed chunking

**Current:** Fixed chunking in `chunk_text()` splits by token count only, which can cut sentences in half.

**Proposal:** When splitting, prefer breaking at sentence boundaries (e.g. after `.!?`). Implementation: split text into sentences, then greedily group sentences into chunks up to `chunk_size` tokens, with overlap implemented by re-including the last N tokens (or last sentence) of the previous chunk.

---

### 2.3 Chunk size and overlap

**Current:** Defaults in `build_index.py` are `chunk_size=200`, `chunk_overlap=50`.

**Proposal:** Try slightly larger chunks (e.g. 300–400 tokens) with overlap 50–80 for better context continuity, and document the effect on leaderboard. Keep current defaults as baseline.

---

## 3. Reader

### 3.1 Order context by relevance (small but correct)

**Current:** `_build_context()` in `reader.py` uses the order of `retrieved_chunks` and stops when hitting `max_chars`. Order is already that of the retriever, but when truncating, the highest-scoring chunk should be first.

**Proposal:** Ensure chunks are sorted by score descending before building context (and optionally put the single best chunk in a “Most relevant:” line). This avoids accidentally dropping the best chunk when trimming to `MAX_CONTEXT_CHARS`.

---

### 3.2 Conciseness and format

**Current:** Prompt asks for “final answer only” and “Do not show reasoning steps.” Max new tokens = 256.

**Proposal:**
- For factual QA, consider **max_new_tokens=64–128** to reduce drift and speed up inference.
- Add one **1–2 shot example** in the prompt (question + short context + ideal short answer) to reinforce “one phrase or one sentence” format and improve F1/ROUGE on exact-match style evaluation.

---

### 3.3 Optional two-stage reader

**Proposal:** For harder questions, optionally use a first pass that outputs “no relevant info in context” vs “answer in context,” and only then generate the answer. This can reduce hallucinations when retrieval fails. Lower priority than re-ranking and chunking fixes.

---

## 4. Pipeline and efficiency

### 4.1 Batch query embedding (medium impact, speed)

**Current:** For `--queries-file`, the pipeline calls `embed_query(q)` once per question, so the dense model runs N times with batch size 1.

**Proposal:** When running with `--queries-file`, collect all questions, call a new `embed_queries(questions, ...)` that encodes the full list in batches (e.g. 32), then run `dense_search` once with the (N, dim) matrix. This reduces overhead and improves throughput.

---

### 4.2 Batch RRF and re-ranking

**Current:** RRF and retrieval are already structured as per-query lists; re-ranker will be called in a batch (all query–chunk pairs per query, or all queries) to avoid Python loops over queries where possible.

---

## 5. Data and evaluation

### 5.1 Document metadata in context

**Current:** Only chunk text is sent to the reader.

**Proposal:** If available, prepend a short source label to each chunk (e.g. “Source: [Wikipedia – Pittsburgh]”) so the model can weight or cite sources. Requires storing and passing `doc_id` or `url` in the chunk records (already have `doc_id`); add a `title` or `source` field if present in the crawl.

---

### 5.2 Dev set evaluation script

**Proposal:** Add a small script that runs the pipeline on a fixed dev set (e.g. `leaderboard_queries.json` + a reference file), computes F1/ROUGE-L/EM (or whatever the leaderboard uses), and optionally runs statistical significance tests between two runs. This supports iterative tuning without burning leaderboard submissions.

---

## Implementation priority

| Priority | Change                               | Impact      | Effort |
|----------|--------------------------------------|------------|--------|
| 1        | Cross-encoder re-ranking             | High       | Medium |
| 2        | Batch query embedding (queries-file) | Medium     | Low    |
| 3        | Reuse model in semantic chunking     | Medium     | Low    |
| 4        | Sort context by score in reader      | Small      | Low    |
| 5        | Sentence-boundary fixed chunking     | Small–Med  | Medium |
| 6        | Tune RRF k / candidate_k / rerank   | Small–Med  | Low    |
| 7        | 1–2 shot in prompt / max_new_tokens | Small      | Low    |

The following changes are implemented in code:

- **Re-ranker module** (`rag_utils/reranker.py`) and integration in `run_rag.py`.
- **Batch query embedding** in `run_rag.py` when using `--queries-file`.
- **Reuse of embedding model** in semantic chunking (optional `model` in `semantic_chunk_text`, and single model in `build_index.py`).
- **Context sorted by score** in `reader._build_context()`.

You can then tune re-ranker choice, RRF k, candidate_k, and reader prompt/max_new_tokens on the leaderboard or your dev set.
