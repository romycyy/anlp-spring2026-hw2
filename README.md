# RAG System for Pittsburgh & CMU QA

A **retrieval-augmented generation (RAG)** pipeline that answers factual questions about Pittsburgh and Carnegie Mellon University.

**Report:** See **[REPORT.md](REPORT.md)** for the assignment report (data creation, model details, results, and analysis). A LaTeX version following [ACL style](https://github.com/acl-org/acl-style-files) is in **report.tex**; to compile, place \texttt{acl.sty} (and any required files) from that repository in the same directory or your TeX path. It crawls or ingests documents, chunks and indexes them (dense + sparse), then retrieves relevant passages and generates answers with an LLM.

---

## Introduction to the Codebase

### High-level flow

1. **Raw data** → HTML under `data/raw/html/<domain>/`, PDFs in `data/raw/pdf/` (from crawler or manual placement).
2. **Parse & clean** → `run_pipeline.py` produces `data/parsed/docs.jsonl`.
3. **Chunk & embed** → `build_index.py` produces chunks (e.g. `data/parsed/rag/chunks.jsonl`) and embeddings (`.npy`).
4. **Retrieve & generate** → `run_rag.py` loads the index, retrieves top-k chunks (dense, sparse, or RRF), optionally reranks, then calls the reader LLM to produce an answer.

### Directory layout

| Path | Purpose |
|------|--------|
| **`data_source.md`** | Seed URL list for crawling. The pipeline reads this file to get URLs and infer allowed domains. Must exist before running the pipeline. |
| **`data_prep/`** | Data collection and corpus building. |
| `config.py` | All paths and crawl/parse options (`CrawlConfig`). |
| `extract_seed.py` | Loads `data_source.md` and extracts URLs with a regex. |
| `crawl.py` | Playwright-based crawler (Chromium); writes HTML/PDF and metadata. |
| `build_corpus.py` | Reads raw HTML/PDF → parse → clean → dedupe → write `docs.jsonl`. |
| `parse_html.py`, `parse_pdf.py` | Extract main text from HTML/PDF. |
| `clean.py`, `dedupe.py` | Normalize text, filter by length/language, exact dedupe. |
| **`rag_utils/`** | Chunking, retrieval, and generation. |
| `chunking.py` | Fixed-size (token) and semantic (embedding-based) chunking. |
| `dense_retriever.py` | Embed documents/queries, build FAISS index, search. |
| `sparse_retriever.py` | BM25 over chunk texts (rank-bm25). |
| `hybrid_retrieval.py` | Reciprocal Rank Fusion (RRF) of dense + sparse. |
| `reranker.py` | Cross-encoder re-ranking of retrieved chunks. |
| `reader.py` | Loads a HuggingFace causal/seq2seq model and generates the final answer from question + context. |
| **`scripts/`** | Entry points; run from repo root. |
| `run_pipeline.py` | Build corpus: (optionally) crawl, then parse/clean/dedupe → `docs.jsonl`. |
| `build_index.py` | Chunk docs, embed chunks, save `chunks.jsonl` + `embeddings_*.npy`. |
| `run_rag.py` | Run retrieval (dense / sparse / rrf) + reader; single question, batch file, or interactive. |
| `reclean_docs.py` | Re-run cleaning on existing `docs.jsonl` in place. |
| `txt_to_leaderboard_json.py` | Convert a one-question-per-line `.txt` file to JSON query list (e.g. for leaderboard/test). |
| **`data/`** | Created by the pipeline. `raw/` holds crawled HTML/PDF; `parsed/` holds `docs.jsonl` and `rag/` (chunks + embeddings). |
| **`system_outputs/`** | Default directory for RAG output JSON (id → answer). |

### Config and paths

All paths are defined in **`data_prep/config.py`** (`CrawlConfig`). Important defaults:

- **Input:** `data_source_path = "data_source.md"` (seed URLs).
- **Raw:** `raw_html_dir = "data/raw/html"`, `raw_pdf_dir = "data/raw/pdf"`.
- **Corpus:** `parsed_docs_path = "data/parsed/docs.jsonl"`.
- **RAG:** `rag_dir = "data/parsed/rag"`, `rag_chunks_path = "data/parsed/rag/chunks.jsonl"`, `rag_embeddings_path = "data/parsed/rag/embeddings.npy"` (actual filenames add `_<embed>_<chunking>` for embeddings).

---

## How to Run

### 1. Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- **FAISS:** The project uses `faiss-gpu-cu12` in `requirements.txt`. On machines without CUDA 12, replace it with `faiss-cpu` in `requirements.txt` before installing.
- **Crawler:** If you use the built-in crawler, install Playwright’s Chromium: `playwright install chromium`.

### 2. Seed URLs

Ensure **`data_source.md`** exists in the repo root and contains the URLs you want to use (e.g. Pittsburgh/CMU sites). The pipeline reads this file to get seed URLs and to infer allowed domains when crawling.

### 3. Build the document corpus

You need **`data/parsed/docs.jsonl`** before building the index.

**Option A – You already have raw files**

- Put HTML files under **`data/raw/html/<domain>/`** (e.g. `data/raw/html/en.wikipedia.org/page.html`).
- Put PDFs in **`data/raw/pdf/`**.
- Run:

```bash
python3 scripts/run_pipeline.py
```

Crawling is disabled by default in `run_pipeline.py`; only **build_corpus** runs (parse → clean → dedupe → write `docs.jsonl`).

**Option B – Use the crawler**

- In **`scripts/run_pipeline.py`**, uncomment the line that runs the crawler, e.g. `asyncio.run(crawl(cfg, seed_urls))`.
- Run:

```bash
python3 scripts/run_pipeline.py
```

This will crawl from URLs in `data_source.md`, then build the corpus as above.

**Re-clean without re-crawling**

To re-apply cleaning to an existing `docs.jsonl`:

```bash
python3 scripts/reclean_docs.py
# Or: python3 scripts/reclean_docs.py --input data/parsed/docs.jsonl
```

### 4. Build the RAG index

This step chunks the corpus and computes embeddings. The **embedding model** and **chunking strategy** must match what you use later in `run_rag.py`.

```bash
# Default: sentence-transformers, fixed chunking
python3 scripts/build_index.py

# Semantic chunking
python3 scripts/build_index.py --chunking semantic

# BAAI/bge-m3 (requires FlagEmbedding from requirements.txt)
python3 scripts/build_index.py --embed BAAI

# Force rebuild of chunks even if chunks.jsonl exists
python3 scripts/build_index.py --rebuild
```

Outputs go under `data/parsed/rag/`:

- `chunks.jsonl` or `chunks_semantic.jsonl`
- `embeddings_<embed>_<chunking>.npy` (and corresponding `ids_*.npy`)

### 5. Run the RAG pipeline

**Single question**

```bash
python3 scripts/run_rag.py --question "When was Carnegie Mellon University founded?"
```

**Batch from a JSON file**

Use a JSON file that contains a list of objects with `"question"` (or `"query"`) and `"id"` (or `"qid"`), e.g. `leaderboard_queries.json` or a test set:

```bash
python3 scripts/run_rag.py --queries-file leaderboard_queries.json --output system_outputs/system_output_1.json
```

**Retrieval mode and model choices**

- `--mode`: `dense` (FAISS only), `sparse` (BM25 only), or `rrf` (Reciprocal Rank Fusion; default).
- `--embed`: `sentence-transformers` (default) or `BAAI`. Must match the index you built.
- `--chunking`: `fixed` (default) or `semantic`. Must match the index.

Example:

```bash
python3 scripts/run_rag.py --queries-file leaderboard_queries.json \
  --mode rrf --embed sentence-transformers --chunking fixed \
  --output system_outputs/system_output_1.json
```

**Optional re-ranking**

```bash
python3 scripts/run_rag.py --queries-file leaderboard_queries.json --rerank --rerank-top-k 20 --output system_outputs/system_output_rerank.json
```

**Interactive mode**

If you omit both `--question` and `--queries-file`, the script reads questions from stdin in a loop.

**Output format**

The script writes a JSON object mapping question **id** (string) to a single **answer** string, e.g. `{"1": "Answer 1", "2": "Answer 2", ...}`. This is the format expected for leaderboard and test-set submissions.

### 6. Convert a question list to JSON

If you have a plain-text file with one question per line (e.g. `test_set_day_3.txt`):

```bash
python3 scripts/txt_to_leaderboard_json.py
```

By default it reads `test_set_day_3.txt` and writes `test_set_day_3_queries.json`. Edit the script to change input or output paths.

---

## Quick reference

| Step | Command |
|------|--------|
| Install deps | `pip install -r requirements.txt` (use `faiss-cpu` if no CUDA 12) |
| Build corpus | `python3 scripts/run_pipeline.py` (requires raw HTML/PDF or enabled crawler) |
| Build index | `python3 scripts/build_index.py` [ `--chunking semantic` \| `--embed BAAI` ] |
| Single Q | `python3 scripts/run_rag.py --question "..."` |
| Batch | `python3 scripts/run_rag.py --queries-file <path> --output system_outputs/out.json` |
| RRF + rerank | `python3 scripts/run_rag.py --queries-file <path> --mode rrf --rerank --output ...` |

Keep **`--embed`** and **`--chunking`** consistent between `build_index.py` and `run_rag.py`; otherwise you’ll get missing-file or shape mismatches.
