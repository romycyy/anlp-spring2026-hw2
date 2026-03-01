"""Closed-book Q&A: answer questions using only the base model (no retrieval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.utils import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Closed-book Q&A using base model only (no RAG)."
    )
    ap.add_argument(
        "--question", type=str, default=None, help="Ask a single question and exit."
    )
    ap.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="JSON file with list of {id, question} or {id, query} objects.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct",
        help="HuggingFace model name for the reader.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON output (used with --queries-file).",
    )
    args = ap.parse_args()

    try:
        from rag_utils.reader import answer_question
    except Exception as e:
        raise SystemExit(
            f"Failed to import reader.\n{repr(e)}\n"
            "Fix: pip install -U -r requirements.txt"
        ) from e

    def run_one(q: str) -> str:
        return answer_question(q, [], model_name=args.model, max_new_tokens=args.max_new_tokens)

    # Single question
    if args.question:
        ans = run_one(args.question.strip())
        print(f"\nAnswer:\n{ans}\n")
        return

    # Batch from file
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        items = [
            x for x in queries
            if x.get("question") or x.get("query")
        ]
        questions = [
            (x.get("question") or x.get("query", "")).strip()
            for x in items
        ]
        results = {}
        for i, item in enumerate(items):
            qid = str(item.get("id", item.get("qid", i)))
            q = questions[i]
            print(f"[{qid}] {q}")
            results[qid] = run_one(q)
            print(f"  -> {results[qid][:200]}{'...' if len(results[qid]) > 200 else ''}\n")

        out_path = args.output
        if not out_path:
            ensure_dir("system_outputs")
            out_path = "system_outputs/closedbook_qa_output.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} answers to {out_path}")
        return

    # Interactive
    print("Closed-book Q&A (no retrieval). Type a question and press Enter. Ctrl-D to quit.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except EOFError:
            print()
            break
        if not q:
            continue
        print(f"\nAnswer:\n{run_one(q)}\n")


if __name__ == "__main__":
    main()
