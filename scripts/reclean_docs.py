"""Re-clean an existing docs.jsonl without re-crawling.

Applies `remove_wikipedia_artifacts` (and all other clean.py filters) to
the already-parsed corpus, overwriting docs.jsonl in place.

Usage:
  python3 scripts/reclean_docs.py
  python3 scripts/reclean_docs.py --input data/parsed/docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.clean import (  # noqa: E402
    normalize_text,
    remove_boilerplate_lines,
    remove_wikipedia_artifacts,
)
from data_prep.config import CrawlConfig  # noqa: E402


def reclean(input_path: str) -> None:
    tmp_path = input_path + ".tmp"

    n_in = n_out = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            n_in += 1

            text = remove_boilerplate_lines(text)
            text = remove_wikipedia_artifacts(text)
            text = normalize_text(text)

            if not text:
                continue

            rec["text"] = text
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    os.replace(tmp_path, input_path)
    print(f"Re-cleaned {n_in} docs → {n_out} docs saved to {input_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-clean docs.jsonl in place.")
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to docs.jsonl (defaults to CrawlConfig.parsed_docs_path).",
    )
    args = ap.parse_args()

    input_path = args.input or CrawlConfig().parsed_docs_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"docs.jsonl not found at {input_path}")

    reclean(input_path)


if __name__ == "__main__":
    main()
