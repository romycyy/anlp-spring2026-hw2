#!/usr/bin/env python3
"""Convert test_set_day_3.txt to leaderboard_queries.json style."""

import json
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parent.parent
    txt_path = repo / "test_set_day_3.txt"
    out_path = repo / "test_set_day_3_queries.json"

    queries = []
    with open(txt_path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            q = line.strip()
            if not q:
                continue
            # Strip surrounding quotes and unescape doubled quotes
            if q.startswith('"') and q.endswith('"'):
                q = q[1:-1].replace('""', '"')
            queries.append({"question": q, "id": str(i)})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(queries)} queries to {out_path}")


if __name__ == "__main__":
    main()
