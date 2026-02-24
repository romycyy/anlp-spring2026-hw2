import os
from typing import Iterable, Dict, Set
from .utils import sha256_text


def exact_dedupe(docs: Iterable[Dict]):
    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"
    seen: Set[str] = set()
    for d in docs:
        h = sha256_text(d.get("text", ""))
        if h in seen:
            if debug:
                print(f"[dedupe] dropped_duplicate url={d.get('url')}")
            continue
        seen.add(h)
        d["text_sha256"] = h
        yield d
