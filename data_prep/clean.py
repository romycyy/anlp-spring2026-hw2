import re
from typing import Optional
from langdetect import detect, LangDetectException

_WS_RE = re.compile(r"\s+")
_BOILERPLATE_LINE_RE = re.compile(
    r"(cookie|privacy|terms|sign in|log in|subscribe|accept all|all rights reserved)",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def remove_boilerplate_lines(text: str) -> str:
    # conservative line filtering (keep long lines)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept = []
    for ln in lines:
        if len(ln) <= 120 and _BOILERPLATE_LINE_RE.search(ln):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def language_ok(text: str, lang: Optional[str]) -> bool:
    if not lang:
        return True
    d = detect_language(text)
    return d == lang


def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except LangDetectException:
        return None
