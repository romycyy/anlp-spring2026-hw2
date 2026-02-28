import re
from typing import Optional
from langdetect import detect, LangDetectException

_WS_RE = re.compile(r"\s+")
_BOILERPLATE_LINE_RE = re.compile(
    r"(cookie|privacy|terms|sign in|log in|subscribe|accept all|all rights reserved)",
    re.IGNORECASE,
)

# Wikipedia-specific patterns
_WIKI_INLINE_CITE_RE = re.compile(r"\[\d+\]")
_WIKI_SPECIAL_CITE_RE = re.compile(
    r"\[(?:citation needed|note \d+|nb \d+|clarification needed|dubious)\]",
    re.IGNORECASE,
)
_WIKI_EDIT_RE = re.compile(r"\s*\[edit\]")

# Section headers that signal the end of article body content.
# Matches both line-level (newline-bounded) and inline (after normalize_text).
_WIKI_TRAILING_SECTION_LINE_RE = re.compile(
    r"^\s*(?:See also|References?|Notes?|Citations?|Bibliography"
    r"|Further reading|External links?|Sources?)\s*(?:\[edit\])?\s*$",
    re.IGNORECASE,
)
_WIKI_TRAILING_SECTION_INLINE_RE = re.compile(
    r"\s+(?:See also|References?|Notes?|Citations?|Bibliography"
    r"|Further reading|External links?|Sources?)"
    r"(?:\s*\[edit\])?(?=\s*[-\^]|\s*$)",
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


def remove_wikipedia_artifacts(text: str) -> str:
    """Strip Wikipedia-specific noise from document text.

    Removes:
    - Trailing sections (References, See also, Notes, Bibliography,
      Further reading, External links, Citations) and everything after them.
    - Inline numeric citations: [1], [23], etc.
    - Editorial markers: [citation needed], [edit], etc.

    Works on both pre-normalized text (newlines present) and single-line
    text that has already passed through ``normalize_text``.
    """
    if "\n" in text:
        # Line-based pass: stop at the first trailing-section heading.
        lines = text.splitlines()
        kept = []
        for ln in lines:
            if _WIKI_TRAILING_SECTION_LINE_RE.match(ln):
                break
            kept.append(ln)
        text = "\n".join(kept)
    else:
        # Single-line (post-normalize) pass.
        m = _WIKI_TRAILING_SECTION_INLINE_RE.search(text)
        if m:
            text = text[: m.start()]

    text = _WIKI_INLINE_CITE_RE.sub("", text)
    text = _WIKI_SPECIAL_CITE_RE.sub("", text)
    text = _WIKI_EDIT_RE.sub("", text)
    return text.strip()


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
