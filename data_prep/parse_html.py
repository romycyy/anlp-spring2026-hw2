"""Extract main text and metadata from HTML for the knowledge corpus."""
from typing import Any, Dict, Optional

from trafilatura import extract
from trafilatura import extract_metadata


def extract_main_text(html: str, url: Optional[str] = None) -> Dict[str, Any]:
    """Extract main content, title, and optional date from HTML.
    Returns dict with keys: title (str), text (str), date (str or None).
    """
    text = extract(html, url=url) or ""
    meta = extract_metadata(html)
    title = ""
    date = None
    if meta:
        if getattr(meta, "title", None):
            title = str(meta.title).strip()
        if getattr(meta, "date", None):
            date = str(meta.date)
    return {"title": title, "text": (text or "").strip(), "date": date}
