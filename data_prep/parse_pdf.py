from typing import Dict
from pypdf import PdfReader

from .utils import looks_like_pdf_bytes


def parse_pdf(path: str) -> Dict:
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
        if not looks_like_pdf_bytes(head):
            return {"title": "", "text": ""}
    except OSError:
        return {"title": "", "text": ""}

    try:
        reader = PdfReader(path)
    except Exception:
        # Covers truncated PDFs, HTML saved as .pdf, encrypted PDFs, etc.
        return {"title": "", "text": ""}
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t)
    text = "\n".join(texts).strip()
    # title: try metadata
    title = ""
    try:
        md = reader.metadata
        if md and getattr(md, "title", None):
            title = str(md.title) or ""
    except Exception:
        pass
    return {"title": title, "text": text}
