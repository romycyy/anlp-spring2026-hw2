import hashlib
import json
import os
from urllib.parse import urlparse, urljoin, urldefrag


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_url(url: str, base: str = None) -> str:
    if base:
        url = urljoin(base, url)
    url, _frag = urldefrag(url)
    return url.strip()


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def write_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_filename_from_url(url: str) -> str:
    # stable filename from hash
    return sha1(url)


def looks_like_pdf(url: str) -> bool:
    return url.lower().endswith(".pdf")


def looks_like_pdf_bytes(content: bytes) -> bool:
    """
    Cheap magic-byte sniff for PDFs.
    Real PDFs start with "%PDF-" (possibly after a few whitespace bytes).
    """
    if not content:
        return False
    head = content[:1024].lstrip()
    return head.startswith(b"%PDF-")


def looks_like_html_bytes(content: bytes) -> bool:
    if not content:
        return False
    head = content[:2048].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or head.startswith(
        b"<head"
    )
