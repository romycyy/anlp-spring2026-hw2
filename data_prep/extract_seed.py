import re
from typing import List
from urllib.parse import urlparse

from .utils import is_http_url, normalize_url, get_domain

URL_RE = re.compile(r"(https?://[^\s\)\]\}>'\"`]+)")


def extract_urls_from_readme(readme_text: str) -> List[str]:
    urls = URL_RE.findall(readme_text)
    cleaned = []
    for u in urls:
        u = u.strip().strip(".,;")
        if is_http_url(u):
            cleaned.append(normalize_url(u))
    # unique but keep order
    seen = set()
    out = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def infer_domains(urls: List[str]) -> List[str]:
    domains = []
    seen = set()
    for u in urls:
        d = get_domain(u)
        if d and d not in seen:
            seen.add(d)
            domains.append(d)
    return domains


def load_readme(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
