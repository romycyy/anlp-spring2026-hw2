from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class CrawlConfig:
    # Inputs
    data_source_path: str = "data_source.md"

    # Crawl controls
    max_pages: int = 2000
    max_depth: int = 1
    concurrency: int = 3  # chromium is heavy; keep small
    per_domain_delay_s: float = 1.0
    nav_timeout_ms: int = 30000
    wait_until: str = "domcontentloaded"  # or "networkidle"
    obey_robots: bool = True
    save_rendered_html: bool = True

    # Scope control
    allow_domains: List[str] = field(
        default_factory=list
    )  # if empty, inferred from README URLs
    allow_url_prefixes: List[str] = field(default_factory=list)
    deny_url_patterns: List[str] = field(
        default_factory=lambda: [
            r"/logout",
            r"/signout",
            r"/signin",
            r"/login",
            r"\?share=",
            r"\?utm_",
            r"/search",
            r"/tags?",
            r"#",  # fragments handled separately
        ]
    )

    # Data dirs
    data_dir: str = "data"
    raw_html_dir: str = "data/raw/html"
    raw_pdf_dir: str = "data/raw/pdf"
    raw_meta_path: str = "data/raw/meta.jsonl"

    parsed_dir: str = "data/parsed"
    parsed_docs_path: str = "data/parsed/docs.jsonl"

    # RAG artifacts (built from parsed_docs_path)
    rag_dir: str = "data/parsed/rag"
    rag_chunks_path: str = "data/parsed/rag/chunks.jsonl"
    rag_embeddings_path: str = "data/parsed/rag/embeddings.npy"

    # Parsing/filtering
    min_text_chars: int = 300
    language: str = "en"  # set None to disable language filter

    # Browser
    headless: bool = False
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def compiled_deny(self):
        return [re.compile(pat, re.IGNORECASE) for pat in self.deny_url_patterns]
