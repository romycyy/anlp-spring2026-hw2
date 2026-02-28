import os
from typing import Iterable

from tqdm import tqdm

from .config import CrawlConfig
from .utils import ensure_dir, write_jsonl
from .parse_html import extract_main_text
from .parse_pdf import parse_pdf
from .clean import normalize_text, remove_boilerplate_lines, remove_wikipedia_artifacts, language_ok, detect_language
from .dedupe import exact_dedupe


def build_docs_from_raw(cfg: CrawlConfig) -> Iterable[dict]:
    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"

    for domain in os.listdir(cfg.raw_html_dir):
        domain_dir = os.path.join(cfg.raw_html_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        for filename in os.listdir(domain_dir):
            if not filename.endswith((".html", ".htm")):
                continue
            path = os.path.join(domain_dir, filename)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    html = f.read()
                parsed = extract_main_text(html, url=None)
                text = parsed.get("text", "")
            except Exception as e:
                if debug:
                    print(f"[html parse error] path={path} error={repr(e)}")
                continue
            if debug:
                print(f"[html] path={path} text_len={len(text)}")
            yield {"text": text}

    for filename in os.listdir(cfg.raw_pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(cfg.raw_pdf_dir, filename)
        try:
            parsed = parse_pdf(path)
            text = parsed.get("text", "")
        except Exception as e:
            if debug:
                print(f"[pdf parse error] path={path} error={repr(e)}")
            continue
        if debug:
            print(f"[pdf] path={path} text_len={len(text)}")
        yield {"text": text}


def clean_filter_docs(cfg: CrawlConfig, docs: Iterable[dict]) -> Iterable[dict]:
    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"
    for d in docs:
        text = d.get("text", "")
        text = remove_boilerplate_lines(text)
        text = remove_wikipedia_artifacts(text)
        text = normalize_text(text)
        if len(text) < cfg.min_text_chars:
            if debug:
                print(f"[doc filter] too_short len={len(text)} min={cfg.min_text_chars}")
            continue
        if not language_ok(text[:2000], cfg.language):
            if debug:
                detected = detect_language(text[:2000])
                print(f"[doc filter] language_mismatch detected={detected} expected={cfg.language}")
            continue
        yield {"text": text}


def build_corpus(cfg: CrawlConfig) -> None:
    ensure_dir(cfg.parsed_dir)
    if os.path.exists(cfg.parsed_docs_path):
        os.remove(cfg.parsed_docs_path)

    raw_docs = build_docs_from_raw(cfg)
    cleaned = clean_filter_docs(cfg, raw_docs)
    deduped = exact_dedupe(cleaned)

    n = 0
    for d in tqdm(deduped, desc="Writing docs.jsonl"):
        write_jsonl(cfg.parsed_docs_path, {"text": d["text"]})
        n += 1

    print(f"✅ Wrote {n} docs to {cfg.parsed_docs_path}")
