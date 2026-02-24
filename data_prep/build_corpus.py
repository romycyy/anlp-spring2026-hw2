import os
from typing import Dict, Iterable

from tqdm import tqdm

from .config import CrawlConfig
from .utils import ensure_dir, read_jsonl, write_jsonl, sha1
from .parse_html import extract_main_text
from .parse_pdf import parse_pdf
from .clean import normalize_text, remove_boilerplate_lines, language_ok, detect_language
from .dedupe import exact_dedupe


def build_docs_from_raw(cfg: CrawlConfig) -> Iterable[Dict]:
    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"
    for rec in read_jsonl(cfg.raw_meta_path):
        kind = rec.get("kind")
        url = rec.get("url")
        source = rec.get("source") or "web"
        saved_path = rec.get("saved_path")
        fetched_at = rec.get("fetched_at")

        if not saved_path or not os.path.exists(saved_path):
            if debug:
                meta = rec.get("meta", {}) or {}
                print(
                    "[raw->doc skip] missing_raw_file "
                    f"url={url} kind={kind} saved_path={saved_path} "
                    f"status={meta.get('status')} ctype={meta.get('content_type')} "
                    f"error={meta.get('error')}"
                )
            continue

        if kind == "html":
            with open(saved_path, "r", encoding="utf-8", errors="ignore") as f:
                html = f.read()
            parsed = extract_main_text(html, url)
            title, text = parsed.get("title", ""), parsed.get("text", "")
            date = parsed.get("date")
            if debug:
                print(
                    "[raw->doc html] "
                    f"url={url} title_len={len(title or '')} text_len={len(text or '')} "
                    f"raw_path={saved_path}"
                )
            yield {
                "doc_id": sha1(url),
                "source": source,
                "url": url,
                "title": title or "",
                "text": text or "",
                "content_type": "text/html",
                "fetched_at": fetched_at,
                "metadata": {
                    "date": date,
                    "raw_path": saved_path,
                    "final_url": rec.get("meta", {}).get("final_url"),
                    "status": rec.get("meta", {}).get("status"),
                },
            }

        elif kind == "pdf":
            try:
                parsed = parse_pdf(saved_path)
            except Exception as e:
                if debug:
                    print(
                        "[raw->doc pdf] parse_error "
                        f"url={url} raw_path={saved_path} error={repr(e)}"
                    )
                parsed = {"title": "", "text": ""}
            title, text = parsed.get("title", ""), parsed.get("text", "")
            if debug:
                print(
                    "[raw->doc pdf] "
                    f"url={url} title_len={len(title or '')} text_len={len(text or '')} "
                    f"raw_path={saved_path}"
                )
            yield {
                "doc_id": sha1(url),
                "source": source,
                "url": url,
                "title": title or "",
                "text": text or "",
                "content_type": "application/pdf",
                "fetched_at": fetched_at,
                "metadata": {"raw_path": saved_path},
            }
        else:
            if debug:
                print(f"[raw->doc skip] unsupported_kind url={url} kind={kind}")


def clean_filter_docs(cfg: CrawlConfig, docs: Iterable[Dict]) -> Iterable[Dict]:
    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"
    for d in docs:
        url = d.get("url")
        text = d.get("text", "")
        text = remove_boilerplate_lines(text)
        text = normalize_text(text)
        if len(text) < cfg.min_text_chars:
            if debug:
                print(
                    "[doc filter] too_short "
                    f"url={url} len={len(text)} min={cfg.min_text_chars}"
                )
            continue
        if not language_ok(text[:2000], cfg.language):
            if debug:
                detected = detect_language(text[:2000])
                print(
                    "[doc filter] language_mismatch "
                    f"url={url} detected={detected} expected={cfg.language}"
                )
            continue
        d["text"] = text
        d["title"] = normalize_text(d.get("title", ""))
        yield d


def build_corpus(cfg: CrawlConfig) -> None:
    ensure_dir(cfg.parsed_dir)
    # reset output
    if os.path.exists(cfg.parsed_docs_path):
        os.remove(cfg.parsed_docs_path)

    raw_docs = build_docs_from_raw(cfg)
    cleaned = clean_filter_docs(cfg, raw_docs)
    deduped = exact_dedupe(cleaned)

    n = 0
    for d in tqdm(deduped, desc="Writing docs.jsonl"):
        write_jsonl(cfg.parsed_docs_path, d)
        n += 1

    print(f"✅ Wrote {n} docs to {cfg.parsed_docs_path}")
