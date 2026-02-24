import asyncio
import os
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
import urllib.robotparser

from playwright.async_api import async_playwright

from .config import CrawlConfig
from .utils import (
    ensure_dir,
    normalize_url,
    get_domain,
    is_http_url,
    safe_filename_from_url,
    write_jsonl,
    looks_like_pdf,
    looks_like_pdf_bytes,
    looks_like_html_bytes,
)


def _same_or_subdomain(domain: str, allowed: List[str]) -> bool:
    domain = domain.lower()
    for a in allowed:
        a = a.lower()
        if domain == a or domain.endswith("." + a):
            return True
    return False


def _allowed_by_prefix(url: str, prefixes: List[str]) -> bool:
    if not prefixes:
        return True
    return any(url.startswith(p) for p in prefixes)


def _denied(url: str, deny_res) -> bool:
    for r in deny_res:
        if r.search(url):
            return True
    return False


class RobotsCache:
    def __init__(self):
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def can_fetch(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots is unreachable, be conservative but not blocking
                pass
            self._cache[base] = rp
        rp = self._cache[base]
        try:
            return rp.can_fetch(user_agent, url)
        except Exception:
            return True


async def fetch_with_chromium(
    page, url: str, cfg: CrawlConfig
) -> Tuple[Optional[bytes], Optional[str], Dict]:
    """
    Returns: (content_bytes, content_type, meta)
    """
    meta = {"final_url": url, "status": None, "content_type": None}
    try:
        resp = await page.goto(
            url, wait_until=cfg.wait_until, timeout=cfg.nav_timeout_ms
        )
        if resp is None:
            return None, None, meta
        meta["status"] = resp.status
        meta["final_url"] = page.url
        headers = resp.headers
        ctype = headers.get("content-type", "")
        if ";" in ctype:
            ctype = ctype.split(";", 1)[0].strip().lower()
        meta["content_type"] = ctype or None
        meta["headers"] = headers

        # If PDF, fetch body directly
        body = await resp.body()
        return body, ctype, meta
    except Exception as e:
        # Some downloads (notably PDFs) can fail with net::ERR_ABORTED in page navigation.
        # Fallback to a direct request fetch when possible.
        err = repr(e)
        meta["error"] = err
        try:
            if "ERR_ABORTED" in err or "net::ERR_ABORTED" in err:
                r = await page.request.get(url, timeout=cfg.nav_timeout_ms)
                meta["status"] = r.status
                meta["final_url"] = str(r.url)
                headers = r.headers
                ctype = headers.get("content-type", "")
                if ";" in ctype:
                    ctype = ctype.split(";", 1)[0].strip().lower()
                meta["content_type"] = ctype or None
                meta["headers"] = headers
                meta["fallback"] = "page.request.get"
                body = await r.body()
                return body, ctype, meta
        except Exception as e2:
            meta["fallback_error"] = repr(e2)
        return None, None, meta


async def crawl(cfg: CrawlConfig, seed_urls: List[str]) -> None:
    ensure_dir(cfg.raw_html_dir)
    ensure_dir(cfg.raw_pdf_dir)
    ensure_dir(os.path.dirname(cfg.raw_meta_path))

    debug = os.getenv("PIPELINE_DEBUG", "0") == "1"
    deny_res = cfg.compiled_deny()
    robots = RobotsCache()

    # Frontier: (url, depth, source)
    q: asyncio.Queue = asyncio.Queue()
    for u in seed_urls:
        await q.put((normalize_url(u), 0, get_domain(u)))

    seen: Set[str] = set()
    pages_fetched = 0

    # Per-domain delay tracking
    last_fetch: Dict[str, float] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=cfg.headless)
        context = await browser.new_context(user_agent=cfg.user_agent)
        # Reduce some bot friction
        await context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """
        )

        sem = asyncio.Semaphore(cfg.concurrency)

        async def worker(worker_id: int):
            nonlocal pages_fetched
            page = await context.new_page()
            while True:
                try:
                    url, depth, source = await q.get()
                except asyncio.CancelledError:
                    break
                try:
                    if pages_fetched >= cfg.max_pages:
                        if debug:
                            print(f"[crawl skip] max_pages url={url}")
                        q.task_done()
                        continue

                    url = normalize_url(url)
                    if not is_http_url(url):
                        if debug:
                            print(f"[crawl skip] non_http url={url}")
                        q.task_done()
                        continue
                    if url in seen:
                        if debug:
                            print(f"[crawl skip] already_seen url={url}")
                        q.task_done()
                        continue

                    domain = get_domain(url)
                    if cfg.allow_domains and not _same_or_subdomain(
                        domain, cfg.allow_domains
                    ):
                        if debug:
                            print(
                                f"[crawl skip] domain_not_allowed url={url} domain={domain}"
                            )
                        q.task_done()
                        continue
                    if not _allowed_by_prefix(url, cfg.allow_url_prefixes):
                        if debug:
                            print(f"[crawl skip] prefix_not_allowed url={url}")
                        q.task_done()
                        continue
                    if _denied(url, deny_res):
                        if debug:
                            print(f"[crawl skip] denied_pattern url={url}")
                        q.task_done()
                        continue

                    # robots.txt check
                    if cfg.obey_robots and not robots.can_fetch(url, cfg.user_agent):
                        if debug:
                            print(f"[crawl skip] robots_disallow url={url}")
                        q.task_done()
                        continue

                    # per-domain delay
                    now = time.time()
                    lf = last_fetch.get(domain, 0.0)
                    wait = cfg.per_domain_delay_s - (now - lf)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    last_fetch[domain] = time.time()

                    async with sem:
                        content, ctype, meta = await fetch_with_chromium(page, url, cfg)

                    seen.add(url)
                    pages_fetched += 1

                    # Save raw + meta
                    meta_obj = {
                        "url": url,
                        "source": source,
                        "depth": depth,
                        "fetched_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "meta": meta,
                    }

                    if content is None or ctype is None:
                        if debug:
                            err = meta.get("error")
                            print(
                                "[crawl fetch_failed] "
                                f"url={url} final_url={meta.get('final_url')} "
                                f"status={meta.get('status')} ctype={meta.get('content_type')} "
                                f"error={err}"
                            )
                        write_jsonl(cfg.raw_meta_path, meta_obj)
                        q.task_done()
                        continue

                    # PDF handling: don't trust URL suffix alone; sniff bytes.
                    maybe_pdf = ctype.startswith("application/pdf") or looks_like_pdf(url)
                    is_pdf = maybe_pdf and looks_like_pdf_bytes(content)
                    meta_obj["meta"]["sniffed_is_pdf"] = bool(is_pdf)
                    if maybe_pdf and not is_pdf:
                        meta_obj["meta"]["sniff_mismatch"] = True

                    if is_pdf:
                        fn = safe_filename_from_url(url) + ".pdf"
                        path = os.path.join(cfg.raw_pdf_dir, fn)
                        with open(path, "wb") as f:
                            f.write(content)
                        meta_obj["saved_path"] = path
                        meta_obj["kind"] = "pdf"
                        write_jsonl(cfg.raw_meta_path, meta_obj)
                        q.task_done()
                        continue

                    if not ctype.startswith("text/html"):
                        # Some servers lie about content-type; save HTML when we can.
                        if looks_like_html_bytes(content):
                            ctype = "text/html"
                            meta_obj["meta"]["content_type_overridden"] = ctype
                        else:
                            if debug:
                                print(
                                    "[crawl skip] non_html_non_pdf "
                                    f"url={url} ctype={ctype} status={meta.get('status')}"
                                )
                            meta_obj["kind"] = "other"
                            write_jsonl(cfg.raw_meta_path, meta_obj)
                            q.task_done()
                            continue

                    if cfg.save_rendered_html:
                        try:
                            html = await page.content()
                            meta_obj["meta"]["saved_rendered_html"] = True
                        except Exception:
                            html = content.decode("utf-8", errors="ignore")
                            meta_obj["meta"]["saved_rendered_html"] = False
                    else:
                        html = content.decode("utf-8", errors="ignore")
                    fn = safe_filename_from_url(url) + ".html"
                    path = os.path.join(cfg.raw_html_dir, fn)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(html)
                    meta_obj["saved_path"] = path
                    meta_obj["kind"] = "html"
                    write_jsonl(cfg.raw_meta_path, meta_obj)

                    # Link discovery
                    if depth < cfg.max_depth:
                        # Use DOM links (more accurate than parsing raw html sometimes)
                        try:
                            hrefs = await page.eval_on_selector_all(
                                "a[href]", "els => els.map(e => e.getAttribute('href'))"
                            )
                        except Exception:
                            hrefs = []

                        # Normalize + enqueue
                        for h in hrefs or []:
                            if not h:
                                continue
                            nu = normalize_url(h, base=meta.get("final_url") or url)
                            if not is_http_url(nu):
                                continue
                            if _denied(nu, deny_res):
                                continue
                            # stay scoped
                            if cfg.allow_domains and not _same_or_subdomain(
                                get_domain(nu), cfg.allow_domains
                            ):
                                continue
                            if not _allowed_by_prefix(nu, cfg.allow_url_prefixes):
                                continue
                            if nu not in seen:
                                await q.put((nu, depth + 1, source))

                    q.task_done()

                except Exception:
                    q.task_done()
                    continue

            await page.close()

        workers = [asyncio.create_task(worker(i)) for i in range(cfg.concurrency)]
        await q.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        await context.close()
        await browser.close()
