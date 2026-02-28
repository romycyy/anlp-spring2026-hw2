import asyncio
import os
import sys
from pathlib import Path

# Allow running as `python3 scripts/run_pipeline.py` from anywhere.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.config import CrawlConfig  # noqa: E402
from data_prep.extract_seed import load_readme, extract_urls_from_readme, infer_domains  # noqa: E402
from data_prep.crawl import crawl  # noqa: E402
from data_prep.build_corpus import build_corpus  # noqa: E402


def main():
    cfg = CrawlConfig()

    if not os.path.exists(cfg.data_source_path):
        raise FileNotFoundError(
            f"Could not find {cfg.data_source_path} in current directory."
        )

    readme = load_readme(cfg.data_source_path)
    seed_urls = extract_urls_from_readme(readme)
    if not seed_urls:
        raise ValueError("No URLs found in README.md. Add URLs or adjust regex.")

    if not cfg.allow_domains:
        cfg.allow_domains = infer_domains(seed_urls)

    print(f"🌱 Found {len(seed_urls)} seed URLs from README.")
    print(f"🔒 Allowed domains: {cfg.allow_domains}")

    # 1) Crawl with Chromium
    asyncio.run(crawl(cfg, seed_urls))

    # 2) Parse + clean + dedupe => docs.jsonl
    build_corpus(cfg)


if __name__ == "__main__":
    main()
