"""Daily curated news collection for high-volume stocks (e.g. 삼성전자, SK하이닉스).

Unlike collect_news_backfill.py (grabs everything, hits the same-day wall for
high-volume stocks), this pulls a larger raw pool, drops near-duplicate
wire-reprints by title similarity, caps per-outlet share, and keeps a target
count per stock. Run this once a day -- each run appends today's curated
picks (deduped by link against everything collected so far) so a real
day-by-day spread builds up over time.

Usage:
    python scripts/collect_news_curated.py                  # default targets
    python scripts/collect_news_curated.py 삼성전자 SK하이닉스 카카오
"""
import os
import re
import sys
import time
import difflib
from collections import Counter
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

from collect_multi_sample import STOCKS, NAVER_NEWS_URL, NAVER_HEADERS, _strip_html

load_dotenv()

DEFAULT_TARGETS = ["삼성전자", "SK하이닉스"]
POOL_SIZE = 300          # raw articles fetched before curation
PAGE_SIZE = 100
TARGET_COUNT = 80        # curated articles kept per stock per run
TITLE_SIM_THRESHOLD = 0.6
MAX_PER_SOURCE = 3
OUT_PATH = "data/raw/news_daily_curated.csv"


def _source_from_link(link: str) -> str:
    m = re.search(r"https?://([^/]+)", link or "")
    return m.group(1) if m else "unknown"


def _normalize_title(title: str) -> str:
    t = re.sub(r"^\[[^\]]*\]\s*", "", title or "")  # strip leading [단독] 등
    t = re.sub(r"\s*[-–]\s*[^-–]{2,20}$", "", t)      # strip trailing " - 언론사"
    return t.strip()


def fetch_pool(query: str, pool_size: int = POOL_SIZE) -> list:
    items = []
    seen_links = set()
    for start in range(1, pool_size + 1, PAGE_SIZE):
        params = {"query": query, "display": PAGE_SIZE, "start": start, "sort": "date"}
        resp = requests.get(NAVER_NEWS_URL, headers=NAVER_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            link = item.get("link")
            if link not in seen_links:
                seen_links.add(link)
                items.append(item)
        time.sleep(0.3)
    return items


def curate(items: list, target: int = TARGET_COUNT, max_per_source: int = MAX_PER_SOURCE) -> list:
    kept = []
    kept_norms = []
    source_counts = Counter()
    for item in items:
        norm = _normalize_title(_strip_html(item.get("title")))
        source = _source_from_link(item.get("link"))

        if source_counts[source] >= max_per_source:
            continue
        if any(difflib.SequenceMatcher(None, norm, k).ratio() >= TITLE_SIM_THRESHOLD for k in kept_norms):
            continue

        kept.append(item)
        kept_norms.append(norm)
        source_counts[source] += 1
        if len(kept) >= target:
            break
    return kept


def collect_curated(query: str, sector: str) -> pd.DataFrame:
    pool = fetch_pool(query)
    curated = curate(pool)
    today = date.today().isoformat()
    return pd.DataFrame([{
        "collected_date": today,
        "stock": query,
        "sector": sector,
        "date": item.get("pubDate"),
        "title": _strip_html(item.get("title")),
        "description": _strip_html(item.get("description")),
        "source": _source_from_link(item.get("link")),
        "link": item.get("link"),
    } for item in curated]), len(pool)


if __name__ == "__main__":
    targets = sys.argv[1:] or DEFAULT_TARGETS

    frames = []
    for name in targets:
        if name not in STOCKS:
            print(f"skip: '{name}' not in STOCKS list (collect_multi_sample.py)")
            continue
        _, sector = STOCKS[name]
        df, pool_n = collect_curated(name, sector)
        print(f"{name} ({sector}): pool={pool_n} -> curated={len(df)}")
        frames.append(df)

    new_df = pd.concat(frames, ignore_index=True)

    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        new_df = pd.concat([existing, new_df], ignore_index=True)

    new_df = new_df.drop_duplicates(subset=["link"])
    new_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\ntotal rows (deduped, all runs so far): {len(new_df)}")
    print(f"saved to {OUT_PATH}")
