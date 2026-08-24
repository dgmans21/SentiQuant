"""Phase 1 backfill: pull up to the API's 1000-result cap per stock.

NAVER news search allows `start` up to 1000, so display=100 x 10 pages
is the maximum retrievable per query. Runs once; future runs should just
append newly published articles (daily incremental), not re-backfill.
"""
import time
import pandas as pd
from collect_multi_sample import STOCKS, NAVER_NEWS_URL, NAVER_HEADERS, _strip_html
import requests

MAX_START = 1000
PAGE_SIZE = 100


def _fetch_page(query: str, start: int, retries: int = 3):
    params = {"query": query, "display": PAGE_SIZE, "start": start, "sort": "date"}
    for attempt in range(retries):
        resp = requests.get(NAVER_NEWS_URL, headers=NAVER_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if len(items) == PAGE_SIZE or start + PAGE_SIZE > MAX_START:
            return items
        # short page before the true end -- likely a transient truncation, retry
        time.sleep(0.5 * (attempt + 1))
    return items


def collect_news_full(query: str, sector: str) -> pd.DataFrame:
    rows = []
    for start in range(1, MAX_START + 1, PAGE_SIZE):
        items = _fetch_page(query, start)
        if not items:
            break
        rows.extend([{
            "stock": query,
            "sector": sector,
            "date": item.get("pubDate"),
            "title": _strip_html(item.get("title")),
            "description": _strip_html(item.get("description")),
            "link": item.get("link"),
        } for item in items])
        if len(items) < PAGE_SIZE:
            break
        time.sleep(0.3)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os

    all_news = []
    total_calls = 0
    for name, (ticker, sector) in STOCKS.items():
        df = collect_news_full(name, sector)
        calls = -(-len(df) // PAGE_SIZE) or 1
        total_calls += calls
        print(f"{name} ({sector}): {len(df)} articles")
        all_news.append(df)
        time.sleep(0.2)

    news_df = pd.concat(all_news, ignore_index=True)

    out_path = "data/raw/news_backfill.csv"
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        news_df = pd.concat([existing, news_df], ignore_index=True)

    news_df = news_df.drop_duplicates(subset=["link"])
    news_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\ntotal rows (deduped, merged with existing): {len(news_df)}")
    print(f"approx API calls used this run: {total_calls}")
    print(f"saved to {out_path}")
