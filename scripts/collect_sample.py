"""Phase 1 quick taste-test: small-scale news + price collection for one ticker.
Not the full collector -- just checking the pipeline shape works end-to-end.
"""
import os
import re
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

CODE = "005930"          # 삼성전자
COMPANY_NAME = "삼성전자"
YF_TICKER = "005930.KS"

NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
NAVER_HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": os.environ["NAVER_API_KEY_ID"],
    "X-NCP-APIGW-API-KEY": os.environ["NAVER_API_KEY"],
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").replace("&quot;", '"').replace("&amp;", "&")


def collect_news_sample(query: str, display: int = 100) -> pd.DataFrame:
    params = {"query": query, "display": display, "sort": "date"}
    resp = requests.get(NAVER_NEWS_URL, headers=NAVER_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    rows = [{
        "date": item.get("pubDate"),
        "title": _strip_html(item.get("title")),
        "description": _strip_html(item.get("description")),
        "link": item.get("link"),
    } for item in items]
    return pd.DataFrame(rows)


def collect_price_sample(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1mo", interval="1d", progress=False)
    df = df.reset_index()
    return df


if __name__ == "__main__":
    print(f"Collecting news sample for {COMPANY_NAME} ...")
    news_df = collect_news_sample(COMPANY_NAME)
    print(f"  -> {len(news_df)} headlines collected")
    news_path = "data/raw/news_sample.csv"
    news_df.to_csv(news_path, index=False, encoding="utf-8-sig")
    print(f"  saved to {news_path}")
    print(news_df.head(10).to_string())

    print(f"\nCollecting price sample for {YF_TICKER} ...")
    price_df = collect_price_sample(YF_TICKER)
    print(f"  -> {len(price_df)} rows collected")
    price_path = "data/raw/price_sample.csv"
    price_df.to_csv(price_path, index=False)
    print(f"  saved to {price_path}")
    print(price_df.tail(5).to_string())
