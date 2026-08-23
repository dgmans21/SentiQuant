"""Phase 1 taste-test round 2: news + price for several major KR stocks."""
import os
import re
import time
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

STOCKS = {
    # 반도체
    "삼성전자": ("005930.KS", "반도체"),
    "SK하이닉스": ("000660.KS", "반도체"),
    # 2차전지
    "LG에너지솔루션": ("373220.KS", "2차전지"),
    "삼성SDI": ("006400.KS", "2차전지"),
    # 자동차
    "현대차": ("005380.KS", "자동차"),
    "기아": ("000270.KS", "자동차"),
    # 플랫폼/인터넷
    "NAVER": ("035420.KS", "플랫폼"),
    "카카오": ("035720.KS", "플랫폼"),
    # 바이오
    "삼성바이오로직스": ("207940.KS", "바이오"),
    "셀트리온": ("068270.KS", "바이오"),
    # 금융
    "KB금융": ("105560.KS", "금융"),
    "신한지주": ("055550.KS", "금융"),
    # 화학/에너지
    "LG화학": ("051910.KS", "화학에너지"),
    "한화에어로스페이스": ("012450.KS", "화학에너지"),
    # 철강/조선
    "POSCO홀딩스": ("005490.KS", "철강조선"),
    "HD현대중공업": ("329180.KS", "철강조선"),
    # 통신
    "SK텔레콤": ("017670.KS", "통신"),
    "KT": ("030200.KS", "통신"),
    # 유통/소비재
    "삼성물산": ("028260.KS", "유통소비재"),
    "LG생활건강": ("051900.KS", "유통소비재"),
}

NAVER_NEWS_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
NAVER_HEADERS = {
    "X-NCP-APIGW-API-KEY-ID": os.environ["NAVER_API_KEY_ID"],
    "X-NCP-APIGW-API-KEY": os.environ["NAVER_API_KEY"],
}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").replace("&quot;", '"').replace("&amp;", "&")


def collect_news(query: str, sector: str, display: int = 20) -> pd.DataFrame:
    params = {"query": query, "display": display, "sort": "date"}
    resp = requests.get(NAVER_NEWS_URL, headers=NAVER_HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return pd.DataFrame([{
        "stock": query,
        "sector": sector,
        "date": item.get("pubDate"),
        "title": _strip_html(item.get("title")),
        "description": _strip_html(item.get("description")),
        "link": item.get("link"),
    } for item in items])


if __name__ == "__main__":
    all_news = []
    for name, (ticker, sector) in STOCKS.items():
        df = collect_news(name, sector)
        print(f"{name} ({sector}): {len(df)} headlines")
        all_news.append(df)
        time.sleep(0.3)

    news_df = pd.concat(all_news, ignore_index=True)
    news_df.to_csv("data/raw/news_multi_sample.csv", index=False, encoding="utf-8-sig")
    print(f"total news rows: {len(news_df)} -> data/raw/news_multi_sample.csv")

    tickers = [ticker for ticker, _ in STOCKS.values()]
    price_df = yf.download(tickers, period="1mo", interval="1d", progress=False)
    price_df.to_csv("data/raw/price_multi_sample.csv")
    print(f"price rows: {len(price_df)} -> data/raw/price_multi_sample.csv")
