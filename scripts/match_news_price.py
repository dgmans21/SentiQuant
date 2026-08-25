"""Match each news item to the next trading day's close-to-close return.

Rule: for a news item published on calendar date D, find the first trading
day strictly after D (target_day) and the trading day immediately before it
(base_day) in the price series. Label value = pct change from base_day's
close to target_day's close. This deliberately skips same-day price action
so the label represents "how the market moved after this news came out",
not a same-day mix of before/after-news price movement.
"""
import pandas as pd

from collect_multi_sample import STOCKS

NEWS_PATH = "data/raw/news_backfill.csv"
CURATED_PATH = "data/raw/news_daily_curated.csv"  # 삼성전자/SK하이닉스 일일 큐레이션
PRICE_PATH = "data/raw/price_history.csv"
OUT_PATH = "data/processed/news_price_matched.csv"


def load_news() -> pd.DataFrame:
    import os

    cols = ["stock", "sector", "date", "title", "description", "link"]
    news = pd.read_csv(NEWS_PATH)[cols]
    if os.path.exists(CURATED_PATH):
        curated = pd.read_csv(CURATED_PATH)[cols]
        news = pd.concat([news, curated], ignore_index=True)
    return news.drop_duplicates(subset=["link"])


def load_price_history() -> pd.DataFrame:
    df = pd.read_csv(PRICE_PATH, index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def match_one(news_date: pd.Timestamp, close_series: pd.Series):
    later = close_series.index[close_series.index > news_date]
    if len(later) == 0:
        return None  # no trading day yet after this news (too recent)
    target_day = later[0]
    pos = close_series.index.get_loc(target_day)
    if pos == 0:
        return None  # no prior trading day to use as base
    base_day = close_series.index[pos - 1]
    base_close = close_series.loc[base_day]
    target_close = close_series.loc[target_day]
    return_pct = (target_close - base_close) / base_close * 100
    return base_day, target_day, return_pct


if __name__ == "__main__":
    import os

    news = load_news()
    news["news_date"] = pd.to_datetime(news["date"], format="mixed", utc=True).dt.tz_convert(
        "Asia/Seoul"
    ).dt.tz_localize(None).dt.normalize()

    prices = load_price_history()
    ticker_by_stock = {name: ticker for name, (ticker, _) in STOCKS.items()}

    rows = []
    skipped_no_price = 0
    skipped_too_recent = 0
    for _, row in news.iterrows():
        ticker = ticker_by_stock.get(row["stock"])
        if ticker is None or ticker not in prices.columns:
            skipped_no_price += 1
            continue
        result = match_one(row["news_date"], prices[ticker].dropna())
        if result is None:
            skipped_too_recent += 1
            continue
        base_day, target_day, return_pct = result
        rows.append({
            "stock": row["stock"],
            "sector": row["sector"],
            "news_date": row["news_date"].date(),
            "title": row["title"],
            "description": row["description"],
            "link": row["link"],
            "base_trading_day": base_day.date(),
            "target_trading_day": target_day.date(),
            "return_pct": round(return_pct, 3),
        })

    matched = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    matched.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"total news rows: {len(news)}")
    print(f"matched: {len(matched)}")
    print(f"skipped (no price for ticker): {skipped_no_price}")
    print(f"skipped (too recent, no next trading day yet): {skipped_too_recent}")
    print(f"saved to {OUT_PATH}")
