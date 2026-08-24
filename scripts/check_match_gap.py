import pandas as pd
from collect_multi_sample import STOCKS

news = pd.read_csv("data/raw/news_backfill.csv")
news["news_date"] = pd.to_datetime(news["date"], format="mixed", utc=True).dt.tz_convert(
    "Asia/Seoul"
).dt.tz_localize(None).dt.normalize()

matched = pd.read_csv("data/processed/news_price_matched.csv")
matched_links = set(matched["link"])

news["matched"] = news["link"].isin(matched_links)
unmatched = news[~news["matched"]]

print("=== 미매칭 뉴스의 날짜 분포 (상위 10) ===")
print(unmatched["news_date"].dt.date.value_counts().sort_index(ascending=False).head(10))

print("\n=== 미매칭 뉴스의 종목별 건수 (상위 10) ===")
print(unmatched["stock"].value_counts().head(10))
