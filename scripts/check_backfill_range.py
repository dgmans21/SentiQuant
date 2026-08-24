import pandas as pd

df = pd.read_csv("data/raw/news_backfill.csv")
df["date"] = pd.to_datetime(df["date"], format="mixed", utc=True)
summary = df.groupby("stock")["date"].agg(["count", "min", "max"])
summary.to_csv("data/raw/backfill_range_summary.csv", encoding="utf-8-sig")
print("전체 범위:", df["date"].min(), "~", df["date"].max())
