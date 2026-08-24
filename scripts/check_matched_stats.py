import pandas as pd

df = pd.read_csv("data/processed/news_price_matched.csv")
print("=== return_pct 기술통계 ===")
print(df["return_pct"].describe())

print("\n=== 종목별 매칭 건수 ===")
print(df["stock"].value_counts())

print("\n=== news_date 범위 ===")
print(df["news_date"].min(), "~", df["news_date"].max())
