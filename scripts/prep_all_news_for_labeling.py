"""Build a labeling input from ALL raw news (backfill + curated), independent
of price matching. Qwen sentiment labeling only needs title/description text,
so it doesn't need to wait for next-trading-day price data to exist.

label_qwen_full.py already resumes by link (skips anything already in
news_qwen_labeled.csv), so pointing it at this full raw pool naturally
processes only the not-yet-labeled remainder -- currently the ~5,835 items
still "too recent" for price matching.
"""
import pandas as pd
from match_news_price import load_news

OUT_PATH = "data/processed/news_all_for_labeling.csv"

if __name__ == "__main__":
    df = load_news()
    df["news_date"] = df["date"]
    df["label"] = "미매칭"  # placeholder; not used by the labeling prompt
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"total rows: {len(df)}")
    print(f"saved to {OUT_PATH}")
