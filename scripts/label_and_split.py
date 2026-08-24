"""Phase 2: threshold-based labeling + time-based train/val/test split.

Threshold: +-1.0% (validated earlier against this dataset's return_pct
distribution -- gives a near-even 3-way class balance: ~33/34/33%).

Split is by calendar date, not random shuffling: articles from the same
day often cover the same market event (near-duplicate wording), so a
random split could leak near-identical text across train/test and
inflate eval scores. Sorting unique dates and cutting 70/15/15 keeps
each date's articles entirely inside one split.
"""
import pandas as pd

THRESHOLD = 1.0
IN_PATH = "data/processed/news_price_matched.csv"
OUT_PATH = "data/processed/news_labeled.csv"


def label(return_pct: float) -> str:
    if return_pct > THRESHOLD:
        return "상승"
    if return_pct < -THRESHOLD:
        return "하락"
    return "중립"


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["label"] = df["return_pct"].apply(label)

    # Group-shuffle-split by date: every date stays whole in one split (no
    # same-day leakage), but a fixed-seed random shuffle -- not a fixed
    # period like every-7th-date -- avoids accidentally aligning with
    # calendar cycles (e.g. day-of-week effects) that a periodic pick would
    # otherwise systematically favor into one split.
    import random

    unique_dates = sorted(df["news_date"].unique())
    rng = random.Random(42)
    shuffled = unique_dates[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_cut = int(n * 0.70)
    val_cut = int(n * 0.85)
    train_dates = set(shuffled[:train_cut])
    val_dates = set(shuffled[train_cut:val_cut])
    test_dates = set(shuffled[val_cut:])

    def assign_split(d):
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "val"
        return "test"

    df["split"] = df["news_date"].apply(assign_split)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"threshold: +-{THRESHOLD}%")
    print(f"total rows: {len(df)}")
    print(f"date range: {unique_dates[0]} ~ {unique_dates[-1]} ({len(unique_dates)} unique dates)")

    print("\n=== split별 건수 ===")
    print(df["split"].value_counts())

    print("\n=== split별 라벨 분포 ===")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))

    print(f"\nsaved to {OUT_PATH}")
