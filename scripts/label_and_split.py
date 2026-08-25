"""Phase 2: threshold-based labeling + group-shuffle train/val/test split.

Split: dates are randomly shuffled (fixed seed) then cut 70/15/15, keeping
each date whole in one split (no same-day leakage from near-duplicate
same-event articles landing in both train and test). A periodic pick
(e.g. every 7th date) was tried first and rejected -- it aligned with
day-of-week effects and skewed label distributions per split.

Usage:
    python scripts/label_and_split.py <value_column> <low_threshold> <high_threshold> <out_path> [seed]

Examples:
    python scripts/label_and_split.py return_pct -1.0 1.0 data/processed/news_labeled.csv 42
    python scripts/label_and_split.py excess_return_pct -3.561 1.656 data/processed/news_labeled_excess.csv 184

Seed picked via find_good_seed.py -- with only ~111 unique dates, a random
group-split's per-split label balance varies a lot by seed, so the seed is
chosen to minimize skew against the overall label distribution (based only
on label counts, not any trained model, so no leakage into model selection).
"""
import sys
import random
import pandas as pd

IN_PATH = "data/processed/news_price_matched.csv"


def label(value: float, low: float, high: float) -> str:
    if value > high:
        return "상승"
    if value < low:
        return "하락"
    return "중립"


if __name__ == "__main__":
    value_col = sys.argv[1] if len(sys.argv) > 1 else "return_pct"
    low = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    high = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    out_path = sys.argv[4] if len(sys.argv) > 4 else "data/processed/news_labeled.csv"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    df = pd.read_csv(IN_PATH)
    df["label"] = df[value_col].apply(lambda v: label(v, low, high))

    unique_dates = sorted(df["news_date"].unique())
    rng = random.Random(seed)
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
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"value_col: {value_col}, threshold: [{low}, {high}]")
    print(f"total rows: {len(df)}")
    print(f"date range: {unique_dates[0]} ~ {unique_dates[-1]} ({len(unique_dates)} unique dates)")

    print("\n=== split별 건수 ===")
    print(df["split"].value_counts())

    print("\n=== split별 라벨 분포 ===")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))

    print(f"\nsaved to {out_path}")
