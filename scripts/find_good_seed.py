"""Search random seeds for a date group-split whose per-split label balance
is closest to the overall dataset's label balance. With only ~111 unique
dates, a single seed can land a lopsided period entirely in one split, so
picking the least-skewed seed is worth doing (this only looks at label
distribution, not any trained model's performance -- no leakage into
model selection).
"""
import sys
import random
import pandas as pd

IN_PATH = "data/processed/news_price_matched.csv"


def label(value, low, high):
    if value > high:
        return "상승"
    if value < low:
        return "하락"
    return "중립"


if __name__ == "__main__":
    value_col = sys.argv[1] if len(sys.argv) > 1 else "return_pct"
    low = float(sys.argv[2]) if len(sys.argv) > 2 else -1.0
    high = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    df = pd.read_csv(IN_PATH)
    df["label"] = df[value_col].apply(lambda v: label(v, low, high))
    overall = df["label"].value_counts(normalize=True)

    unique_dates = sorted(df["news_date"].unique())
    n = len(unique_dates)
    train_cut, val_cut = int(n * 0.70), int(n * 0.85)

    best_seed, best_score = None, float("inf")
    for seed in range(200):
        rng = random.Random(seed)
        shuffled = unique_dates[:]
        rng.shuffle(shuffled)
        train_dates = set(shuffled[:train_cut])
        val_dates = set(shuffled[train_cut:val_cut])
        test_dates = set(shuffled[val_cut:])

        split = df["news_date"].map(
            lambda d: "train" if d in train_dates else ("val" if d in val_dates else "test")
        )
        score = 0.0
        for s in ["train", "val", "test"]:
            dist = df.loc[split == s, "label"].value_counts(normalize=True)
            for lbl in ["상승", "중립", "하락"]:
                score += abs(dist.get(lbl, 0) - overall.get(lbl, 0))
        if score < best_score:
            best_score, best_seed = score, seed

    print(f"overall label dist:\n{overall.round(3)}")
    print(f"\nbest seed: {best_seed} (score={best_score:.3f})")
