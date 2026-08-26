"""Filter out '무관'/parse-failures from Qwen sentiment labels, then group-shuffle
split by date (same leakage-avoidance logic as label_and_split.py, but the
label here is Qwen's own categorical judgment -- no threshold needed).
"""
import random
import pandas as pd

IN_PATH = "data/processed/news_qwen_labeled.csv"
OUT_PATH = "data/processed/news_sentiment_final.csv"
VALID_LABELS = {"긍정", "부정", "중립"}


def find_best_seed(df: pd.DataFrame, n_trials: int = 200):
    overall = df["label"].value_counts(normalize=True)
    unique_dates = sorted(df["news_date"].unique())
    n = len(unique_dates)
    train_cut, val_cut = int(n * 0.70), int(n * 0.85)

    best_seed, best_score = None, float("inf")
    for seed in range(n_trials):
        rng = random.Random(seed)
        shuffled = unique_dates[:]
        rng.shuffle(shuffled)
        train_dates = set(shuffled[:train_cut])
        val_dates = set(shuffled[train_cut:val_cut])

        split = df["news_date"].map(
            lambda d: "train" if d in train_dates else ("val" if d in val_dates else "test")
        )
        score = 0.0
        for s in ["train", "val", "test"]:
            dist = df.loc[split == s, "label"].value_counts(normalize=True)
            for lbl in VALID_LABELS:
                score += abs(dist.get(lbl, 0) - overall.get(lbl, 0))
        if score < best_score:
            best_score, best_seed = score, seed
    return best_seed, best_score


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["label"] = df["qwen_label"]
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    print(f"필터링 후: {len(df)}건 (무관/파싱실패 제외)")

    seed, score = find_best_seed(df)
    print(f"best seed: {seed} (score={score:.3f})")

    unique_dates = sorted(df["news_date"].unique())
    rng = random.Random(seed)
    shuffled = unique_dates[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_cut, val_cut = int(n * 0.70), int(n * 0.85)
    train_dates = set(shuffled[:train_cut])
    val_dates = set(shuffled[train_cut:val_cut])

    def assign_split(d):
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "val"
        return "test"

    df["split"] = df["news_date"].apply(assign_split)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== split별 건수 ===")
    print(df["split"].value_counts())
    print("\n=== split별 라벨 분포 ===")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))
    print(f"\nsaved to {OUT_PATH}")
