"""Filter out '무관'/parse-failures from Qwen sentiment labels, then split into
train/val/test stratified by (stock, label) so every stock keeps some val/test
representation.

Previous version split by news_date bucket (70/15/15 of unique dates), which
made sense for the old price-based labels (avoid same-day market-wide-move
leakage across stocks). Qwen's tone label is a property of the text alone, not
of that day's market move, so date-leakage isn't a concern here -- and the
date-bucket approach left 10/18 stocks with zero val/test rows whenever their
articles' dates all happened to land in the train bucket (e.g. 카카오, whose
167 rows spanned only 2 unique dates). Stratifying by (stock, label) at the
row level guarantees every stock x label combination is represented in all
three splits, as long as it has enough rows.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/processed/news_qwen_labeled.csv"
OUT_PATH = "data/processed/news_sentiment_final.csv"
VALID_LABELS = {"긍정", "부정", "중립"}
RANDOM_STATE = 42

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    df["label"] = df["qwen_label"]
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    print(f"필터링 후: {len(df)}건 (무관/파싱실패 제외)")

    strat_key = df["stock"] + "_" + df["label"]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=strat_key, random_state=RANDOM_STATE
    )
    temp_strat_key = strat_key.loc[temp_df.index]
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_strat_key, random_state=RANDOM_STATE
    )

    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== split별 건수 ===")
    print(df["split"].value_counts())
    print("\n=== split별 라벨 분포 ===")
    print(df.groupby("split")["label"].value_counts(normalize=True).round(3))
    print("\n=== 종목별 split 커버리지 (0건인 칸이 없어야 함) ===")
    print(pd.crosstab(df["stock"], df["split"]))
    print(f"\nsaved to {OUT_PATH}")
