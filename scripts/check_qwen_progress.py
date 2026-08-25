import pandas as pd

df = pd.read_csv("data/processed/news_qwen_labeled.csv")
print(f"total labeled: {len(df)}")
print("\n=== qwen_label 분포 ===")
print(df["qwen_label"].value_counts())
print(f"\n파싱실패: {(df['qwen_label'] == '파싱실패').sum()}건")
