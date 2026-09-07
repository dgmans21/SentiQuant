import pandas as pd

df = pd.read_csv("data/processed/rag_eval_judged.csv")
df["is_relevant"] = (df["relevant"] == "관련").astype(int)

print("=== 전체 판정 분포 ===")
print(df["relevant"].value_counts())

print("\n=== 방법별 Precision@5 (쿼리별 평균) ===")
per_query = df.groupby(["method", "query_idx"])["is_relevant"].mean()
summary = per_query.groupby("method").agg(["mean", "std", "count"])
print(summary)

print("\n=== 방법별 순위(rank)별 relevant 비율 ===")
print(df.groupby(["method", "rank"])["is_relevant"].mean().unstack("method"))

n_queries = df["query_idx"].nunique()
print(f"\n쿼리 수: {n_queries}")
