import pandas as pd

df = pd.read_csv("data/processed/news_price_matched.csv")
r = df["return_pct"]

print("=== 분위수(percentile) ===")
for p in [10, 20, 33, 40, 50, 60, 67, 80, 90]:
    print(f"{p}%: {r.quantile(p/100):.3f}")

print("\n=== 고정 임계값 후보별 클래스 분포 ===")
for t in [0.3, 0.5, 1.0, 1.5, 2.0]:
    down = (r < -t).sum()
    neutral = ((r >= -t) & (r <= t)).sum()
    up = (r > t).sum()
    total = len(r)
    print(f"±{t}%: 하락 {down} ({down/total:.1%}) / 중립 {neutral} ({neutral/total:.1%}) / 상승 {up} ({up/total:.1%})")

print("\n=== 균형 3분할(각 33%)을 위한 임계값 ===")
low = r.quantile(1/3)
high = r.quantile(2/3)
print(f"하락/중립 경계: {low:.3f}%, 중립/상승 경계: {high:.3f}%")
