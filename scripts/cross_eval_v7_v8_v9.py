"""v7/v8/v9를 같은 test 행(v9 test split), 같은 정답지로 교차 평가한다.
1) 이전 라벨 기준 / 새 라벨 기준 전체  2) 라벨이 안 바뀐 행만  3) 바뀐 행에서 어느 쪽 라벨에 동의하는지
+ v9 vs v7 짝지은(McNemar) 비교. 추론만 수행(학습 없음).
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OLD_PATH = "data/processed/news_sentiment_final.csv"
NEW_PATH = "data/processed/news_sentiment_v9.csv"
OUT_PATH = "data/raw/cross_eval_preds.csv"
MODELS = {"v7": "models/klue-bert-qwen-sentiment-v7",
          "v8": "models/klue-bert-qwen-sentiment-v8",
          "v9": "models/klue-bert-qwen-sentiment-v9"}
LABEL2ID = {"부정": 0, "중립": 1, "긍정": 2}


def predict(model_dir, texts, batch=64):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cuda").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to("cuda")
            out.extend(model(**enc).logits.argmax(-1).cpu().tolist())
    del model
    torch.cuda.empty_cache()
    return np.array(out)


def macro_f1(y, p):
    fs = []
    for c in (0, 1, 2):
        tp = ((p == c) & (y == c)).sum()
        fp = ((p == c) & (y != c)).sum()
        fn = ((p != c) & (y == c)).sum()
        fs.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(fs))


def se(acc, n):
    return math.sqrt(acc * (1 - acc) / n) * 100


if __name__ == "__main__":
    new = pd.read_csv(NEW_PATH, encoding="utf-8-sig")
    old = pd.read_csv(OLD_PATH, encoding="utf-8-sig")[["link", "label"]].rename(columns={"label": "old"})
    df = new[new["split"] == "test"].rename(columns={"label": "new"}).merge(old, on="link", how="left")
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    y_new = df["new"].map(LABEL2ID).values
    y_old = df["old"].map(LABEL2ID).values
    changed = y_new != y_old
    print(f"test rows {len(df)} | changed {int(changed.sum())} | unchanged {int((~changed).sum())}")

    preds = {}
    for name, path in MODELS.items():
        preds[name] = predict(path, df["text"].tolist())
        df[f"pred_{name}"] = preds[name]
        print(f"{name} done")
    df.drop(columns=["text"]).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== accuracy / macroF1 (SE in pp) ===")
    print(f"{'model':5} {'old-key acc':>12} {'new-key acc':>12} {'new-key F1':>11} {'unchanged acc':>14} {'unch F1':>8}")
    for name in MODELS:
        p = preds[name]
        a_old = (p == y_old).mean()
        a_new = (p == y_new).mean()
        u = ~changed
        a_u = (p[u] == y_new[u]).mean()
        print(f"{name:5} {a_old*100:8.2f}%    {a_new*100:8.2f}%    {macro_f1(y_new, p)*100:8.2f}%    "
              f"{a_u*100:8.2f}%(+-{se(a_u, u.sum()):.2f}) {macro_f1(y_new[u], p[u])*100:7.2f}%")

    print("\n=== on the changed rows only (n=%d): agree with OLD label / NEW label / neither ===" % changed.sum())
    for name in MODELS:
        p = preds[name][changed]
        ao = (p == y_old[changed]).mean() * 100
        an = (p == y_new[changed]).mean() * 100
        print(f"{name:5} old {ao:5.1f}%  new {an:5.1f}%  neither {100-ao-an:5.1f}%")

    def mcnemar(a, b, mask, key, label):
        ra = preds[a][mask] == key[mask]
        rb = preds[b][mask] == key[mask]
        c = int((~ra & rb).sum())
        d = int((ra & ~rb).sum())
        z = (c - d) / math.sqrt(c + d) if c + d else 0.0
        print(f"{label}: {b} better on {c}, {a} better on {d}, z={z:.1f}")

    print("\n=== paired comparison (McNemar-style) ===")
    mcnemar("v7", "v9", ~changed, y_new, "unchanged rows, v7 vs v9")
    mcnemar("v7", "v8", ~changed, y_new, "unchanged rows, v7 vs v8")
    mcnemar("v8", "v9", ~changed, y_new, "unchanged rows, v8 vs v9")
    mcnemar("v7", "v9", np.ones(len(df), bool), y_new, "all rows new-key, v7 vs v9")
    mcnemar("v7", "v9", np.ones(len(df), bool), y_old, "all rows old-key, v7 vs v9")
    print(f"\nsaved predictions to {OUT_PATH}")
