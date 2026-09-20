"""라벨 품질 개선 2단계: v8 분류기 예측과 Qwen(요약문) 라벨이 불일치하는 케이스를
naver.com이 아닌 링크에서 찾는다. find_ambiguous_cases.py의 비-naver 버전.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/klue-bert-qwen-sentiment-v8"
DATA_PATH = "data/processed/news_qwen_labeled.csv"
OUT_PATH = "data/raw/ambiguous_cases_nonnaver.csv"
VALID_LABELS = {"긍정", "부정", "중립"}
LABEL2ID = {"부정": 0, "중립": 1, "긍정": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["label"] = df["qwen_label"]
    df = df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
    df = df[~df["link"].str.contains("naver.com", na=False)].reset_index(drop=True)
    print(f"non-naver 링크 대상(유효라벨만): {len(df)}건")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to("cuda")
    model.eval()

    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    preds = []
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(df), BATCH):
            batch_text = df["text"].iloc[i:i+BATCH].tolist()
            inputs = tokenizer(batch_text, truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
            logits = model(**inputs).logits
            batch_preds = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(batch_preds)
            if i % 12800 == 0:
                print(f"{i}/{len(df)}")

    df["pred_label"] = [ID2LABEL[p] for p in preds]
    mismatch = df[df["label"] != df["pred_label"]].copy()
    print(f"\n불일치: {len(mismatch)}/{len(df)} ({len(mismatch)/len(df)*100:.1f}%)")
    print(mismatch.groupby(["label", "pred_label"]).size().sort_values(ascending=False).head(10))

    target = mismatch[
        ((mismatch["label"] == "중립") & (mismatch["pred_label"] == "긍정")) |
        ((mismatch["label"] == "긍정") & (mismatch["pred_label"] == "중립")) |
        ((mismatch["label"] == "중립") & (mismatch["pred_label"] == "부정")) |
        ((mismatch["label"] == "부정") & (mismatch["pred_label"] == "중립"))
    ]
    print(f"\n중립<->긍정/부정 혼동 (재라벨링 후보): {len(target)}건")

    target[["stock", "sector", "news_date", "title", "description", "link", "label", "pred_label"]].to_csv(
        OUT_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"\nsaved {len(target)} rows to {OUT_PATH}")
