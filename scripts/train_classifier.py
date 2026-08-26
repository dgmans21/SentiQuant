"""Phase 3: fine-tune KLUE-BERT for 상승/중립/하락 classification.

Input text = title + description. Label = the ±1.0%-threshold label from
Phase 2. Uses the existing train/val/test split (grouped by date, random
shuffle -- see label_and_split.py) as-is, no reshuffling here.
"""
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import sys

MODEL_NAME = "klue/bert-base"
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/processed/news_labeled.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "models/klue-bert-sentiment"
LABEL2ID = {"부정": 0, "중립": 1, "긍정": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 256


def load_split(df: pd.DataFrame, split: str) -> Dataset:
    sub = df[df["split"] == split].copy()
    sub["text"] = sub["title"].fillna("") + " " + sub["description"].fillna("")
    sub["labels"] = sub["label"].map(LABEL2ID)
    return Dataset.from_pandas(sub[["text", "labels"]], preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"total rows: {len(df)}")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {(df['split'] == s).sum()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    train_ds = load_split(df, "train").map(tokenize, batched=True)
    val_ds = load_split(df, "val").map(tokenize, batched=True)
    test_ds = load_split(df, "test").map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        bf16=True,
        logging_steps=20,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== test set 평가 ===")
    test_result = trainer.evaluate(test_ds)
    print(test_result)

    preds = trainer.predict(test_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = preds.label_ids
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    print("\n=== 혼동행렬 (행=실제, 열=예측) ===")
    print("       부정예측 중립예측 긍정예측")
    for i, row_name in enumerate(["부정실제", "중립실제", "긍정실제"]):
        print(row_name, cm[i])

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nsaved model to {OUT_DIR}")
